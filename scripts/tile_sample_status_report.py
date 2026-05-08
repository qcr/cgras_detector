#!/usr/bin/env python3
"""
Tile Sample Status Report Generator

Queries the CGRAS detector SQLite database and produces a PNG table image
showing status breakdown of the most recent N tile samples, with failed
samples listed by type (Reconstruction failure, LocTile failure, Other).

Usage:
    python scripts/tile_sample_status_report.py
    python scripts/tile_sample_status_report.py --n 264 --db ~/cgras_data/database/detector.db --out report.png
"""

import sqlite3
import os
import argparse
import math
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DB  = os.path.expanduser('~/cgras_data/database/detector.db')
DEFAULT_N   = 264
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'tile_sample_status_report.png')

# ---------------------------------------------------------------------------
# Status mapping
# ---------------------------------------------------------------------------
STATUS_MAP = {0: 'QUEUED', 1: 'DONE', 2: 'FLAGGED', 4: 'REJECTED'}

PALETTE = {
    'DONE':           '#27ae60',
    'QUEUED':         '#2980b9',
    'FLAGGED':        '#e67e22',
    'REJECTED':       '#c0392b',
    'Reconstruction': '#922b21',
    'LocTile':        '#6c3483',
    'Other':          '#616a6b',
    'header':         '#1a252f',
}

# ---------------------------------------------------------------------------
# Failure classification keywords (matched against lower-cased remarks)
# ---------------------------------------------------------------------------
RECO_KEYWORDS = [
    'wayward homography',
    'cannot combine',
    'cannot merge adjacent',
    'cannot obtain camera transform',
    'lack sufficient features',
    'lack sufficient between row',
    'cannot find warp rois',
    'aspect ratio of one or more rois',
    'roi corners not regularly',
    'unable to obtain camera transforms for every image',
]

LOCTILE_KEYWORDS = [
    'not all four corners',
    'angle of rotation',
    'aspect ratio of the detected frame',
]


def classify_failure(remarks: str) -> str:
    if not remarks:
        return 'Other'
    r = remarks.lower()
    for kw in RECO_KEYWORDS:
        if kw in r:
            return 'Reconstruction'
    for kw in LOCTILE_KEYWORDS:
        if kw in r:
            return 'LocTile'
    return 'Other'


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def fetch_samples(db_path: str, n: int) -> list:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        'SELECT id, tile_id, batch_id, status, remarks, modify_time '
        'FROM tile_sample ORDER BY modify_time DESC LIMIT ?', (n,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def build_report(samples: list) -> tuple:
    buckets  = {'DONE': [], 'QUEUED': [], 'FLAGGED': [], 'REJECTED': []}
    failures = {'Reconstruction': [], 'LocTile': [], 'Other': []}
    for s in samples:
        status = STATUS_MAP.get(s['status'], 'UNKNOWN')
        if status in buckets:
            buckets[status].append(s['id'])
        if status in ('REJECTED', 'FLAGGED'):
            ftype = classify_failure(s['remarks'] or '')
            failures[ftype].append((s['id'], s['remarks'] or ''))
    return buckets, failures


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def hex_rgba(hex_colour: str, alpha: float = 0.20):
    h = hex_colour.lstrip('#')
    r, g, b = (int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b, alpha)


def cols_of(items: list, n_cols: int) -> list:
    per_col = math.ceil(len(items) / n_cols) if items else 1
    return [items[i * per_col:(i + 1) * per_col] for i in range(n_cols)]


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def draw_summary_table(ax, buckets, failures, n_actual):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    total = n_actual

    data_rows = [
        # (label, count, colour_key, is_sub, note)
        ('DONE',
         len(buckets['DONE']), 'DONE', False,
         'Successfully processed'),
        ('QUEUED',
         len(buckets['QUEUED']), 'QUEUED', False,
         'Awaiting processing'),
        ('FLAGGED',
         len(buckets['FLAGGED']), 'FLAGGED', False,
         'Aborted / recoverable error'),
        ('REJECTED  (total)',
         len(buckets['REJECTED']), 'REJECTED', False,
         'Pipeline failure — rejected by system'),
        ('  ↳ Reconstruction failure',
         len(failures['Reconstruction']), 'Reconstruction', True,
         'Homography / feature matching error (RECO stage)'),
        ('  ↳ LocTile failure',
         len(failures['LocTile']), 'LocTile', True,
         'Tile corner not detected correctly (LOCTILE stage)'),
        ('  ↳ Other failure',
         len(failures['Other']), 'Other', True,
         'Input data / OS / YOLO model error'),
    ]

    col_labels = ['Status', 'Count', '% of Total', 'Note']
    col_x      = [0.00, 0.45, 0.57, 0.71]
    col_widths = [0.45, 0.12, 0.14, 0.29]

    n_rows = len(data_rows)              # 7 data rows
    row_h  = 1.0 / (n_rows + 1)         # +1 for the header row

    # Header row occupies the top slot: y from (1 - row_h) to 1.0
    hdr_y = 1.0 - row_h
    for cx, cw, label in zip(col_x, col_widths, col_labels):
        ax.add_patch(FancyBboxPatch(
            (cx, hdr_y), cw - 0.004, row_h,
            boxstyle='square,pad=0', fc=PALETTE['header'], ec='none', zorder=2))
        ax.text(cx + 0.01, hdr_y + row_h / 2, label,
                va='center', ha='left', fontsize=11,
                fontweight='bold', color='white', zorder=3)

    # Data rows occupy slots below the header
    # Slot i (0-indexed from top): y from (1 - (i+2)*row_h) to (1 - (i+1)*row_h)
    for i, (label, count, colour_key, is_sub, note) in enumerate(data_rows):
        y_bot = 1.0 - (i + 2) * row_h   # bottom edge of this row
        pct   = 100.0 * count / total if total else 0.0
        alpha = 0.10 if is_sub else 0.22
        bg    = hex_rgba(PALETTE[colour_key], alpha)

        for cx, cw in zip(col_x, col_widths):
            ax.add_patch(FancyBboxPatch(
                (cx, y_bot), cw - 0.004, row_h,
                boxstyle='square,pad=0', fc=bg, ec='none', zorder=2))

        values  = [label, str(count), f'{pct:.1f}%', note]
        fsizes  = [10.5, 10.5, 10.5, 9.5]
        weights = ['normal' if is_sub else 'bold', 'bold', 'bold', 'normal']

        for cx, val, fs, fw in zip(col_x, values, fsizes, weights):
            ax.text(cx + 0.01, y_bot + row_h / 2, val,
                    va='center', ha='left', fontsize=fs,
                    fontweight=fw, color='#1a252f', zorder=3)

        # Thin separator line below each row
        ax.axhline(y=y_bot, xmin=0, xmax=1,
                   color='#bdc3c7', linewidth=0.5, zorder=1)


# ---------------------------------------------------------------------------
# Failure detail section
# ---------------------------------------------------------------------------
def draw_failure_section(ax, title, colour, items):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Coloured title bar across the top
    ax.add_patch(FancyBboxPatch(
        (0, 0.88), 1.0, 0.12,
        boxstyle='square,pad=0', fc=hex_rgba(colour, 0.15), ec='none'))
    ax.axhline(y=0.88, color=colour, linewidth=2.5)
    ax.text(0.01, 0.94, f'{title}  ({len(items)} samples)',
            va='center', ha='left',
            fontsize=12, fontweight='bold', color=colour)

    if not items:
        ax.text(0.5, 0.44, 'None', va='center', ha='center',
                fontsize=10, color='#95a5a6', style='italic')
        return

    n_cols   = 3
    cols     = cols_of(items, n_cols)
    col_w    = 1.0 / n_cols
    content_h = 0.86              # height available below title bar (0 to 0.88)

    for ci, col_items in enumerate(cols):
        if not col_items:
            continue
        x = ci * col_w + 0.01
        # Space rows evenly in the available content height
        slot = content_h / max(len(col_items), 1)
        id_frac   = 0.52   # fraction of slot for the ID line
        rmk_frac  = 0.40   # fraction for the remark line

        for ri, (tile_id, remark) in enumerate(col_items):
            y_slot_top = 0.86 - ri * slot
            y_id  = y_slot_top - slot * (1 - id_frac) / 2
            y_rmk = y_slot_top - slot * id_frac - slot * (1 - rmk_frac) / 2

            ax.text(x, y_id, tile_id,
                    va='top', ha='left', fontsize=7.5,
                    fontfamily='monospace', color='#1a252f',
                    fontweight='bold', clip_on=True)

            short = (remark[:75] + '…') if len(remark) > 75 else remark
            ax.text(x + 0.005, y_rmk, short,
                    va='top', ha='left', fontsize=6.8,
                    fontfamily='monospace', color='#7f8c8d', clip_on=True)


# ---------------------------------------------------------------------------
# Main figure assembly
# ---------------------------------------------------------------------------
def generate_report(buckets, failures, n_queried, n_actual, output_path):

    fig = plt.figure(figsize=(18, 14), facecolor='white')

    # ------------------------------------------------------------------
    # Layout (all in figure-normalised coords, y=0 at bottom):
    #   0.965 – 1.000  title block        (fig.text only, no axes)
    #   0.770 – 0.955  summary table      axes
    #   0.510 – 0.755  reconstruction     axes
    #   0.270 – 0.500  loctile            axes
    #   0.030 – 0.260  other              axes
    #   0.000 – 0.025  footer             (fig.text only)
    # ------------------------------------------------------------------

    # Title
    fig.text(0.5, 0.985,
             'CGRAS — Tile Sample Processing Status Report',
             ha='center', va='top',
             fontsize=19, fontweight='bold', color=PALETTE['header'])
    fig.text(0.5, 0.968,
             f'Most recent {n_queried} samples requested  |  '
             f'{n_actual} records retrieved  |  '
             f'Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}',
             ha='center', va='top', fontsize=10, color='#7f8c8d')

    # Summary table
    ax_sum = fig.add_axes([0.04, 0.770, 0.92, 0.185])
    draw_summary_table(ax_sum, buckets, failures, n_actual)

    # Failure sections
    sections = [
        ('Reconstruction Failures', 'Reconstruction', PALETTE['Reconstruction'], [0.04, 0.510, 0.92, 0.245]),
        ('LocTile Failures',        'LocTile',        PALETTE['LocTile'],        [0.04, 0.270, 0.92, 0.235]),
        ('Other Failures',          'Other',           PALETTE['Other'],          [0.04, 0.030, 0.92, 0.235]),
    ]
    for title, ftype, colour, rect in sections:
        ax = fig.add_axes(rect)
        draw_failure_section(ax, title, colour, failures[ftype])

    # Footer
    fig.text(0.5, 0.008,
             'Status codes: DONE=1  QUEUED=0  FLAGGED=2  REJECTED=4  '
             '|  Failure type classified from remarks column in tile_sample table',
             ha='center', va='bottom', fontsize=8, color='#aab7b8')

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Report saved → {output_path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Generate CGRAS tile sample status report image')
    parser.add_argument('--db',  default=DEFAULT_DB,
                        help='Path to detector.db')
    parser.add_argument('--n',   default=DEFAULT_N, type=int,
                        help='Number of most recent samples to inspect')
    parser.add_argument('--out', default=DEFAULT_OUT,
                        help='Output PNG path')
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f'ERROR: database not found at {args.db}')
        return

    print(f'Querying {args.n} most recent samples from {args.db} …')
    samples = fetch_samples(args.db, args.n)
    print(f'  Retrieved {len(samples)} records')

    buckets, failures = build_report(samples)
    print(f'  DONE:    {len(buckets["DONE"])}')
    print(f'  QUEUED:  {len(buckets["QUEUED"])}')
    print(f'  FLAGGED: {len(buckets["FLAGGED"])}')
    print(f'  REJECTED:{len(buckets["REJECTED"])}  '
          f'(Reco:{len(failures["Reconstruction"])}  '
          f'LocTile:{len(failures["LocTile"])}  '
          f'Other:{len(failures["Other"])})')

    generate_report(buckets, failures, args.n, len(samples), args.out)


if __name__ == '__main__':
    main()
