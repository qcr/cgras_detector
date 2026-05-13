#!/usr/bin/env python3
"""
Sort CGRAS coral season images into per-tile subfolders following the
convention used in ~/cgras_data/Source/2024:
    {system_abbrev}_T{tile}_{date}/

The system abbreviation strips trailing lowercase letters from the system ID:
    MIS5b -> MIS5,  MIS1a -> MIS1,  MIS6b -> MIS6,  Room9 -> Room9

Parsed from filename:  CGRAS_{species}_{system}_{date}_{week}_T{tile}_{imgnum}.jpg
Output subfolder name: {system_abbrev}_T{tile}_{date}

Usage:
    python sort_dataset_2024_nov.py                          # copy 2024_nov (default)
    python sort_dataset_2024_nov.py --dry-run                # preview without copying
    python sort_dataset_2024_nov.py --source /path/to/src --dest /path/to/dst
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

DEFAULT_BASE = Path(
    '/media/dtsai/DATA/cgras_datasets/'
    'cgras_2024_aims_camera_trolley_fixed_filenames/'
    'cgras_2024_aims_camera_trolley'
)
DEFAULT_DEST_BASE = Path(
    '/media/dtsai/DATA/cgras_datasets/'
    'cgras_2024_aims_camera_trolley_fixed_filenames_season_tile_sorted'
)

FILENAME_RE = re.compile(
    r'^CGRAS_(?P<species>[^_]+)_(?P<system>[^_]+)_(?P<date>\d{8})'
    r'_(?P<week>w\d+)_(?P<tile>T\d+)_(?P<imgnum>\d+)\.jpg$',
    re.IGNORECASE,
)


def system_abbrev(system: str) -> str:
    """Strip trailing lowercase letters: MIS5b -> MIS5, Room9 -> Room9."""
    return re.sub(r'[a-z]+$', '', system)


def main():
    parser = argparse.ArgumentParser(
        description='Sort CGRAS coral images into per-tile subfolders.'
    )
    parser.add_argument(
        '--source', type=Path,
        default=DEFAULT_BASE / 'corals_spawned_2024_nov',
        help='Source directory containing .jpg files (searched recursively)',
    )
    parser.add_argument(
        '--dest', type=Path,
        default=DEFAULT_DEST_BASE / '2024_nov',
        help='Destination root directory for sorted subfolders',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Preview destination paths without copying any files',
    )
    args = parser.parse_args()

    source_root = args.source
    dest_root = args.dest

    if not source_root.exists():
        print(f'ERROR: source directory not found:\n  {source_root}', file=sys.stderr)
        sys.exit(1)

    jpg_files = sorted(source_root.rglob('*.jpg'))
    total = len(jpg_files)
    if total == 0:
        print(f'No .jpg files found under {source_root}')
        return

    print(f'Source: {source_root}')
    print(f'Found {total} .jpg files')
    if args.dry_run:
        print('DRY RUN — no files will be copied\n')
    else:
        dest_root.mkdir(parents=True, exist_ok=True)
        print(f'Destination: {dest_root}\n')

    copied = skipped = errors = 0
    for i, src in enumerate(jpg_files, 1):
        if i % 200 == 0 or i == total:
            print(f'  [{i}/{total}] copied={copied} skipped={skipped} errors={errors}', end='\r')

        m = FILENAME_RE.match(src.name)
        if not m:
            print(f'  SKIP (unrecognised filename): {src}')
            skipped += 1
            continue

        abbrev = system_abbrev(m.group('system'))
        folder_name = f"{abbrev}_{m.group('tile')}_{m.group('date')}"
        dest_dir = dest_root / folder_name
        dest = dest_dir / src.name

        if dest.exists():
            skipped += 1
            continue

        if args.dry_run:
            rel = src.relative_to(source_root)
            print(f'  {rel}')
            print(f'    -> {folder_name}/{src.name}')
            copied += 1
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dest)
            copied += 1
        except Exception as e:
            print(f'  ERROR copying {src}: {e}')
            errors += 1

    print()  # clear progress line
    action = 'Would copy' if args.dry_run else 'Copied'
    print(f'{action} {copied} files, skipped {skipped} already-existing, {errors} errors')


if __name__ == '__main__':
    main()
