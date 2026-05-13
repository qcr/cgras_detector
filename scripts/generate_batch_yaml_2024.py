#!/usr/bin/env python3
"""
Generate CCVS tile sample YAML files for the 2024Nov and 2024Oct datasets.

For each subfolder {system}_T{tile}_{date} in the sorted source directories,
one YAML file is produced following the format used in ~/cgras_data/batch_yaml_2025Dec.

Image grid indexing follows CCVSTileConfigGenerator.py lines 226-228:
    x = idx % 4,  y = idx // 4   (4 columns, 6 rows = 24 images)
"""

import re
import sys
from pathlib import Path
import yaml

# ---------------------------------------------------------------------------
# Season configuration
# ---------------------------------------------------------------------------
SEASON_CONFIG = {
    '2024Nov': {
        'season_tag': '2024Nov',
        'settling_time': '2024-11-25',
        'spawning_time': '2024-11-11',
        'source_root': Path(
            '/media/dtsai/DATA/cgras_datasets/'
            'cgras_2024_aims_camera_trolley_fixed_filenames_season_tile_sorted/2024_nov'
        ),
        'output_dir': Path.home() / 'cgras_data/batch_yaml_2024Nov',
        'container_source_parent': '/home/qcr/cgras_data/Source/2024_nov',
    },
    '2024Oct': {
        'season_tag': '2024Oct',
        'settling_time': '2024-10-30',
        'spawning_time': '2024-10-20',
        'source_root': Path(
            '/media/dtsai/DATA/cgras_datasets/'
            'cgras_2024_aims_camera_trolley_fixed_filenames_season_tile_sorted/2024_oct'
        ),
        'output_dir': Path.home() / 'cgras_data/batch_yaml_2024Oct',
        'container_source_parent': '/home/qcr/cgras_data/Source/2024_oct',
    },
}

SPECIES_MAP = {
    'amag': 'acropora kenti',
    'pdae': 'platygyra deadalae',
    'amil': 'acropora millepora',
    'maeq': 'montipora aequituberculata',
}

FOLDER_RE = re.compile(r'^(?P<system>.+)_T(?P<tile>\d+)_(?P<date>\d{8})$')
FILENAME_RE = re.compile(
    r'^CGRAS_(?P<species>[A-Za-z]+)_[^_]+_\d{8}_w\d+_T\d+_\d+\.jpg$',
    re.IGNORECASE,
)

EXPECTED_IMAGE_COUNT = 24


# ---------------------------------------------------------------------------
# YAML helpers — ensure date/time strings are quoted in output
# ---------------------------------------------------------------------------
class _QuotedStr(str):
    pass


def _quoted_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")


yaml.add_representer(_QuotedStr, _quoted_presenter)


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------
def generate_yaml_for_season(config: dict) -> int:
    season_tag = config['season_tag']
    source_root: Path = config['source_root']
    output_dir: Path = config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    subfolders = sorted(d for d in source_root.iterdir() if d.is_dir())
    generated = skipped = warned = 0

    for subfolder in subfolders:
        m = FOLDER_RE.match(subfolder.name)
        if not m:
            print(f'  SKIP (unrecognised folder name): {subfolder.name}')
            skipped += 1
            continue

        tile_str = m.group('tile')   # e.g. '09'
        date_str = m.group('date')   # e.g. '20241126'

        images = sorted(subfolder.glob('*.jpg'))
        if len(images) != EXPECTED_IMAGE_COUNT:
            print(f'  WARN ({len(images)} images, expected {EXPECTED_IMAGE_COUNT}): {subfolder.name} — skipping')
            warned += 1
            continue

        fm = FILENAME_RE.match(images[0].name)
        if not fm:
            print(f'  SKIP (cannot parse image filename): {images[0].name}')
            skipped += 1
            continue

        species_key = fm.group('species').lower()
        species_full = SPECIES_MAP.get(species_key)
        if species_full is None:
            print(f'  SKIP (unknown species {species_key!r}): {subfolder.name}')
            skipped += 1
            continue

        tile_id    = f'{season_tag}-T{tile_str}'
        # batch_id uses 6-digit YYMMDD + 000000
        batch_id   = f'CG0-{date_str[2:]}000000'
        batch_time = _QuotedStr(f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} 00:00:00')

        yaml_data = {
            'tile_id':                   tile_id,
            'species':                   species_full,
            'settling_time':             _QuotedStr(config['settling_time']),
            'spawning_time':             _QuotedStr(config['spawning_time']),
            'season':                    season_tag,
            'num_tabs':                  [20, 20],
            'frame_size':                [294, 294],
            'batch_id':                  batch_id,
            'batch_time':                batch_time,
            'importer_id':               'AIMS',
            'operator':                  'camera_trolley',
            'image_files_parent_folder': f"{config['container_source_parent']}/{subfolder.name}",
            'images': [
                {'x': idx % 4, 'y': idx // 4, 'file': img.name}
                for idx, img in enumerate(images)
            ],
        }

        output_file = output_dir / f'Tile_{tile_id}_CG0-{date_str}_manual.yaml'
        with open(output_file, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        generated += 1

    print(f'  Generated: {generated}  Skipped: {skipped}  Warned (bad count): {warned}')
    return generated


def main():
    total = 0
    for season_key, config in SEASON_CONFIG.items():
        print(f'\n=== {season_key} ===')
        if not config['source_root'].exists():
            print(f'  SOURCE NOT FOUND: {config["source_root"]}')
            continue
        total += generate_yaml_for_season(config)

    print(f'\nDone. Total YAML files generated: {total}')


if __name__ == '__main__':
    main()
