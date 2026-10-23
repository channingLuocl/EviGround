"""Download and prepare the fixed NR3D size selection; no inference API calls."""
import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from eviground.cli.prepare_scene import prepare
from eviground.rendering.run import ROOT

SUFFIXES = ('.txt', '.aggregation.json', '_vh_clean_2.0.010000.segs.json',
            '_vh_clean_2.ply')


def download(url, destination):
    if destination.exists() and destination.stat().st_size:
        return
    partial = destination.with_name(destination.name + '.part')
    with urlopen(url, timeout=120) as response, partial.open('wb') as output:
        shutil.copyfileobj(response, output)
        expected = response.headers.get('Content-Length')
    if expected is not None and partial.stat().st_size != int(expected):
        raise ValueError(f'Incomplete download: {destination.name}')
    partial.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / 'data/validation/nr3d_size_cases.json')
    parser.add_argument('--downloads', type=Path, default=ROOT / 'local_outputs/downloads')
    parser.add_argument('--output', type=Path, default=ROOT / 'local_outputs')
    parser.add_argument('--base-url', default='https://kaldir.vc.in.tum.de/scannet/v2/scans')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    scenes = sorted({row['scene'] for row in manifest['rows']})
    args.downloads.mkdir(parents=True, exist_ok=True)
    # Check the existing OLT before downloading any point clouds.
    with ZipFile(args.downloads / 'olt.zip') as archive:
        for scene in scenes:
            for kind in ('gt', 'pred'):
                archive.getinfo(f'seeground_object_lookup_table/nr3d/{kind}/{scene}.json')
    for scene in scenes:
        print(f'Downloading {scene}', flush=True)
        def fetch(suffix):
            name = scene + suffix
            download(f'{args.base_url.rstrip("/")}/{scene}/{name}', args.downloads / name)
            print(f'  ready: {name}', flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch, SUFFIXES))
        output = args.output / scene
        complete = all((output / name).exists() for name in (
            'scene.npz', 'objects.json', 'objects_gt.json', 'scene_instances.json',
            'overview.png', 'provenance.json'))
        if complete:
            provenance = json.loads((output / 'provenance.json').read_text())
            if provenance.get('scene') != scene or provenance.get('max_bbox_coordinate_error_m', 1) > .002:
                raise ValueError(f'Existing scene has invalid provenance: {output}')
            print(f'  using previously prepared scene: {scene}', flush=True)
        else:
            prepare(args.downloads, scene, output)
        objects = json.loads((output / 'objects_gt.json').read_text())
        known = {obj['bbox_id'] for obj in objects}
        for row in manifest['rows']:
            if row['scene'] == scene and not set(row['stimulus_candidate_ids']).issubset(known):
                raise ValueError(f'NR3D candidate IDs missing: CSV row {row["csv_row"]}')
        print(f'Validated {scene}: point cloud / OLT alignment and NR3D IDs', flush=True)
    print(f'All {len(scenes)} scenes prepared. No inference calls made.', flush=True)


if __name__ == '__main__':
    main()
