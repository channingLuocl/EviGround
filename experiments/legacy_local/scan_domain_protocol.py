"""Freeze a research-dev/room-holdout protocol using audited historical manifests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import urllib.request

from category_common_v82 import file_hash, normalized_category, run_lock
from category_v10.data import SUFFIXES, assert_disjoint, make_manifest, room, scene_names
from run_storage import write_json

normalized_category = lru_cache(None)(normalized_category)


def plan(official_train, official_val, historical):
    train, old_dev, research = (historical[s] for s in ('train', 'dev', 'eval'))
    if set(train['scenes']) | set(old_dev['scenes']) != set(official_train):
        raise ValueError('Historical train/dev must cover official train')
    assert_disjoint(official_train, official_val)
    if not set(research['scenes']) <= set(official_val):
        raise ValueError('Historical evaluation is outside official validation')
    used_rooms = {room(s) for m in historical.values() for s in m['scenes']}
    holdout = sorted(s for s in official_val if room(s) not in used_rooms)
    if not holdout:
        raise ValueError('No unused validation rooms remain')
    rows = sorted(train['samples'] + [r for r in old_dev['samples'] if r['label'] >= 0],
                  key=lambda r: (r['scene'], r['bbox_id']))
    merged = {**train, 'scenes': sorted(official_train), 'samples': rows,
              'sources': sorted(train['sources'] + old_dev['sources'], key=lambda r: r['scene']),
              'excluded_training_instances': train['excluded_training_instances'] +
                  [r for r in old_dev['samples'] if r['label'] < 0],
              'excluded_structural_instances': train['excluded_structural_instances'] + old_dev['excluded_structural_instances'],
              'evaluation_status': 'official training scenes; former internal-dev merged into training'}
    development = {**research, 'split': 'dev',
                   'evaluation_status': 'previously inspected 3413-object validation explicitly repurposed for research development'}
    assert_disjoint(merged['scenes'], development['scenes'])
    assert_disjoint(merged['scenes'] + development['scenes'], holdout)
    seen = [(r['scene'], r['bbox_id']) for r in rows]
    if len(set(seen)) != len(rows):
        raise ValueError('Duplicate training object')
    quarantine = sorted(set(official_val) - set(development['scenes']) - set(holdout))
    return merged, development, {'version': 'scan-domain-room-holdout-r1', 'holdout_scenes': holdout,
        'holdout_rooms': sorted({room(s) for s in holdout}), 'quarantined_rescans': quarantine,
        'train_objects': len(rows), 'research_dev_objects': len(development['samples']),
        'policy': 'all previously uninspected validation rooms reserved; old evaluation explicitly becomes research dev',
        'limitation': 'held out from audited new-head experiments, not an official test set; original backbone exposure unknown'}


def prepare(args):
    historical = {s: json.loads((args.historical / f'manifests/{s}.json').read_text()) for s in ('train', 'dev', 'eval')}
    train, dev, protocol = plan(scene_names(args.official_train), scene_names(args.official_val), historical)
    protocol['inputs'] = {str(p): file_hash(p) for p in [args.official_train, args.official_val] +
                          [args.historical / f'manifests/{s}.json' for s in historical]}
    protocol['source_sha256'] = file_hash(Path(__file__))
    with run_lock(args.output):
        if (args.output / 'protocol.json').exists():
            raise ValueError('Protocol exists; preserve it and choose fresh output')
        manifest_root = args.output / 'manifests'
        manifest_root.mkdir()
        write_json(manifest_root / 'train.json', train)
        write_json(manifest_root / 'dev.json', dev)
        write_json(args.output / 'protocol.json', protocol)
        write_json(args.output / 'PROTOCOL_FROZEN.json', {'files': {name: file_hash(args.output / name) for name in
                   ('protocol.json', 'manifests/train.json', 'manifests/dev.json')}})
        print(json.dumps({k: v for k, v in protocol.items() if k != 'inputs'}), flush=True)


def fetch(args):
    from category_v10.data import convert_scene
    protocol = json.loads((args.output / 'protocol.json').read_text())
    for name, digest in json.loads((args.output / 'PROTOCOL_FROZEN.json').read_text())['files'].items():
        if file_hash(args.output / name) != digest:
            raise ValueError('Frozen protocol changed')
    scene_root = args.output / 'heldout_scenes'
    args.raw.mkdir(parents=True, exist_ok=True)

    def acquire(item):
        scene, suffix = item
        path = args.raw / scene / (scene + suffix)
        if path.exists():
            if not path.stat().st_size:
                raise ValueError('Empty existing raw file')
            return
        path.parent.mkdir(exist_ok=True)
        part = path.with_suffix(path.suffix + '.partial')
        # Failed downloads remain partial, never accepted as complete inputs.
        with urllib.request.urlopen(f'https://kaldir.vc.in.tum.de/scannet/v2/scans/{scene}/{scene+suffix}', timeout=60) as response:
            if response.status != 200:
                raise ValueError(f'Unexpected ScanNet response {response.status}')
            with part.open('wb') as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
        if not part.stat().st_size:
            raise ValueError('Empty download')
        part.replace(path)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(acquire, [(s, suffix) for s in protocol['holdout_scenes'] for suffix in SUFFIXES]))
    for i, scene in enumerate(protocol['holdout_scenes'], 1):
        if not (scene_root / scene).exists():
            convert_scene(args.raw, scene, scene_root / scene)
        print(json.dumps({'stage': 'holdout_asset_conversion', 'scenes': i, 'total': len(protocol['holdout_scenes'])}), flush=True)
    manifest = make_manifest(scene_root, protocol['holdout_scenes'], 'eval')
    manifest['evaluation_status'] = protocol['limitation']
    path = args.output / 'manifests/eval.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Heldout manifest changed')
    write_json(path, manifest)
    write_json(args.output / 'HOLDOUT_FROZEN.json', {'manifest_sha256': file_hash(path),
        'source_sha256': file_hash(Path(__file__)), 'objects': len(manifest['samples']),
        'scenes': len(manifest['scenes']), 'rooms': len(protocol['holdout_rooms']), 'performance_not_computed': True})
    print(json.dumps({'stage': 'holdout_assets_sealed', 'scenes': len(manifest['scenes']),
                      'rooms': len(protocol['holdout_rooms']), 'objects': len(manifest['samples'])}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'fetch'))
    for name in ('official-train', 'official-val', 'historical', 'output', 'raw'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.stage == 'prepare' else fetch)(args)
