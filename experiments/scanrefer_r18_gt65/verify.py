"""Verify frozen runtime and per-split input hashes before any model call."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify(split, version):
    frozen = json.loads((ROOT / 'versions' / (version + '.json')).read_text())
    runtime = ROOT / 'source_r18/local'
    actual = {p.name: digest(p) for p in runtime.glob('*.py')
              if not p.name.startswith(('test_', '._'))}
    if actual != frozen['source_sha256']:
        raise ValueError('Runtime differs from frozen version; freeze a new version and use fresh outputs')
    if split == 'full' and not frozen['allow_full']:
        raise ValueError('This development version is not approved for full validation')
    expected = 9508 if split == 'full' else int(split)
    manifest_path = ROOT / 'inputs' / ('full_manifest.json' if split == 'full' else f'dev{split}_manifest.json')
    manifest = json.loads(manifest_path.read_text())
    rows = manifest['rows']
    if len(rows) != expected or len({r['csv_row'] for r in rows}) != expected:
        raise ValueError('Manifest has wrong denominator or duplicate IDs')
    scenes = {r['scene'] for r in rows}
    if split == 'full' and len(scenes) != 141:
        raise ValueError('Full validation must contain all 141 scenes')
    for name, checksum in frozen['input_sha256'].items():
        parts = Path(name).parts
        if name.startswith('inputs/scenes/') and parts[2] not in scenes:
            continue
        if digest(ROOT / name) != checksum:
            raise ValueError(f'Frozen input changed: {name}')
    required = [manifest_path.relative_to(ROOT).as_posix()]
    required += [f'inputs/scenes/{scene}/{name}' for scene in scenes
                 for name in ('scene.npz', 'scene_instances.json', 'objects_gt.json')]
    required += [f'inputs/r8_predictions/{name}' for name in ('predictions.json', 'config.json', 'EXPORT_VERIFIED.json')]
    missing = sorted(set(required) - frozen['input_sha256'].keys())
    if missing:
        raise ValueError(f'Inputs not frozen ({len(missing)}): {missing[:5]}')
    for name, checksum in frozen['wrapper_sha256'].items():
        if digest(ROOT / name) != checksum:
            raise ValueError(f'Frozen experiment wrapper changed: {name}; freeze a new version')
    return frozen, manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('32', '84', 'full'), default='84')
    parser.add_argument('--version', required=True)
    args = parser.parse_args()
    frozen, manifest = verify(args.split, args.version)
    print(json.dumps({'version': args.version, 'queries': len(json.loads(manifest.read_text())['rows']),
                      'source_files': len(frozen['source_sha256']), 'model_calls_sent': 0}))


if __name__ == '__main__':
    main()
