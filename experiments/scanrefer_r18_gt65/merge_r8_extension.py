"""Merge original r8 rows unchanged with the historically bridged missing scenes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--extension', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bridge = json.loads((args.extension/'CPU_BRIDGE.json').read_text())
    if not bridge['passed']:
        raise ValueError('CPU historical bridge must pass')
    predictions, configs, matrices = [], [], []
    for root in (args.original, args.extension):
        marker = json.loads((root/'EXPORT_VERIFIED.json').read_text())
        if marker['status'] != 'passed':
            raise ValueError('Unverified input export')
        for name, checksum in marker['files'].items():
            if digest(root/name) != checksum:
                raise ValueError(f'Changed export: {root}/{name}')
        predictions.append(json.loads((root/'predictions.json').read_text()))
        configs.append(json.loads((root/'config.json').read_text()))
        matrices.append(np.load(root/'all_scores.npy', allow_pickle=False))
    if (args.original/'vocabulary.json').read_bytes() != (args.extension/'vocabulary.json').read_bytes():
        # Formatting may differ; vocabulary order must be identical.
        if json.loads((args.original/'vocabulary.json').read_text()) != json.loads((args.extension/'vocabulary.json').read_text()):
            raise ValueError('Different classifier vocabulary/order')
    rows = predictions[0]['results'] + predictions[1]['results']
    sources = configs[0]['sources'] + configs[1]['sources']
    if len({(r['scene'], r['bbox_id']) for r in rows}) != len(rows):
        raise ValueError('Original and extension objects overlap')
    if len({r['scene'] for r in sources}) != len(sources):
        raise ValueError('Original and extension scenes overlap')
    args.output.mkdir(parents=True, exist_ok=False)
    summary = dict(predictions[0]['summary'], count=len(rows), scenes=len(sources), complete=True, errors=0)
    save(args.output/'predictions.json', {'version': predictions[0]['version'], 'results': rows, 'summary': summary})
    config = {'version': configs[0]['version'], 'score_type': configs[0]['score_type'],
              'sources': sources, 'method': 'original r8 rows plus CPU-bridged original r8 missing scenes',
              'input_exports': [{name: digest(root/name) for name in ('predictions.json', 'config.json', 'EXPORT_VERIFIED.json')}
                               for root in (args.original, args.extension)],
              'cpu_bridge_sha256': digest(args.extension/'CPU_BRIDGE.json')}
    save(args.output/'config.json', config)
    save(args.output/'vocabulary.json', json.loads((args.original/'vocabulary.json').read_text()))
    np.save(args.output/'all_scores.npy', np.concatenate(matrices))
    save(args.output/'EXPORT_VERIFIED.json', {'status': 'passed', 'files': {
        name: digest(args.output/name) for name in ('predictions.json', 'config.json', 'vocabulary.json', 'all_scores.npy')}})
    print(json.dumps({'objects': len(rows), 'scenes': len(sources), 'original_rows_unchanged': True}))


if __name__ == '__main__':
    main()
