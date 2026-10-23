"""Independently recount fixed RGB fusion, dev selection, and paired room CI."""
import argparse
import json
from pathlib import Path

import numpy as np

from seeground.experiments.audit_shape_scale import recount
from seeground.categories.category_common_v82 import file_hash
from seeground.evaluation.run_storage import write_json


def standardized(scores):
    return (scores - np.mean(scores, axis=1, keepdims=True)) / np.maximum(np.std(scores, axis=1, keepdims=True), 1e-6)


def compose(parts, w, c, seed):
    base = standardized(parts['base'])
    result = (1-w)*base + w*standardized(parts['image'])
    result[~parts['valid']] = base[~parts['valid']]
    return result + c*parts['size'][seed]


def main(output):
    config = json.loads((output / 'config.json').read_text())
    for marker in ('QA_FROZEN.json', 'SELECTION_FROZEN.json', 'RESULTS_FROZEN.json'):
        for name, digest in json.loads((output / marker).read_text())['files'].items():
            assert file_hash(output / name) == digest, name
    for name, digest in config['code_sha256'].items():
        assert file_hash(output / 'source_snapshot' / name) == digest, name
    manifests = {s: json.loads((output / f'manifests/{s}.json').read_text()) for s in ('train', 'dev', 'eval')}
    for s in manifests:
        assert file_hash(output / f'manifests/{s}.json') == config['manifests'][s]
    grid = json.loads((output / 'dev_grid.json').read_text())
    selection = json.loads((output / 'selection.json').read_text())
    with np.load(output / 'dev/components.npz') as parts:
        for method, rows in grid.items():
            expected_pairs = [(w, c) for w in config['weights'] for c in ([0.] if method == 'V2' else config['size_weights'])]
            assert [tuple(row['weights']) for row in rows] == expected_pairs
            for row in rows:
                scores = [recount(compose(parts, *row['weights'], seed), manifests['dev'])[0]['accuracy']
                          for seed in (range(1) if method == 'V2' else range(3))]
                np.testing.assert_allclose(scores, row['accuracies'], atol=1e-12, rtol=0)
                assert abs(np.mean(scores) - row['mean_accuracy']) < 1e-12
            assert selection[method] == min(rows, key=lambda r: (-r['mean_accuracy'], r['weights'][1], r['weights'][0]))
    results = json.loads((output / 'results.json').read_text())
    hits = {}
    with np.load(output / 'eval/components.npz') as parts:
        assert len(parts['valid']) == 3413
        assert int(parts['valid'].sum()) == results['valid_RGB']
        baseline, base_hit = recount(parts['base'], manifests['eval'])
        assert baseline['correct'] == 2262
        for record in results['records']:
            method = record['method']
            if method == 'V0':
                expected = parts['base']
            elif method == 'V1':
                expected = compose(parts, 1., 0., 0)
            else:
                group = 'V3' if method.startswith('V3') else 'V2'
                index = int(method[-2:]) - 42 if group == 'V3' else 0
                expected = compose(parts, *selection[group]['weights'], index)
            actual = np.load(output / f'eval/{method}.npy')
            np.testing.assert_array_equal(actual, expected)
            summary, hit = recount(actual, manifests['eval'])
            for key, value in summary.items():
                assert abs(value - record['summary'][key]) < 1e-12
            assert int((~base_hit & hit).sum()) == record['corrected']
            assert int((base_hit & ~hit).sum()) == record['broken']
            hits[method] = hit.astype(float)
    rooms = [row['scene'].split('_')[0] for row in manifests['eval']['samples']]
    unique = sorted(set(rooms))
    counts = np.array([rooms.count(room) for room in unique])
    rng = np.random.RandomState(20261019)
    weights = rng.multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=10000)
    denominators = weights @ counts
    intervals = {}
    hits['V3_mean_three_heads'] = np.mean([hits[f'V3_seed{s}'] for s in (42, 43, 44)], axis=0)
    for method, hit in hits.items():
        if method == 'V0':
            continue
        differences = hit - hits['V0']
        sums = np.array([differences[np.array(rooms) == room].sum() for room in unique])
        distribution = (weights @ sums) / denominators
        intervals[method] = {'accuracy_delta_percentage_points': float(differences.mean()*100),
                             'paired_room_bootstrap_95CI_pp': (np.quantile(distribution, [.025, .975])*100).tolist()}
    report = {'status': 'passed', 'objects': 3413, 'physical_rooms': len(unique),
              'auditor_sha256': file_hash(Path(__file__)), 'intervals': intervals,
              'checks': ['all frozen file hashes and executed sources', 'all dev grid scores and selection',
                         'all eval logits and metrics', 'RGB fallback coverage', 'baseline exact reproduction'],
              'bootstrap': {'iterations': 10000, 'seed': 20261019, 'unit': 'physical room',
                            'limitation': 'sampling uncertainty only, no multiple-comparisons adjustment'}}
    write_json(output / 'audit.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args().output)
