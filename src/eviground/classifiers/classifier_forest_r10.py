"""A bounded ExtraTrees control on frozen RGB and joint object features."""
from eviground.paths import source_file
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from eviground.experiments.audit_pair_train_test import recount
from eviground.experiments.audit_pair_views import z
from eviground.categories.category_common_v82 import file_hash, run_lock
from eviground.experiments.run_rgb_experiment import check_frozen, freeze
from eviground.evaluation.run_storage import write_json

SPECS = ('rgb_plain', 'rgb_balanced', 'joint_plain', 'joint_balanced')
SEEDS = (42, 43, 44)
WEIGHTS = (0., .1, .2, .4, .6, .8, 1.)


def read(path):
    return json.loads(Path(path).read_text())


def mapped_probability(model, x, classes):
    values = np.zeros((len(x), classes), dtype=np.float64)
    values[:, model.classes_.astype(int)] = model.predict_proba(x)
    assert np.isfinite(values).all() and (values >= 0).all()
    np.testing.assert_allclose(values.sum(1), 1, atol=1e-12, rtol=1e-12)
    return values


def choose(rows, ref, mode):
    if mode == 'B':
        rows = [r for r in rows if r['mean_macro'] >= ref['mean_macro'] - 1e-10]
    elif mode == 'M':
        rows = [r for r in rows if r['mean_correct'] >= ref['mean_correct']]
    else:
        assert mode == 'P'
    key = lambda r: ((-r['mean_macro'], -r['mean_correct']) if mode == 'M' else (-r['mean_correct'], -r['mean_macro'])) + (
        r['nodes'], r['name'], r['base'], r['weight'])
    return min(rows, key=key)


def setup(args):
    import sklearn
    assert sklearn.__version__ == '1.5.2'
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    for marker in ('FEATURES_FROZEN.json', 'EVAL_FEATURES_FROZEN.json'):
        check_frozen(args.reference, marker)
    for root in (args.r4, args.r8):
        check_frozen(root, 'TEST_FROZEN.json')
    manifests = {s: read(args.reference / f'manifests/{s}.json') for s in ('train', 'eval', 'dev')}
    assert [len(m['samples']) for m in manifests.values()] == [28730, 404, 3413]
    rooms = {s: {r['scene'].split('_')[0] for r in m['samples']} for s, m in manifests.items()}
    assert not rooms['train'] & rooms['eval'] and not rooms['train'] & rooms['dev'] and not rooms['eval'] & rooms['dev']
    assert len(manifests['train']['vocabulary']) == 197
    assert all(m['vocabulary'] == manifests['train']['vocabulary'] for m in manifests.values())
    paths = [args.design, args.output / 'DEPENDENCIES.json', args.reference / 'features/text.npy',
             args.r4 / 'validation/locked_M_v0.npy', args.r8 / 'validation/locked_M.npy']
    paths += [args.reference / f'manifests/{s}.json' for s in manifests]
    paths += [args.reference / f'features/{n}' for n in ('shape_train.npz', 'shape_eval.npz',
              'rgb_train/features.npz', 'rgb_eval/features.npz')]
    source = ('classifier_forest_r10.py', 'audit_pair_train_test.py', 'audit_pair_views.py',
              'category_common_v82.py', 'run_storage.py', 'run_rgb_experiment.py')
    config = {'version': 'frozen-extra-trees-r10', 'specs': SPECS, 'seeds': SEEDS, 'sklearn': sklearn.__version__,
              'params': {'n_estimators': 256, 'min_samples_leaf': 2, 'max_features': 'sqrt',
                         'max_leaf_nodes': 1024, 'bootstrap': False, 'n_jobs': 8},
              'weights': WEIGHTS, 'score': 'log(max(mapped_probability,1e-6)), per-object z-score',
              'reference_scale': 'stored scores unchanged', 'source': {n: file_hash(source_file(n)) for n in source},
              'inputs': {str(p): file_hash(p) for p in paths},
              'limitation': 'historically exposed evaluation; pretrained scene exposure unknown; r8 rotation precision limit'}
    config = json.loads(json.dumps(config))
    if (out / 'config.json').exists():
        assert read(out / 'config.json') == config
    else:
        write_json(out / 'config.json', config)
        (out / 'source_snapshot').mkdir()
        (out / 'manifests').mkdir()
        for n in source:
            shutil.copyfile(source_file(n), out / 'source_snapshot' / n)
        shutil.copyfile(args.design, out / 'source_snapshot/design.md')
        for split in manifests:
            shutil.copyfile(args.reference / f'manifests/{split}.json', out / f'manifests/{split}.json')
        freeze(out, 'PROTOCOL_FROZEN.json', ['config.json', 'DEPENDENCIES.json'] + [str(p.relative_to(out))
               for d in ('source_snapshot', 'manifests') for p in (out / d).iterdir()])
    check_frozen(out, 'PROTOCOL_FROZEN.json')
    return manifests


def features(args, split):
    rgb_path = args.reference / ('features/rgb_dev.npz' if split == 'dev' else f'features/rgb_{split}/features.npz')
    data = np.load(rgb_path)
    assert data['valid'].all()
    rgb = data['features'].astype(np.float32)
    shape = np.load(args.reference / f'features/shape_{split}.npz')['features'].astype(np.float32)
    assert len(shape) == len(rgb)
    return {'rgb': rgb, 'joint': np.concatenate((shape, rgb), axis=1)}


def bases(args, role, split):
    text = np.load(args.reference / 'features/text.npy').astype(float)
    shape = np.load(args.reference / f'features/shape_{split}.npz')['features'].astype(float)
    return {'S0': z(10 * shape @ text.T), 'r4_M': np.load(args.r4 / f'{role}/locked_M_v0.npy').astype(float),
            'r8_M': np.load(args.r8 / f'{role}/locked_M.npy').astype(float)}


def train_select(args):
    import joblib
    from sklearn.ensemble import ExtraTreesClassifier
    manifests = setup(args)
    assert not (args.output / 'SELECTION_FROZEN.json').exists()
    train, validation = features(args, 'train'), features(args, 'eval')
    y = np.array([r['label'] for r in manifests['train']['samples']])
    assert (y >= 0).all()
    classes = len(manifests['train']['vocabulary'])
    validation_raw = args.output / 'validation_raw'
    validation_raw.mkdir()
    candidates = {}
    for spec in SPECS:
        modality, balance = spec.split('_')
        arrays, nodes = [], []
        for seed in SEEDS:
            directory = args.output / f'models/{spec}/seed{seed}'
            directory.mkdir(parents=True)
            started = time.time()
            model = ExtraTreesClassifier(**read(args.output / 'config.json')['params'],
                                        class_weight=None if balance == 'plain' else 'balanced', random_state=seed)
            model.fit(train[modality], y)
            assert np.array_equal(model.classes_, np.unique(y))
            # Sum trees serially at inference so floating-point accumulation order is fixed.
            model.set_params(n_jobs=1)
            array = mapped_probability(model, validation[modality], classes)
            np.save(validation_raw / f'{spec}_seed{seed}.npy', array)
            model_path = directory / 'model.joblib'
            joblib.dump(model, model_path, compress=3)
            count = sum(t.tree_.node_count for t in model.estimators_)
            record = {'spec': spec, 'seed': seed, 'train_objects': len(y), 'classes': model.classes_.tolist(),
                      'nodes': count, 'parameters': model.get_params(), 'fit_n_jobs': 8, 'sha256': file_hash(model_path),
                      'config_sha256': file_hash(args.output / 'config.json'), 'seconds': time.time() - started}
            write_json(directory / 'model.json', record)
            print(json.dumps({'stage': 'forest', **record}), flush=True)
            arrays.append(z(np.log(np.maximum(array, 1e-6))))
            nodes.append(count)
            del model
        candidates[spec] = (arrays, float(np.mean(nodes)))
    rows = []
    base = bases(args, 'validation', 'eval')
    for spec, (arrays, nodes) in candidates.items():
        for name, reference in base.items():
            for weight in WEIGHTS:
                measures = [recount((1 - weight) * reference + weight * a, manifests['eval'])[0] for a in arrays]
                rows.append({'name': spec, 'base': name, 'weight': weight, 'nodes': nodes,
                             'correct_by_seed': [m['correct'] for m in measures],
                             'mean_correct': float(np.mean([m['correct'] for m in measures])),
                             'mean_macro': float(np.mean([m['class_mean_accuracy'] for m in measures]))})
    for name, values in base.items():
        summary = recount(values, manifests['eval'])[0]
        rows.append({'name': name, 'base': name, 'weight': 0., 'nodes': 0., 'correct_by_seed': [summary['correct']] * 3,
                     'mean_correct': float(summary['correct']), 'mean_macro': summary['class_mean_accuracy']})
    assert len(rows) == 87
    ref = next(r for r in rows if r['name'] == 'r8_M')
    selection = {'locked': {m: choose(rows, ref, m) for m in ('P', 'B', 'M')},
                 'per_candidate': {s: choose([r for r in rows if r['name'] == s], ref, 'P') for s in SPECS}}
    write_json(args.output / 'validation_grid.json', rows)
    write_json(args.output / 'selection.json', selection)
    freeze(args.output, 'SELECTION_FROZEN.json', ['config.json', 'PROTOCOL_FROZEN.json', 'validation_grid.json', 'selection.json'] + [
           str(p.relative_to(args.output)) for d in ('models', 'validation_raw') for p in (args.output / d).rglob('*') if p.is_file()])
    print(json.dumps({'stage': 'selection_frozen', 'selection': selection}), flush=True)


def evaluate(args):
    import joblib
    manifests = setup(args)
    check_frozen(args.output, 'SELECTION_FROZEN.json')
    assert not (args.output / 'TEST_FROZEN.json').exists()
    selection = read(args.output / 'selection.json')
    chosen = {**selection['per_candidate'], **{'locked_' + m: r for m, r in selection['locked'].items()}}
    chosen.update({n: {'name': n, 'base': n, 'weight': 0.} for n in ('S0', 'r4_M', 'r8_M')})
    records = []
    for role, split in (('validation', 'eval'), ('historical_test', 'dev')):
        inputs = features(args, split)
        base = bases(args, role, split)
        directory = args.output / role
        directory.mkdir()
        raw = {}
        for spec in SPECS:
            raw[spec] = []
            for seed in SEEDS:
                folder = args.output / f'models/{spec}/seed{seed}'
                assert file_hash(folder / 'model.joblib') == read(folder / 'model.json')['sha256']
                model = joblib.load(folder / 'model.joblib')
                values = mapped_probability(model, inputs[spec.split('_')[0]], 197)
                if split == 'eval':
                    np.testing.assert_allclose(values, np.load(args.output / f'validation_raw/{spec}_seed{seed}.npy'), atol=1e-13, rtol=1e-13)
                np.save(directory / f'{spec}_seed{seed}_probability.npy', values)
                records.append({'role': role, 'method': 'raw_' + spec, 'seed': seed, 'selection': None,
                                'summary': recount(np.log(np.maximum(values, 1e-6)), manifests[split])[0]})
                raw[spec].append(z(np.log(np.maximum(values, 1e-6))))
                del model
        for name, sel in chosen.items():
            values = [base[sel['name']]] * 3 if sel['name'] in base else [
                (1 - sel['weight']) * base[sel['base']] + sel['weight'] * v for v in raw[sel['name']]]
            for seed, value in zip(SEEDS, values):
                np.save(directory / f'{name}_seed{seed}.npy', value)
                records.append({'role': role, 'method': name, 'seed': seed, 'selection': sel,
                                'summary': recount(value, manifests[split])[0]})
        print(json.dumps({'stage': role, 'locked': [r for r in records if r['role'] == role and r['method'].startswith('locked_')]}), flush=True)
    write_json(args.output / 'results.json', {'records': records})
    freeze(args.output, 'TEST_FROZEN.json', ['results.json', 'SELECTION_FROZEN.json'] + [str(p.relative_to(args.output))
           for d in ('validation', 'historical_test') for p in (args.output / d).iterdir()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('train-select', 'evaluate'))
    for n in ('output', 'reference', 'r4', 'r8', 'design'):
        parser.add_argument('--' + n, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.output / 'deps_existing'))
    with run_lock(args.output):
        {'train-select': train_select, 'evaluate': evaluate}[args.stage](args)
