"""Fit semantic residuals with paired support scaling and room-weighting controls."""
from eviground.paths import source_file
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from eviground.experiments.audit_pair_train_test import recount
from eviground.experiments.audit_pair_views import z
from eviground.categories.category_common_v82 import category_vocabulary, file_hash, run_lock
from eviground.classifiers.classifier_residual_r9 import bases, choose
from eviground.experiments.run_rgb_experiment import check_frozen, freeze
from eviground.evaluation.run_storage import write_json
from eviground.experiments.support_classifier_r13 import SPECS, SupportHead, training_support, weighted_loss

SEEDS = (42, 43, 44)
REFERENCES = ('S0', 'r4_M', 'r8_M')
WEIGHTS = (0., .2, .4, .6, .8, 1.)
SOURCE_NAMES = ('classifier_support_r13.py', 'support_classifier_r13.py',
    'audit_classifier_support_r13.py', 'test_support_classifier_r13.py',
    'audit_classifier_balance_r4.py', 'audit_classifier_residual_r9.py',
    'category_common_v82.py', 'audit_pair_train_test.py', 'audit_pair_views.py',
    'classifier_residual_r9.py', 'pointnext_residual_r9.py', 'classifier_backbone_r5.py',
    'classifier_sweep_r3.py', 'train_pair_views.py', 'run_rgb_experiment.py',
    'rgb_classifier.py', 'shape_scale.py', 'run_storage.py', 'category_v10/data.py', 'category_v10/run.py')


def read(path):
    return json.loads(Path(path).read_text())


def setup(args):
    root = args.output; root.mkdir(parents=True, exist_ok=True)
    manifests = {s: read(args.reference / f'manifests/{s}.json') for s in ('train', 'eval', 'dev')}
    assert [len(m['samples']) for m in manifests.values()] == [28730, 404, 3413]
    rooms = {s: {r['scene'].split('_')[0] for r in m['samples']} for s, m in manifests.items()}
    assert all(not rooms[a] & rooms[b] for a, b in (('train', 'eval'), ('train', 'dev'), ('eval', 'dev')))
    assert all(m['vocabulary'] == list(category_vocabulary()) for m in manifests.values())
    assert all(r['label'] >= 0 for r in manifests['train']['samples'])
    for marker in ('FEATURES_FROZEN.json', 'EVAL_FEATURES_FROZEN.json'):
        check_frozen(args.reference, marker)
    for ref in (args.r4, args.r8):
        check_frozen(ref, 'TEST_FROZEN.json')
    paths = [args.design] + [args.reference / f'manifests/{s}.json' for s in manifests]
    paths += [args.reference / f'features/{n}' for n in ('text.npy', 'shape_eval.npz',
              'rgb_train/features.npz', 'rgb_eval/features.npz')]
    paths += [args.r4 / 'validation/locked_M_v0.npy', args.r8 / 'validation/locked_M.npy']
    config = {'version': 'training-support-room-r13', 'seeds': SEEDS, 'specs': SPECS,
        'epochs': 40, 'batch': 256, 'lr': .001, 'weight_decay': .01,
        'model': '10*x@frozen_text.T + gains*LN512-linear256-GELU-dropout.3-linear197; last linear zero',
        'gain': 'sqrt[n/(n+16)*R/(R+4)] in support arms; one in controls; TRAIN only',
        'room_factor': 'n_c/(sqrt(n_cr)*sum_r sqrt(n_cr)); one in object controls',
        'loss': 'batch-weight-normalized CE; (n_c+1)^(-1/2) class weight times room factor',
        'input': 'existing frozen four-view average unit CLIP512; residual input dropout .1 only',
        'precision': 'float32, TF32 off; no backbone training or new renders',
        'selection': '75 validation rows, last epoch40 only; P/B/M relative to r8_M',
        'reference_scale': 'original stored S0/r4/r8; only new scores per-object z-score',
        'source': {n: file_hash(source_file(n)) for n in SOURCE_NAMES},
        'inputs': {str(p): file_hash(p) for p in dict.fromkeys(paths)}, 'torch': str(torch.__version__),
        'limitations': 'historical evaluation exposed; pretrained scene exposure unknown; r8 rotation precision limit'}
    config = json.loads(json.dumps(config))
    if (root / 'config.json').exists():
        assert read(root / 'config.json') == config
    else:
        write_json(root / 'config.json', config)
        for folder in ('source_snapshot', 'manifests'):
            (root / folder).mkdir()
        for name in SOURCE_NAMES:
            dest = root / 'source_snapshot' / name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file(name), dest)
        shutil.copyfile(args.design, root / 'source_snapshot/design.md')
        for split in manifests:
            shutil.copyfile(args.reference / f'manifests/{split}.json', root / f'manifests/{split}.json')
        freeze(root, 'PROTOCOL_FROZEN.json', ['config.json'] + [str(p.relative_to(root))
               for folder in ('source_snapshot', 'manifests') for p in (root / folder).rglob('*') if p.is_file()])
    check_frozen(root, 'PROTOCOL_FROZEN.json')
    return manifests


def load_data(args, split):
    if split == 'dev':
        check_frozen(args.output, 'SELECTION_FROZEN.json')
    path = args.reference / ('features/rgb_dev.npz' if split == 'dev' else f'features/rgb_{split}/features.npz')
    rgb = np.load(path); assert rgb['valid'].all()
    values = rgb['features']
    assert values.dtype == np.float32 and values.shape[1] == 512 and np.isfinite(values).all()
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1., atol=2e-6, rtol=0)
    return torch.from_numpy(values).cuda()


def text_features(args):
    text = np.load(args.reference / 'features/text.npy')
    assert text.shape == (197, 512) and text.dtype == np.float32 and np.isfinite(text).all()
    return torch.from_numpy(text).cuda()


@torch.inference_mode()
def predict(model, values):
    model.eval()
    return np.concatenate([model(batch).cpu().numpy() for batch in values.split(256)])


def support_data(manifest):
    return {k: v.cuda() for k, v in training_support(manifest['samples'], 197).items()}


def preflight(args):
    from eviground.experiments.audit_classifier_support_r13 import independent_head
    manifests = setup(args); path = args.output / 'PREFLIGHT.json'; assert not path.exists()
    values, text = load_data(args, 'train')[:32], text_features(args)
    support = support_data(manifests['train']); rows = []
    for spec in SPECS:
        torch.manual_seed(42); model = SupportHead(text, support['gains'], spec).cuda()
        model.eval(); torch.testing.assert_close(model(values), 10*values@text.T, atol=0, rtol=0)
        labels = support['labels'][:32]; weight = support['object_weight'][:32]
        if spec.endswith('_room'):
            weight = weight*support['room_factor'][:32]
        before = float(weighted_loss(model(values), labels, weight).detach())
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        model.train()
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_loss(model(values), labels, weight)
            assert torch.isfinite(loss); loss.backward(); optimizer.step()
        model.eval(); after = float(weighted_loss(model(values), labels, weight).detach()); assert after < before
        torch.testing.assert_close(model.text, text, atol=0, rtol=0)
        expected_gain = support['gains'] if spec.startswith('support') else torch.ones_like(support['gains'])
        torch.testing.assert_close(model.gains, expected_gain, atol=0, rtol=0)
        scores = predict(model, values)
        if spec.startswith('support'):
            torch.testing.assert_close(model(values)[:, support['counts']==0],
                (10*values@text.T)[:, support['counts']==0], atol=0, rtol=0)
        independent = independent_head({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}}, values.cpu().numpy())
        np.testing.assert_allclose(independent, scores, atol=5e-5, rtol=2e-5)
        np.testing.assert_array_equal(np.argsort(-independent, axis=1, kind='stable')[:, :5],
                                      np.argsort(-scores, axis=1, kind='stable')[:, :5])
        rows.append({'spec': spec, 'loss_before': before, 'loss_after': after,
                     'numpy_max_abs': float(np.max(np.abs(independent-scores)))})
    write_json(path, {'status': 'passed', 'train_objects_only': 32, 'steps': 20, 'checks': rows,
                     'text_and_gains': 'exactly_unchanged', 'support_absent_class_raw_logit': 'exact'})
    print(json.dumps(read(path)), flush=True)


def fit(args, spec, seed, values, support, text):
    directory = args.output / f'models/{spec}/seed{seed}'; directory.mkdir(parents=True)
    torch.manual_seed(seed); model = SupportHead(text, support['gains'], spec).cuda()
    initial = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    generator = torch.Generator(device='cuda').manual_seed(seed)
    labels = support['labels']; weight = support['object_weight']
    if spec.endswith('_room'):
        weight = weight*support['room_factor']
    history = []; started = time.time()
    for epoch in range(40):
        model.train(); total = 0.
        for indices in torch.randperm(len(labels), device='cuda', generator=generator).split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_loss(model(values[indices]), labels[indices], weight[indices])
            assert torch.isfinite(loss); loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(indices)
        history.append(total / len(labels))
    torch.testing.assert_close(model.text, text, atol=0, rtol=0)
    expected_gain = support['gains'] if spec.startswith('support') else torch.ones_like(support['gains'])
    torch.testing.assert_close(model.gains, expected_gain, atol=0, rtol=0)
    change = float(sum((p.detach()-before).square().sum() for p, before in zip(model.parameters(), initial)).sqrt())
    assert change > 0
    record = {'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'spec': spec, 'seed': seed, 'epoch': 40, 'train_objects': len(labels),
        'parameters': sum(p.numel() for p in model.parameters()), 'parameter_change_l2': change,
        'config_sha256': file_hash(args.output / 'config.json')}
    path = directory / 'e40.pt'; torch.save(record, path)
    write_json(path.with_suffix('.json'), {'sha256': file_hash(path)})
    write_json(directory / 'history.json', history)
    print(json.dumps({'stage': 'fit_complete', 'spec': spec, 'seed': seed, 'loss': history[-1],
                      'seconds': time.time()-started, 'parameter_change': change}), flush=True)
    return model.eval(), record['parameters']


def make_grid(raw, base, manifest):
    rows = []
    for spec, (arrays, parameters) in raw.items():
        for name, reference in base.items():
            for weight in WEIGHTS:
                metrics = [recount((1-weight)*reference + weight*z(a.astype(float)), manifest)[0] for a in arrays]
                rows.append({'name': spec, 'epoch': 40, 'base': name, 'weight': weight, 'parameters': parameters,
                    'correct_by_seed': [m['correct'] for m in metrics],
                    'mean_correct': float(np.mean([m['correct'] for m in metrics])),
                    'mean_macro': float(np.mean([m['class_mean_accuracy'] for m in metrics]))})
    for name, scores in base.items():
        metric = recount(scores, manifest)[0]
        rows.append({'name': name, 'epoch': 0, 'base': name, 'weight': 0., 'parameters': 0,
            'correct_by_seed': [metric['correct']]*3, 'mean_correct': float(metric['correct']),
            'mean_macro': metric['class_mean_accuracy']})
    reference = next(r for r in rows if r['name'] == 'r8_M')
    return rows, {'locked': {m: choose(rows, reference, m) for m in ('P', 'B', 'M')},
        'per_candidate': {s: choose([r for r in rows if r['name'] == s], reference, 'P') for s in SPECS}}


def train_select(args):
    manifests = setup(args); assert read(args.output / 'PREFLIGHT.json')['status'] == 'passed'
    assert not (args.output / 'SELECTION_FROZEN.json').exists()
    train, validation, text = load_data(args, 'train'), load_data(args, 'eval'), text_features(args)
    support = support_data(manifests['train']); raw = {}
    np.savez(args.output / 'train_support.npz', **{k: v.cpu().numpy() for k, v in support.items()})
    folder = args.output / 'validation_raw'; folder.mkdir()
    for spec in SPECS:
        arrays = []
        for seed in SEEDS:
            model, parameters = fit(args, spec, seed, train, support, text)
            scores = predict(model, validation); arrays.append(scores)
            np.save(folder / f'{spec}_e40_seed{seed}.npy', scores); del model
        raw[spec] = (arrays, parameters)
    grid, selection = make_grid(raw, bases(args, 'validation', 'eval'), manifests['eval']); assert len(grid) == 75
    write_json(args.output / 'validation_grid.json', grid); write_json(args.output / 'selection.json', selection)
    paths = ['validation_grid.json', 'selection.json', 'PREFLIGHT.json', 'train_support.npz']
    paths += [str(p.relative_to(args.output)) for name in ('models', 'validation_raw')
              for p in (args.output / name).rglob('*') if p.is_file()]
    freeze(args.output, 'SELECTION_FROZEN.json', paths)
    print(json.dumps({'stage': 'selection_frozen', 'selection': selection}), flush=True)


def evaluate(args):
    manifests = setup(args); check_frozen(args.output, 'SELECTION_FROZEN.json')
    assert not (args.output / 'TEST_FROZEN.json').exists()
    selection = read(args.output / 'selection.json'); text = text_features(args)
    support = support_data(manifests['train']); records = []; inputs = {}
    selected = {n: {'name': n, 'epoch': 0, 'base': n, 'weight': 0., 'parameters': 0} for n in REFERENCES}
    selected.update(selection['per_candidate']); selected.update({'locked_'+k: v for k, v in selection['locked'].items()})
    for role, split in (('validation', 'eval'), ('historical_test', 'dev')):
        folder = args.output / role; folder.mkdir()
        data, base, raw = load_data(args, split), bases(args, role, split), {}
        rgb_path = args.reference / ('features/rgb_dev.npz' if split == 'dev' else f'features/rgb_{split}/features.npz')
        inputs[str(rgb_path)] = file_hash(rgb_path)
        for spec in SPECS:
            for seed in SEEDS:
                path = args.output / f'models/{spec}/seed{seed}/e40.pt'
                assert file_hash(path) == read(path.with_suffix('.json'))['sha256']
                record = torch.load(path, weights_only=True, map_location='cpu')
                assert record['config_sha256'] == file_hash(args.output / 'config.json')
                model = SupportHead(text, support['gains'], spec).cuda()
                model.load_state_dict(record['state_dict'], strict=True)
                scores = predict(model, data); raw[(spec, seed)] = scores
                np.save(folder / f'{spec}_e40_seed{seed}_raw.npy', scores)
                if split == 'eval':
                    np.testing.assert_array_equal(scores, np.load(args.output / f'validation_raw/{spec}_e40_seed{seed}.npy'))
                records.append({'role': role, 'method': spec+'_pure', 'seed': seed,
                    'selection': {'name': spec, 'epoch': 40, 'base': 'r8_M', 'weight': 1., 'parameters': record['parameters']},
                    'summary': recount(scores, manifests[split])[0]})
                del model
        for name, choice in selected.items():
            for seed in SEEDS:
                scores = base[choice['name']] if choice['name'] in base else (
                    (1-choice['weight'])*base[choice['base']] + choice['weight']*z(raw[(choice['name'], seed)].astype(float)))
                np.save(folder / f'{name}_seed{seed}.npy', scores)
                records.append({'role': role, 'method': name, 'seed': seed, 'selection': choice,
                                'summary': recount(scores, manifests[split])[0]})
        print(json.dumps({'stage': role+'_scored', 'records': [r for r in records if r['role'] == role]}), flush=True)
    write_json(args.output / 'results.json', {'records': records, 'feature_inputs': inputs})
    paths = ['results.json'] + [str(p.relative_to(args.output)) for name in ('validation', 'historical_test')
                               for p in (args.output / name).rglob('*') if p.is_file()]
    freeze(args.output, 'TEST_FROZEN.json', paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('preflight', 'train-select', 'evaluate'))
    for name in ('output', 'reference', 'r4', 'r8', 'design'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    with run_lock(args.output):
        {'preflight': preflight, 'train-select': train_select, 'evaluate': evaluate}[args.stage](args)
