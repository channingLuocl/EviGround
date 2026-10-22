"""Train residual point adapters with a fixed text classifier and frozen geometry weights."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from audit_pair_train_test import recount
from audit_pair_views import z
from category_common_v82 import file_hash, run_lock
from classifier_backbone_r5 import make_encoder, read
from pointnext_residual_r9 import ADAPTER, SPECS, ResidualPointNext, losses, original_state
from run_rgb_experiment import check_frozen, freeze
from run_storage import write_json

SEEDS = (42, 43, 44)
EPOCHS = (4, 8)
WEIGHTS = (0., .2, .4, .6, .8, 1.)
REFERENCES = ('S0', 'r4_M', 'r8_M')


def choose(rows, ref, mode):
    if mode == 'B':
        rows = [r for r in rows if r['mean_macro'] >= ref['mean_macro'] - 1e-10]
    elif mode == 'M':
        rows = [r for r in rows if r['mean_correct'] >= ref['mean_correct']]
    else:
        assert mode == 'P'
    def key(r):
        metrics = (-r['mean_macro'], -r['mean_correct']) if mode == 'M' else (-r['mean_correct'], -r['mean_macro'])
        return (*metrics, r['parameters'], r['name'], r['epoch'], r['base'], r['weight'])
    return min(rows, key=key)


def setup(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    manifests = {s: read(args.reference / f'manifests/{s}.json') for s in ('train', 'eval', 'dev')}
    assert [len(m['samples']) for m in manifests.values()] == [28730, 404, 3413]
    rooms = {s: {r['scene'].split('_')[0] for r in m['samples']} for s, m in manifests.items()}
    assert not rooms['train'] & rooms['eval'] and not rooms['train'] & rooms['dev'] and not rooms['eval'] & rooms['dev']
    assert len(manifests['train']['vocabulary']) == 197
    assert all(m['vocabulary'] == manifests['train']['vocabulary'] for m in manifests.values())
    assert all(r['label'] >= 0 for r in manifests['train']['samples'])
    for marker in ('FEATURES_FROZEN.json', 'EVAL_FEATURES_FROZEN.json'):
        check_frozen(args.reference, marker)
    for root in (args.color, args.r4, args.r8):
        check_frozen(root, 'TEST_FROZEN.json')
    assert read(args.color / 'AUDIT.json')['status'] == 'passed'
    assert read(args.r8 / 'AUDIT.json')['status'] == 'passed_with_rotation_precision_limit'
    sources = ('classifier_residual_r9.py', 'pointnext_residual_r9.py', 'classifier_backbone_r5.py',
               'pointcloud_classifier_v82.py', 'category_common_v82.py', 'run_rgb_experiment.py',
               'run_storage.py', 'audit_pair_views.py', 'audit_pair_train_test.py')
    inputs = [args.design, args.checkpoint, args.reference / 'features/text.npy',
              args.reference / 'features/shape_train.npz', args.reference / 'features/shape_eval.npz',
              args.r4 / 'validation/locked_M_v0.npy', args.r8 / 'validation/locked_M.npy']
    inputs += [args.reference / f'manifests/{s}.json' for s in manifests]
    inputs += [args.color / f'points/{s}.json' for s in ('train', 'eval')]
    config = {'version': 'residual-point-adapter-r9', 'seeds': SEEDS, 'specs': SPECS, 'epochs': 8,
              'checkpoints': EPOCHS, 'batch': 128, 'lr': .001, 'weight_decay': .01,
              'loss': 'inverse-sqrt weighted CE + unit-weight KL(T=2)*T^2, except color_ce',
              'classifier': 'fixed 10 * normalized embedding @ original text.T; no fitted head',
              'frozen': 'all original parameters and BN; only 7->64->32 stem adapter trains',
              'reference_score_scale': 'stored S0/r4/r8; new branch per-object z-score',
              'source': {n: file_hash(Path(__file__).parent / n) for n in sources},
              'inputs': {str(p): file_hash(p) for p in inputs}, 'torch': str(torch.__version__),
              'limitation': 'historically exposed evaluation; original pretrained scene exposure unknown; r8 inherits rotation precision limit'}
    config = json.loads(json.dumps(config))
    if (out / 'config.json').exists():
        assert read(out / 'config.json') == config
    else:
        write_json(out / 'config.json', config)
        (out / 'source_snapshot').mkdir()
        (out / 'manifests').mkdir()
        for n in sources:
            shutil.copyfile(Path(__file__).parent / n, out / 'source_snapshot' / n)
        shutil.copyfile(args.design, out / 'source_snapshot/design.md')
        for split in manifests:
            shutil.copyfile(args.reference / f'manifests/{split}.json', out / f'manifests/{split}.json')
        freeze(out, 'PROTOCOL_FROZEN.json', ['config.json'] + [str(p.relative_to(out))
               for d in ('source_snapshot', 'manifests') for p in (out / d).iterdir()])
    check_frozen(out, 'PROTOCOL_FROZEN.json')
    return manifests


def points(args, split):
    p = args.color / f'points/{split}.npy'
    assert file_hash(p) == read(p.with_suffix('.json'))['sha256']
    return torch.from_numpy(np.load(p)).cuda()


def frozen_check(model, expected):
    state = model.state_dict()
    for name, value in expected.items():
        assert torch.equal(state[name], value), name
    assert all(p.requires_grad == name.startswith(ADAPTER) for name, p in model.named_parameters())
    assert all(not m.training for m in model.modules())


@torch.inference_mode()
def predict(model, x, text):
    return np.concatenate([(10 * F.normalize(model(batch).float(), dim=-1) @ text.T).cpu().numpy()
                           for batch in x.split(32)])


def preflight(args):
    manifests = setup(args)
    assert not (args.output / 'PREFLIGHT.json').exists()
    x = points(args, 'train')[:128]
    teacher = torch.from_numpy(np.load(args.reference / 'features/shape_train.npz')['features'][:128]).cuda()
    text = torch.from_numpy(np.load(args.reference / 'features/text.npy')).cuda()
    y = torch.tensor([r['label'] for r in manifests['train']['samples'][:128]], device='cuda')
    torch.manual_seed(42)
    original = make_encoder(args)
    expected = original_state({k: v.detach().clone() for k, v in original.state_dict().items()})
    with torch.inference_mode():
        before = original(x[..., :4]).clone()
    model = ResidualPointNext(original, 'color_kd')
    with torch.inference_mode():
        first = model(x)
    torch.testing.assert_close(first, before, atol=0, rtol=0)
    torch.testing.assert_close(F.normalize(first, dim=-1), teacher, atol=2e-5, rtol=2e-4)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.01)
    history = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(20):
        optimizer.zero_grad(set_to_none=True)
        total, ce, kd = losses(model(x), teacher, text, y, torch.ones(len(text), device='cuda'), 'color_kd')
        total.backward()
        assert torch.isfinite(total)
        optimizer.step()
        history.append({'loss': float(total.detach()), 'ce': float(ce.detach()), 'kd': float(kd.detach())})
    frozen_check(model, expected)
    assert history[-1]['loss'] < history[0]['loss']
    rgb_grad = float(dict(model.named_parameters())[ADAPTER + '0.weight'].grad[:, 4:].norm())
    assert rgb_grad > 0
    result = {'status': 'passed', 'objects': 128, 'steps': 20, 'losses': history,
              'initial_function_max_abs': float((first - before).abs().max()),
              'teacher_feature_max_abs': float((F.normalize(first, dim=-1) - teacher).abs().max()),
              'rgb_input_grad_norm': rgb_grad, 'original_parameters_and_bn': 'exactly_unchanged',
              'peak_cuda_bytes': torch.cuda.max_memory_allocated()}
    write_json(args.output / 'PREFLIGHT.json', result)
    print(json.dumps(result), flush=True)


def bases(args, role, split):
    text = np.load(args.reference / 'features/text.npy').astype(float)
    shape = np.load(args.reference / f'features/shape_{split}.npz')['features'].astype(float)
    return {'S0': z(10 * shape @ text.T),
            'r4_M': np.load(args.r4 / f'{role}/locked_M_v0.npy').astype(float),
            'r8_M': np.load(args.r8 / f'{role}/locked_M.npy').astype(float)}


def make_grid(raw, base, manifest):
    rows = []
    for (spec, epoch), (arrays, parameters) in raw.items():
        for name, reference in base.items():
            for weight in WEIGHTS:
                metrics = [recount((1 - weight) * reference + weight * z(v.astype(float)), manifest)[0] for v in arrays]
                rows.append({'name': spec, 'epoch': epoch, 'base': name, 'weight': weight, 'parameters': parameters,
                             'correct_by_seed': [m['correct'] for m in metrics],
                             'mean_correct': float(np.mean([m['correct'] for m in metrics])),
                             'mean_macro': float(np.mean([m['class_mean_accuracy'] for m in metrics]))})
    for name, reference in base.items():
        metric = recount(reference, manifest)[0]
        rows.append({'name': name, 'epoch': 0, 'base': name, 'weight': 0., 'parameters': 0,
                     'correct_by_seed': [metric['correct']] * 3, 'mean_correct': float(metric['correct']),
                     'mean_macro': metric['class_mean_accuracy']})
    ref = next(r for r in rows if r['name'] == 'r8_M')
    selection = {'locked': {mode: choose(rows, ref, mode) for mode in ('P', 'B', 'M')},
                 'per_candidate': {s: choose([r for r in rows if r['name'] == s], ref, 'P') for s in SPECS}}
    return rows, selection


def train_select(args):
    manifests = setup(args)
    assert read(args.output / 'PREFLIGHT.json')['status'] == 'passed'
    assert not (args.output / 'SELECTION_FROZEN.json').exists()
    train, validation = points(args, 'train'), points(args, 'eval')
    text = torch.from_numpy(np.load(args.reference / 'features/text.npy')).cuda()
    teacher = torch.from_numpy(np.load(args.reference / 'features/shape_train.npz')['features']).cuda()
    y = torch.tensor([r['label'] for r in manifests['train']['samples']], device='cuda')
    weights = (torch.bincount(y, minlength=len(text)).float() + 1).rsqrt()
    weights /= weights.mean()
    (args.output / 'validation_raw').mkdir(exist_ok=True)
    raw = {}
    for spec in SPECS:
        for seed in SEEDS:
            torch.manual_seed(seed)
            original = make_encoder(args)
            expected = original_state({k: v.detach().clone() for k, v in original.state_dict().items()})
            model = ResidualPointNext(original, spec)
            parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
            generator = torch.Generator(device='cuda').manual_seed(seed)
            directory = args.output / f'models/{spec}/seed{seed}'
            directory.mkdir(parents=True)
            history = []
            for epoch in range(1, 9):
                started = time.time()
                totals = np.zeros(3)
                seen = 0
                for indices in torch.randperm(len(y), device='cuda', generator=generator).split(128):
                    optimizer.zero_grad(set_to_none=True)
                    total, ce, kd = losses(model(train[indices]), teacher[indices], text, y[indices], weights, spec)
                    assert torch.isfinite(total)
                    total.backward()
                    if epoch == 1 and seen == 128:
                        grad = {n: float(p.grad.norm()) for n, p in model.named_parameters() if p.requires_grad}
                        rgb_grad = float(dict(model.named_parameters())[ADAPTER + '0.weight'].grad[:, 4:].norm())
                        assert sum(grad.values()) > 0
                        assert (rgb_grad == 0) if spec == 'geometry_kd' else (rgb_grad > 0)
                        write_json(directory / 'gradients.json', {'all': grad, 'rgb_norm': rgb_grad})
                    optimizer.step()
                    totals += np.array([float(t.detach()) for t in (total, ce, kd)]) * len(indices)
                    seen += len(indices)
                scheduler.step()
                assert seen == 28730
                frozen_check(model, expected)
                record = {'epoch': epoch, 'objects': seen, 'loss': totals[0] / seen, 'ce': totals[1] / seen,
                          'kd': totals[2] / seen, 'seconds': time.time() - started, 'frozen_state': 'exactly_unchanged'}
                history.append(record)
                write_json(directory / 'history.json', history)
                print(json.dumps({'stage': 'epoch', 'spec': spec, 'seed': seed, **record}), flush=True)
                if epoch in EPOCHS:
                    values = predict(model, validation, text)
                    np.save(args.output / f'validation_raw/{spec}_e{epoch}_seed{seed}.npy', values)
                    raw.setdefault((spec, epoch), ([], parameters))[0].append(values)
                    torch.save({'encoder': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                'epoch': epoch, 'seed': seed, 'spec': spec, 'parameters': parameters,
                                'config_sha256': file_hash(args.output / 'config.json')}, directory / f'e{epoch}.pt')
            del model, original, optimizer, expected
    grid, selection = make_grid(raw, bases(args, 'validation', 'eval'), manifests['eval'])
    write_json(args.output / 'validation_grid.json', grid)
    write_json(args.output / 'selection.json', selection)
    freeze(args.output, 'SELECTION_FROZEN.json', ['config.json', 'PROTOCOL_FROZEN.json', 'PREFLIGHT.json',
           'selection.json', 'validation_grid.json'] + [str(p.relative_to(args.output))
           for d in ('models', 'validation_raw') for p in (args.output / d).rglob('*') if p.is_file()])
    print(json.dumps({'stage': 'selection_frozen', 'selection': selection}), flush=True)


@torch.inference_mode()
def evaluate(args):
    manifests = setup(args)
    check_frozen(args.output, 'SELECTION_FROZEN.json')
    assert not (args.output / 'TEST_FROZEN.json').exists()
    text = torch.from_numpy(np.load(args.reference / 'features/text.npy')).cuda()
    selection = read(args.output / 'selection.json')
    chosen = {**selection['per_candidate'], **{'locked_' + m: r for m, r in selection['locked'].items()}}
    chosen.update({n: {'name': n, 'epoch': 0, 'base': n, 'weight': 0.} for n in REFERENCES})
    records = []
    for role, split in (('validation', 'eval'), ('historical_test', 'dev')):
        x = points(args, split)
        base = bases(args, role, split)
        folder = args.output / role
        folder.mkdir()
        raw = {}
        for spec in SPECS:
            for epoch in EPOCHS:
                raw[spec, epoch] = []
                for seed in SEEDS:
                    state = torch.load(args.output / f'models/{spec}/seed{seed}/e{epoch}.pt', map_location='cpu', weights_only=True)
                    assert state['config_sha256'] == file_hash(args.output / 'config.json')
                    model = ResidualPointNext(make_encoder(args), spec)
                    model.load_state_dict(state['encoder'], strict=True)
                    values = predict(model, x, text)
                    np.save(folder / f'{spec}_e{epoch}_seed{seed}_raw.npy', values)
                    if split == 'eval':
                        np.testing.assert_array_equal(values, np.load(args.output / f'validation_raw/{spec}_e{epoch}_seed{seed}.npy'))
                    raw[spec, epoch].append(values.astype(float))
                    del model
        for name, sel in chosen.items():
            values = [base[sel['name']]] * 3 if sel['name'] in base else [
                (1 - sel['weight']) * base[sel['base']] + sel['weight'] * z(v) for v in raw[sel['name'], sel['epoch']]]
            for seed, value in zip(SEEDS, values):
                np.save(folder / f'{name}_seed{seed}.npy', value)
                records.append({'role': role, 'method': name, 'seed': seed, 'selection': sel,
                                'summary': recount(value, manifests[split])[0]})
        print(json.dumps({'stage': role, 'locked': [r for r in records if r['role'] == role and r['method'].startswith('locked_')]}), flush=True)
    write_json(args.output / 'results.json', {'records': records})
    freeze(args.output, 'TEST_FROZEN.json', ['results.json', 'SELECTION_FROZEN.json'] + [str(p.relative_to(args.output))
           for d in ('validation', 'historical_test') for p in (args.output / d).iterdir()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('preflight', 'train-select', 'evaluate'))
    for name in ('output', 'reference', 'r4', 'r8', 'color', 'checkpoint', 'design'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with run_lock(args.output):
        {'preflight': preflight, 'train-select': train_select, 'evaluate': evaluate}[args.stage](args)
