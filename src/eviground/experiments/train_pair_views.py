"""Room-nested training-side comparison of fixed and evidence-dependent fusion."""
from eviground.paths import source_file
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from eviground.categories.category_common_v82 import file_hash, run_lock
from eviground.evaluation.run_storage import write_json


SEEDS = (42, 43, 44)
MODES = ('confidence', 'view_quality', 'class_pair')
SETTINGS = [('ce', t) for t in (0., .1, .25, .5, 1.)] + [('balanced', 0.)]


def room_folds(rows, salt):
    mapping = {s['scene'].split('_')[0]: int.from_bytes(hashlib.sha256(
        f'{salt}:{s["scene"].split("_")[0]}'.encode()).digest()[:4], 'little') % 3 for s in rows}
    result = np.array([mapping[s['scene'].split('_')[0]] for s in rows])
    if set(result) != {0, 1, 2}:
        raise ValueError('Three nonempty room folds required')
    return result


def standardize(x):
    return (x-x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True, correction=0).clamp_min(1e-6)


def metrics(scores, labels):
    hit = scores.argmax(-1) == labels
    counts = torch.bincount(labels, minlength=scores.shape[-1])
    correct = torch.bincount(labels[hit], minlength=scores.shape[-1])
    present = counts > 0
    return {'count': len(labels), 'correct': int(hit.sum()), 'accuracy': float(hit.float().mean()),
            'class_mean': float((correct[present]/counts[present]).mean()),
            'classes_present': int(present.sum())}


def entropy(scores):
    logp = F.log_softmax(scores, dim=-1)
    return -(logp.exp()*logp).sum(-1) / math.log(scores.shape[-1])


def margin(scores):
    top = scores.topk(2, dim=-1).values
    return top[..., 0]-top[..., 1]


def gate_features(base, rgb, views, mode):
    """All gate inputs depend only on predictions; never accept target labels."""
    base, rgb = standardize(base), standardize(rgb)
    bp, rp = base.argmax(-1), rgb.argmax(-1)
    disagreement = bp != rp
    generic = torch.stack([base.max(-1).values, rgb.max(-1).values, margin(base), margin(rgb),
        entropy(base), entropy(rgb), (base-rgb).abs().mean(-1), disagreement.float()], -1)
    if mode == 'confidence':
        pb, pr = base.softmax(-1), rgb.softmax(-1)
        mixture = (pb+pr)/2
        js = .5*((pb*(pb.log()-mixture.log())).sum(-1)+(pr*(pr.log()-mixture.log())).sum(-1))
        tb, tr = base.topk(3, dim=-1).values, rgb.topk(3, dim=-1).values
        extra = torch.stack([pb.max(-1).values, pr.max(-1).values, margin(pb), margin(pr),
            tb[:, 0]-tb[:, 2], tr[:, 0]-tr[:, 2], js, F.cosine_similarity(base, rgb)], -1)
    else:
        v = standardize(views)
        if mode == 'view_quality':
            vm, ve, vx = margin(v), entropy(v), v.max(-1).values
            supported = v.gather(2, rp[:, None, None].expand(-1, 4, 1)).squeeze(2)
            extra = torch.stack([vm.mean(1), vm.std(1, correction=0), ve.mean(1), ve.std(1, correction=0),
                vx.mean(1), vx.std(1, correction=0), (v.argmax(-1) == rp[:, None]).float().mean(1),
                supported.std(1, correction=0)], -1)
        elif mode == 'class_pair':
            a = v.gather(2, rp[:, None, None].expand(-1, 4, 1)).squeeze(2)
            b = v.gather(2, bp[:, None, None].expand(-1, 4, 1)).squeeze(2)
            delta = a-b
            rows = torch.arange(len(base), device=base.device)
            extra = torch.stack([delta.mean(1), delta.std(1, correction=0), delta.min(1).values,
                delta.max(1).values, (delta > 0).float().mean(1),
                base[rows, rp]-base[rows, bp], rgb[rows, rp]-rgb[rows, bp], delta.abs().mean(1)], -1)
        else:
            raise ValueError(mode)
    return torch.cat([generic, extra], -1), disagreement


class Gate(nn.Module):
    def __init__(self, mean, std, weight):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.weight = weight
        self.network = nn.Sequential(nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features, disagreement):
        delta = .5*torch.tanh(self.network(((features-self.mean)/self.std).clamp(-5, 5))).squeeze(-1)
        return (self.weight+delta*disagreement).clamp(0, 1)


def prior(labels, classes):
    counts = torch.bincount(labels, minlength=classes).float()+1
    return counts/counts.sum()


def fit_head(x, y, text, loss_kind, seed, path):
    if path.exists():
        checkpoint = torch.load(path, map_location=x.device, weights_only=True)
        return checkpoint['weight'], checkpoint['bias']
    torch.manual_seed(seed)
    model = nn.Linear(x.shape[1], len(text), device=x.device)
    with torch.no_grad():
        model.weight.copy_(10*text)
        model.bias.zero_()
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    generator = torch.Generator(device=x.device).manual_seed(seed)
    offset = prior(y, len(text)).log() if loss_kind == 'balanced' else torch.zeros(len(text), device=x.device)
    history = []
    for epoch in range(40):
        total = 0.
        for indices in torch.randperm(len(y), generator=generator, device=x.device).split(256):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[indices])+offset, y[indices])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite expert loss')
            loss.backward()
            opt.step()
            total += float(loss.detach())*len(indices)
        history.append(total/len(y))
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {'weight': model.weight.detach().cpu(), 'bias': model.bias.detach().cpu(),
                  'loss': loss_kind, 'seed': seed, 'train_objects': len(y), 'epochs': 40}
    torch.save(checkpoint, path)
    write_json(path.with_suffix('.history.json'), history)
    print(json.dumps({'stage': 'expert', 'path': str(path), 'objects': len(y), 'loss': history[-1]}), flush=True)
    return model.weight.detach(), model.bias.detach()


def load_inputs(reference, device):
    m = json.loads((reference/'manifests/train.json').read_text())
    y = torch.tensor([r['label'] for r in m['samples']], device=device)
    if (y < 0).any():
        raise ValueError('Only official in-vocabulary training objects allowed')
    shape = torch.from_numpy(np.load(reference/'features/shape_train.npz')['features']).to(device)
    rgb = torch.from_numpy(np.load(reference/'features/rgb_train/features.npz')['features']).to(device)
    text = torch.from_numpy(np.load(reference/'features/text.npy')).to(device)
    assert len(shape) == len(rgb) == len(y)
    return m, y, shape, rgb, text


def freeze_plan(args, manifest):
    out = args.output
    config = {'version': 'class-pair-room-nested-pilot-r1', 'seeds': SEEDS, 'outer_folds': 3, 'inner_folds': 3,
        'expert_epochs': 40, 'expert_lr': .003, 'gate_epochs': 40, 'gate_lr': .001, 'weight_decay': .01,
        'batch': 256, 'gate_parameters': 289, 'gate_regularization': .01,
        'settings': SETTINGS, 'weights': [i/10 for i in range(11)], 'selection':
        'max three-seed OOF accuracy subject mean macro >= frozen3D; tie macro, CE, smaller tau, smaller weight',
        'reference_files': {n: file_hash(args.reference/n) for n in ('manifests/train.json',
            'features/shape_train.npz', 'features/rgb_train/features.npz', 'features/text.npy')},
        'sources': {n: file_hash(source_file(n)) for n in ('train_pair_views.py', 'cache_pair_views.py',
            'rgb_classifier.py', 'category_common_v82.py', 'run_storage.py')},
        'design_sha256': file_hash(args.design), 'evaluation':
        'training-side in-vocabulary objects only; original frozen backbone exposure unknown; no old dev/eval reads'}
    config = json.loads(json.dumps(config))
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text()) != config:
        raise ValueError('Frozen pilot changed; use fresh output')
    write_json(out/'config.json', config)
    snapshot = out/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in config['sources']:
        target = snapshot/name
        if not target.exists():
            shutil.copyfile(source_file(name), target)
        assert file_hash(target) == config['sources'][name]
    if not (snapshot/'design.md').exists():
        shutil.copyfile(args.design, snapshot/'design.md')
    folds = room_folds(manifest['samples'], 'pair-outer-20261019')
    split = {'outer': folds.tolist(), 'inner': {str(f): room_folds(manifest['samples'], f'pair-inner-{f}-20261019').tolist()
                                             for f in range(3)}}
    if (out/'folds.json').exists() and json.loads((out/'folds.json').read_text()) != split:
        raise ValueError('Room splits changed')
    write_json(out/'folds.json', split)
    return folds, split, config


def experts(args, m, y, shape, rgb, text, folds, split):
    for outer in range(3):
        train = np.flatnonzero(folds != outer)
        test = np.flatnonzero(folds == outer)
        inner = np.array(split['inner'][str(outer)])
        for seed in SEEDS:
            for kind in ('ce', 'balanced'):
                folder = args.output/f'outer{outer}/seed{seed}/{kind}'
                target = folder/'scores.npz'
                if target.exists():
                    continue
                oof = torch.zeros((len(y), len(text)), device=y.device)
                for fold in range(3):
                    fit = train[inner[train] != fold]
                    cal = train[inner[train] == fold]
                    w, b = fit_head(rgb[fit], y[fit], text, kind, seed, folder/f'inner{fold}.pt')
                    oof[cal] = F.linear(rgb[cal], w, b)
                w, b = fit_head(rgb[train], y[train], text, kind, seed, folder/'outer.pt')
                np.savez_compressed(target, train_indices=train, test_indices=test,
                    oof=oof[train].cpu().numpy(), test=F.linear(rgb[test], w, b).cpu().numpy())
    write_json(args.output/'EXPERTS_COMPLETE.json', {'config_sha256': file_hash(args.output/'config.json'),
        'files': {str(p.relative_to(args.output)): file_hash(p) for p in args.output.glob('outer*/seed*/*/*') if p.is_file()}})


def choose(base, scores, labels, priors, settings):
    baseline = metrics(base, labels)
    rows = []
    for kind, tau in settings:
        adjusted = [standardize(s-tau*priors.log()) for s in scores[kind]]
        for weight in [i/10 for i in range(11)]:
            values = [metrics((1-weight)*base+weight*s, labels) for s in adjusted]
            rows.append({'kind': kind, 'tau': tau, 'weight': weight,
                'total_correct': sum(v['correct'] for v in values),
                'mean_macro': float(np.mean([v['class_mean'] for v in values]))})
    eligible = [r for r in rows if r['mean_macro'] >= baseline['class_mean']-1e-7]
    best = min(eligible, key=lambda r: (-r['total_correct'], -r['mean_macro'],
                                     r['kind'] != 'ce', r['tau'], r['weight']))
    return best, rows


def view_scores(folder, indices, inner, views, train):
    output = torch.zeros((len(indices), 4, 197), device=views.device)
    if train:
        for fold in range(3):
            mask = inner[indices] == fold
            head = torch.load(folder/f'inner{fold}.pt', map_location=views.device, weights_only=True)
            output[mask] = F.linear(views[indices[mask]], head['weight'], head['bias'])
    else:
        head = torch.load(folder/'outer.pt', map_location=views.device, weights_only=True)
        output = F.linear(views[indices], head['weight'], head['bias'])
    return output


def fit_gate(features, disagreement, base, rgb, labels, weight, seed, path):
    torch.manual_seed(seed)
    mean = features.mean(0)
    std = features.std(0, correction=0).clamp_min(1e-6)
    model = Gate(mean, std, weight).to(features.device)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    generator = torch.Generator(device=features.device).manual_seed(seed)
    history = []
    for epoch in range(40):
        total = 0.
        for i in torch.randperm(len(labels), generator=generator, device=features.device).split(256):
            opt.zero_grad(set_to_none=True)
            w = model(features[i], disagreement[i])
            loss = F.cross_entropy((1-w[:, None])*base[i]+w[:, None]*rgb[i], labels[i])
            loss = loss+.01*((w-weight)**2).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite gate loss')
            loss.backward()
            opt.step()
            total += float(loss.detach())*len(i)
        history.append(total/len(labels))
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'weight': weight, 'seed': seed, 'epochs': 40}, path)
    write_json(path.with_suffix('.history.json'), history)
    model.eval()
    return model


def gates(args, m, y, shape, rgb, text, folds, split):
    complete = json.loads((args.output/'EXPERTS_COMPLETE.json').read_text())
    for name, digest in complete['files'].items():
        assert file_hash(args.output/name) == digest
    seal = json.loads((args.views/'COMPLETE.json').read_text())
    assert file_hash(args.views/'features.npz') == seal['features_sha256']
    views = torch.from_numpy(np.load(args.views/'features.npz')['features']).to(y.device)
    assert views.shape == (len(y), 4, 512)
    base = standardize(10*shape @ text.T)
    records = []
    for outer in range(3):
        train, test = np.flatnonzero(folds != outer), np.flatnonzero(folds == outer)
        inner = np.array(split['inner'][str(outer)])
        probabilities = prior(y[train], len(text))
        scores = {k: [] for k in ('ce', 'balanced')}
        for kind in scores:
            for seed in SEEDS:
                archive = np.load(args.output/f'outer{outer}/seed{seed}/{kind}/scores.npz')
                np.testing.assert_array_equal(archive['train_indices'], train)
                scores[kind].append(torch.from_numpy(archive['oof']).to(y.device))
        a, agrid = choose(base[train], scores, y[train], probabilities, [('ce', 0.)])
        b, bgrid = choose(base[train], scores, y[train], probabilities, SETTINGS)
        write_json(args.output/f'outer{outer}/selection.json', {'A': a, 'B': b, 'A_grid': agrid, 'B_grid': bgrid})
        for seed_index, seed in enumerate(SEEDS):
            dest = args.output/f'outer{outer}/seed{seed}/comparison'
            if dest.exists():
                raise ValueError('Comparison exists; preserve it and use a new output')
            dest.mkdir()
            test_values = {'S0': base[test]}
            for name, selected in (('A', a), ('B', b)):
                archive = np.load(args.output/f'outer{outer}/seed{seed}/{selected["kind"]}/scores.npz')
                adjusted = standardize(torch.from_numpy(archive['test']).to(y.device)-selected['tau']*probabilities.log())
                test_values[name] = (1-selected['weight'])*base[test]+selected['weight']*adjusted
            folder = args.output/f'outer{outer}/seed{seed}/{b["kind"]}'
            raw_train = scores[b['kind']][seed_index]-b['tau']*probabilities.log()
            archive = np.load(folder/'scores.npz')
            raw_test = torch.from_numpy(archive['test']).to(y.device)-b['tau']*probabilities.log()
            vt = view_scores(folder, train, inner, views, True)-b['tau']*probabilities.log()
            ve = view_scores(folder, test, inner, views, False)-b['tau']*probabilities.log()
            for method, mode in zip(('C', 'D', 'E'), MODES):
                ft, dt = gate_features(base[train], raw_train, vt, mode)
                fe, de = gate_features(base[test], raw_test, ve, mode)
                model = fit_gate(ft, dt, base[train], standardize(raw_train), y[train], b['weight'], seed, dest/f'{method}.pt')
                with torch.inference_mode():
                    weights = model(fe, de)
                    test_values[method] = (1-weights[:, None])*base[test]+weights[:, None]*standardize(raw_test)
                np.save(dest/f'{method}_weights.npy', weights.cpu().numpy())
            for name, values in test_values.items():
                np.save(dest/f'{name}.npy', values.cpu().numpy())
                row = {'outer': outer, 'seed': seed, 'method': name, **metrics(values, y[test])}
                records.append(row)
                print(json.dumps({'stage': 'outer_result', **row}), flush=True)
    write_json(args.output/'results.json', {'records': records, 'scope': 'training-side in-vocabulary nested room CV'})
    write_json(args.output/'RESULTS_COMPLETE.json', {'config_sha256': file_hash(args.output/'config.json'),
        'views_sha256': seal['features_sha256'], 'files': {str(p.relative_to(args.output)): file_hash(p)
        for pattern in ('outer*/selection.json', 'outer*/seed*/comparison/*', 'results.json')
        for p in args.output.glob(pattern) if p.is_file()}})


def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with run_lock(args.output):
        m, y, shape, rgb, text = load_inputs(args.reference, args.device)
        folds, split, config = freeze_plan(args, m)
        if args.stage == 'experts':
            experts(args, m, y, shape, rgb, text, folds, split)
        else:
            gates(args, m, y, shape, rgb, text, folds, split)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('experts', 'gates'))
    for name in ('reference', 'output', 'views', 'design'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    main(parser.parse_args())
