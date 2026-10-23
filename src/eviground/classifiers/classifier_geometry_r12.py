"""Recompute pretrained geometry features on current room splits, then fit paired heads."""
from eviground.paths import source_file
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from eviground.experiments.audit_pair_train_test import recount
from eviground.experiments.audit_pair_views import z
from eviground.categories.category_common_v82 import category_vocabulary, file_hash, instance_seed, load_cloud, run_lock
from eviground.classifiers.classifier_residual_r9 import bases, choose
from eviground.classifiers.geometry_classifier_r12 import GeometryHead, SPECS, class_weights, predict
from eviground.classifiers.pointcloud_classifier_v82 import prepare_points
from eviground.experiments.run_rgb_experiment import check_frozen, freeze
from eviground.evaluation.run_storage import write_json

SEEDS = (42, 43, 44)
DIMENSIONS = {'uni3d': 1024, 'ulip2': 1280, 'shape': 512, 'rgb': 512}
REFERENCES = ('S0', 'r4_M', 'r8_M')
WEIGHTS = (0., .2, .4, .6, .8, 1.)
SOURCE_NAMES = ('classifier_geometry_r12.py', 'geometry_classifier_r12.py',
    'audit_classifier_geometry_r12.py', 'test_geometry_classifier_r12.py', 'uni3d_classifier.py',
    'ulip2_classifier.py', 'pointcloud_classifier_v82.py', 'category_common_v82.py',
    'audit_pair_train_test.py', 'audit_pair_views.py', 'classifier_residual_r9.py',
    'pointnext_residual_r9.py', 'classifier_backbone_r5.py', 'classifier_sweep_r3.py',
    'train_pair_views.py', 'run_rgb_experiment.py', 'rgb_classifier.py', 'shape_scale.py',
    'run_storage.py', 'category_v10/data.py', 'category_v10/run.py')


def read(path):
    return json.loads(Path(path).read_text())


def assets(args, name):
    root = args.uni3d if name == 'uni3d' else args.ulip2
    if name == 'uni3d':
        return (root / 'assets/weights/BAAI/Uni3D/modelzoo/uni3d-b/model.pt',
                root / 'assets/text_features_eva02e_197.npz', root / 'assets/source')
    return (root / 'weights/datasets/SFXX/ulip/ULIP-2/pretrained_models/'
                   'ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt',
            root / 'text_features_bigg_197.npz', root / 'source')


def setup(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
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
    paths += [args.reference / f'features/{n}' for n in ('text.npy', 'shape_train.npz',
              'shape_eval.npz', 'rgb_train/features.npz', 'rgb_eval/features.npz')]
    paths += [args.r4 / 'validation/locked_M_v0.npy', args.r8 / 'validation/locked_M.npy']
    for name in ('uni3d', 'ulip2'):
        checkpoint, text, source = assets(args, name)
        paths += [checkpoint, text, source / 'sources.json']
        paths += list(source.glob('*.py')) + list(source.glob('*.yaml'))
    config = {'version': 'complementary-geometry-r12', 'seeds': SEEDS, 'specs': SPECS,
        'dimensions': DIMENSIONS, 'epochs': 40, 'batch': 256, 'lr': .001, 'weight_decay': .01,
        'loss': 'train inverse-square-root class frequency CE, plus-one smoothing',
        'features': 'unit frozen Uni3D1024/ULIP21280; original PointNeXt512/RGB512; no text-space mixing',
        'sampling': 'instance_seed(42,scene,bbox_id),10000 XYZRGB,Z-up to Y-up,one object per forward',
        'precision': 'float32, TF32 off; original frozen encoders, no AMP or feature augmentation',
        'selection': '111 validation rows, last epoch40 only; P/B/M relative to r8_M',
        'reference_scale': 'original stored S0/r4/r8; only new scores per-object z-score',
        'source': {n: file_hash(source_file(n)) for n in SOURCE_NAMES},
        'inputs': {str(p): file_hash(p) for p in dict.fromkeys(paths)}, 'torch': str(torch.__version__),
        'limitations': 'historical evaluation exposed; pretrained scene exposure unknown; r8 rotation precision limit'}
    config = json.loads(json.dumps(config))
    if (root / 'config.json').exists():
        assert read(root / 'config.json') == config
    else:
        write_json(root / 'config.json', config)
        for folder in ('source_snapshot', 'manifests', 'features'):
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


def make_classifier(args, name):
    if name == 'uni3d':
        from eviground.classifiers.uni3d_classifier import Uni3DClassifier
        model = Uni3DClassifier(*assets(args, name))
    else:
        from eviground.classifiers.ulip2_classifier import ULIP2Classifier
        model = ULIP2Classifier(*assets(args, name))
    model.encoder.requires_grad_(False)
    assert not model.encoder.training
    return model


@torch.inference_mode()
def embedding(classifier, name, xyz, rgb, seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if name == 'uni3d':
        from eviground.classifiers.uni3d_classifier import prepare_uni3d_points
        points = torch.from_numpy(prepare_uni3d_points(xyz, rgb, seed)[None]).cuda()
        value = classifier.encoder(points[:, :, :3].contiguous(), points[:, :, 3:].contiguous())
    else:
        points = torch.from_numpy(prepare_points(xyz, rgb, 'openshape', seed)[None]).cuda()
        value = classifier.encoder(points)
    assert value.shape == (1, DIMENSIONS[name]) and torch.isfinite(value).all() and value.norm() > 0
    return F.normalize(value.float(), dim=-1)[0]


def instances(manifest, indices=None):
    groups = {}
    for i in range(len(manifest['samples'])) if indices is None else indices:
        row = manifest['samples'][i]
        groups.setdefault(row['scene'], []).append(i)
    sources = {r['scene']: r['cloud_sha256'] for r in manifest['sources']}
    for scene, indices in groups.items():
        assert file_hash(manifest['samples'][indices[0]]['cloud']) == sources[scene]
        xyz, rgb, ids = load_cloud(manifest['samples'][indices[0]]['cloud'])
        assert rgb is not None
        order = np.argsort(ids, kind='stable')
        unique, start, count = np.unique(ids[order], return_index=True, return_counts=True)
        lookup = {int(oid): order[s:s+n] for oid, s, n in zip(unique, start, count)}
        for i in indices:
            row = manifest['samples'][i]; selected = lookup[row['bbox_id']]
            yield i, row, xyz[selected], rgb[selected]


def preflight(args):
    manifests = setup(args); path = args.output / 'PREFLIGHT.json'
    assert not path.exists()
    manifest = manifests['train']; indices = [0, len(manifest['samples'])//2, len(manifest['samples'])-1]
    checks = {}
    for name in ('uni3d', 'ulip2'):
        classifier = make_classifier(args, name)
        before = {k: v.detach().cpu().clone() for k, v in classifier.encoder.state_dict().items()}
        rows = []
        for i, row, xyz, rgb in instances(manifest, indices):
            seed = instance_seed(42, row['scene'], row['bbox_id'])
            value = embedding(classifier, name, xyz, rgb, seed)
            repeat = embedding(classifier, name, xyz, rgb, seed)
            torch.testing.assert_close(value, repeat, atol=0, rtol=0)
            scores = (value[None] @ classifier.text_embeddings.T)[0].cpu().numpy()
            official = np.array(classifier.predict(xyz, rgb, seed)['scores'])
            np.testing.assert_allclose(scores, official, atol=2e-6, rtol=2e-6)
            np.testing.assert_array_equal(np.argsort(-scores, kind='stable')[:5], np.argsort(-official, kind='stable')[:5])
            rows.append({'index': i, 'score_bridge_max_abs': float(np.max(np.abs(scores-official)))})
        assert all(torch.equal(v.cpu(), before[k]) for k, v in classifier.encoder.state_dict().items())
        checks[name] = {'objects': rows, 'dimensions': DIMENSIONS[name], 'metadata': classifier.metadata,
                        'encoder_state': 'exactly_unchanged', 'repeat_features': 'exact'}
        del classifier, before; torch.cuda.empty_cache()
    write_json(path, {'status': 'passed', 'training_objects_only': indices, 'checks': checks})
    print(json.dumps(read(path)), flush=True)


def cache(args, split, name):
    manifests = setup(args)
    assert read(args.output / 'PREFLIGHT.json')['status'] == 'passed'
    if split == 'dev':
        check_frozen(args.output, 'SELECTION_FROZEN.json')
    folder = args.output / 'features'; path = folder / f'{name}_{split}.npy'
    assert not path.exists() and not path.with_suffix('.json').exists()
    classifier = make_classifier(args, name)
    before = {k: v.detach().cpu().clone() for k, v in classifier.encoder.state_dict().items()}
    text = classifier.text_embeddings.cpu().numpy(); text_path = folder / f'{name}_text.npy'
    if text_path.exists():
        np.testing.assert_array_equal(text, np.load(text_path))
    else:
        np.save(text_path, text)
    manifest = manifests[split]
    values = np.empty((len(manifest['samples']), DIMENSIONS[name]), dtype=np.float32)
    started = time.time()
    for number, (i, row, xyz, rgb) in enumerate(instances(manifest), 1):
        seed = instance_seed(42, row['scene'], row['bbox_id'])
        values[i] = embedding(classifier, name, xyz, rgb, seed).cpu().numpy()
        if number % 250 == 0:
            print(json.dumps({'stage': 'cache', 'encoder': name, 'split': split, 'objects': number,
                              'total': len(values), 'seconds': time.time()-started}), flush=True)
    assert np.isfinite(values).all()
    assert all(torch.equal(v.cpu(), before[k]) for k, v in classifier.encoder.state_dict().items())
    np.save(path, values)
    write_json(path.with_suffix('.json'), {'sha256': file_hash(path), 'shape': list(values.shape),
        'manifest_sha256': file_hash(args.output / f'manifests/{split}.json'),
        'text_sha256': file_hash(text_path), 'metadata': classifier.metadata,
        'encoder_state': 'exactly_unchanged', 'seconds': time.time()-started})
    print(json.dumps({'stage': 'cache_complete', 'encoder': name, 'split': split}), flush=True)


def load_data(args, split):
    data = {}
    for name in ('uni3d', 'ulip2'):
        path = args.output / f'features/{name}_{split}.npy'; seal = read(path.with_suffix('.json'))
        assert file_hash(path) == seal['sha256']
        assert file_hash(args.output / f'manifests/{split}.json') == seal['manifest_sha256']
        values = np.load(path); assert values.dtype == np.float32
        assert list(values.shape) == seal['shape'] and values.shape[1] == DIMENSIONS[name]
        assert np.isfinite(values).all()
        np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1., atol=2e-6, rtol=0)
        data[name] = torch.from_numpy(values).cuda()
    data['shape'] = torch.from_numpy(np.load(args.reference / f'features/shape_{split}.npz')['features']).cuda()
    path = args.reference / ('features/rgb_dev.npz' if split == 'dev' else f'features/rgb_{split}/features.npz')
    rgb = np.load(path); assert rgb['valid'].all()
    data['rgb'] = torch.from_numpy(rgb['features']).cuda()
    assert len({len(v) for v in data.values()}) == 1
    return data


def texts(args):
    result = {}
    for name in ('uni3d', 'ulip2'):
        path = args.output / f'features/{name}_text.npy'
        for split in ('train', 'eval'):
            assert file_hash(path) == read(args.output / f'features/{name}_{split}.json')['text_sha256']
        result[name] = torch.from_numpy(np.load(path)).cuda()
    return result


def fit(args, spec, seed, values, labels, text, weights):
    directory = args.output / f'models/{spec}/seed{seed}'; directory.mkdir(parents=True)
    torch.manual_seed(seed)
    model = GeometryHead(spec, DIMENSIONS, 197, text).cuda()
    initial = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    generator = torch.Generator(device='cuda').manual_seed(seed)
    history = []; started = time.time()
    for epoch in range(40):
        model.train(); total = 0.
        for indices in torch.randperm(len(labels), device='cuda', generator=generator).split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model({k: v[indices] for k, v in values.items()}), labels[indices], weight=weights)
            assert torch.isfinite(loss)
            loss.backward(); optimizer.step(); total += float(loss.detach()) * len(indices)
        history.append(total / len(labels))
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
    manifests = setup(args)
    assert read(args.output / 'PREFLIGHT.json')['status'] == 'passed'
    assert not (args.output / 'SELECTION_FROZEN.json').exists()
    train, validation, text = load_data(args, 'train'), load_data(args, 'eval'), texts(args)
    labels = torch.tensor([r['label'] for r in manifests['train']['samples']], device='cuda')
    weights = class_weights(labels, 197); raw = {}
    folder = args.output / 'validation_raw'; folder.mkdir()
    for spec in SPECS:
        arrays = []
        for seed in SEEDS:
            model, parameters = fit(args, spec, seed, train, labels, text, weights)
            scores = predict(model, validation); arrays.append(scores)
            np.save(folder / f'{spec}_e40_seed{seed}.npy', scores); del model
        raw[spec] = (arrays, parameters)
    grid, selection = make_grid(raw, bases(args, 'validation', 'eval'), manifests['eval'])
    assert len(grid) == 111
    write_json(args.output / 'validation_grid.json', grid); write_json(args.output / 'selection.json', selection)
    paths = ['validation_grid.json', 'selection.json', 'PREFLIGHT.json']
    paths += [str(p.relative_to(args.output)) for name in ('features', 'models', 'validation_raw')
              for p in (args.output / name).rglob('*') if p.is_file()]
    freeze(args.output, 'SELECTION_FROZEN.json', paths)
    print(json.dumps({'stage': 'selection_frozen', 'selection': selection}), flush=True)


def evaluate(args):
    manifests = setup(args); check_frozen(args.output, 'SELECTION_FROZEN.json')
    assert not (args.output / 'TEST_FROZEN.json').exists()
    selection = read(args.output / 'selection.json'); text = texts(args); records = []
    selected = {n: {'name': n, 'epoch': 0, 'base': n, 'weight': 0., 'parameters': 0} for n in REFERENCES}
    selected.update(selection['per_candidate']); selected.update({'locked_'+k: v for k, v in selection['locked'].items()})
    for role, split in (('validation', 'eval'), ('historical_test', 'dev')):
        folder = args.output / role; folder.mkdir()
        data, base, raw = load_data(args, split), bases(args, role, split), {}
        for spec in SPECS:
            for seed in SEEDS:
                path = args.output / f'models/{spec}/seed{seed}/e40.pt'
                assert file_hash(path) == read(path.with_suffix('.json'))['sha256']
                record = torch.load(path, weights_only=True, map_location='cpu')
                assert record['config_sha256'] == file_hash(args.output / 'config.json')
                model = GeometryHead(spec, DIMENSIONS, 197, text).cuda()
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
    write_json(args.output / 'results.json', {'records': records})
    paths = ['results.json'] + [str(p.relative_to(args.output)) for name in ('features', 'validation', 'historical_test')
                               for p in (args.output / name).rglob('*') if p.is_file()]
    freeze(args.output, 'TEST_FROZEN.json', paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('preflight', 'cache', 'train-select', 'evaluate'))
    for name in ('output', 'reference', 'r4', 'r8', 'uni3d', 'ulip2', 'design'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'eval', 'dev'))
    parser.add_argument('--encoder', choices=('uni3d', 'ulip2'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    with run_lock(args.output):
        if args.stage == 'cache':
            assert args.split is not None and args.encoder is not None
            cache(args, args.split, args.encoder)
        else:
            {'preflight': preflight, 'train-select': train_select, 'evaluate': evaluate}[args.stage](args)
