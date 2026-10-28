"""Run unchanged r8 B/M scoring on GT clouds using an audited CPU backend."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from r8_cpu_backend import load_encoder

PASSES = tuple((seed, 0.) for seed in range(42, 49)) + ((42, -10.), (42, 10.))
VERSION = 'pointnext-sampling-vote-r8-export-v1'
SCORE_TYPE = 'weighted_standardized_3d_rgb_fusion_not_probability'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def standardize(values):
    return ((values - values.mean(-1, keepdim=True)) /
            values.std(-1, keepdim=True, correction=0).clamp_min(1e-6))


def load_heads(root):
    families = []
    for family in ('old_rgb_mlp_ce', 'old_rgb_mlp_weighted'):
        heads = []
        for seed in (42, 43, 44):
            record = torch.load(root / f'deployment_models/{family}/seed{seed}/model.pt',
                                map_location='cpu', weights_only=True)
            assert record['architecture'] == 'rgb_mlp'
            state = {key.removeprefix('head.'): value for key, value in record['state_dict'].items()}
            classes, width = state['4.weight'].shape
            head = nn.Sequential(nn.LayerNorm(state['0.weight'].numel()),
                                 nn.Linear(state['0.weight'].numel(), width), nn.GELU(),
                                 nn.Dropout(.3), nn.Linear(width, classes))
            head.load_state_dict(state, strict=True)
            heads.append(head.eval())
        families.append(heads)
    return families


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--frozen-model-source', type=Path, required=True)
    parser.add_argument('--runtime-source', type=Path, required=True)
    parser.add_argument('--scene-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bridge-export', type=Path, required=True)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--coverage', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime_source.resolve() / 'local'))
    from category_common_v82 import instance_seed, load_cloud
    from pointcloud_classifier_v82 import prepare_points
    from rgb_classifier import RENDER, load_clip, render_views, verify_text

    torch.set_num_threads(4)
    torch.manual_seed(0)
    model_record = json.loads((args.models / 'MODEL_SOURCES.json').read_text())
    for row in model_record['files']:
        if digest(args.models / row['file']) != row['sha256']:
            raise ValueError(f"Model hash changed: {row['file']}")
    selected = json.loads((args.models / 'r8_selection.json').read_text())['locked']['B']
    assert (selected['name'], selected['rgb_weight'], selected['passes']) == ('sample_yaw9', .6, 9)
    text = np.load(args.models / 'text.npy', allow_pickle=False).astype(np.float64)
    vocabulary = json.loads((args.models / 'vocabulary.json').read_text())
    source_files = list((args.frozen_model_source / 'models').rglob('*.py'))
    source_files += list((args.frozen_model_source / 'models').rglob('*.yaml'))
    identity = {'method': 'original r8 B/M; CPU operator backend',
                'script_sha256': digest(__file__), 'backend_sha256': digest(Path(__file__).with_name('r8_cpu_backend.py')),
                'model_files': model_record['files'],
                'model_source_sha256': {str(p.relative_to(args.frozen_model_source)): digest(p) for p in source_files},
                'torch': torch.__version__, 'bridge_export_sha256': digest(args.bridge_export / 'predictions.json')}
    config_path = args.output / 'CPU_PROTOCOL.json'
    if config_path.exists() and json.loads(config_path.read_text()) != identity:
        raise ValueError('CPU protocol changed; choose a new output')
    save(config_path, identity)
    encoder = load_encoder(args.frozen_model_source, args.models / 'pnext_cls.pth')
    clip, tokenizer = load_clip(args.models / 'clip_vit_b16.bin', 'cpu')
    alignment = verify_text(clip, tokenizer, vocabulary, torch.from_numpy(text), 'cpu')
    save(args.output / 'TEXT_BRIDGE.json', alignment)
    families = load_heads(args.models)
    mean = torch.tensor([.48145466, .4578275, .40821073])[None, :, None, None]
    std = torch.tensor([.26862954, .26130258, .27577711])[None, :, None, None]

    def classify(scene, oid, xyz, rgb, ids):
        mask = ids == oid
        points = xyz[mask]
        samples = []
        for seed, degrees in PASSES:
            value = prepare_points(points, None, 'pointnext', instance_seed(seed, scene, oid))
            angle = np.deg2rad(degrees)
            x, y = value[:, 0].copy(), value[:, 1].copy()
            value[:, 0] = x * np.cos(angle) - y * np.sin(angle)
            value[:, 1] = x * np.sin(angle) + y * np.cos(angle)
            samples.append(value)
        features = F.normalize(encoder(torch.from_numpy(np.stack(samples))).float(), dim=-1).numpy()
        logits = 10 * features.astype(np.float64).mean(0) @ text.T
        shape = (logits - logits.mean()) / max(logits.std(), 1e-6)
        images, coverage = render_views(points, rgb[mask], instance_seed(RENDER['seed'], scene, oid))
        if images is None or min(coverage) <= 0:
            raise ValueError(f'Invalid RGB views: {scene}/{oid}')
        tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255
        merged = F.normalize(clip.encode_image((tensor - mean) / std, normalize=True).mean(0, keepdim=True), dim=-1)
        ensembles = [torch.stack([standardize(head(merged)) for head in heads]).mean(0) for heads in families]
        rgb_scores = standardize(.5 * ensembles[0] + .5 * ensembles[1]).numpy()[0].astype(np.float64)
        score = .4 * shape + .6 * rgb_scores
        if not np.isfinite(score).all():
            raise ValueError(f'Nonfinite score: {scene}/{oid}')
        return score

    historical = json.loads((args.bridge_export / 'predictions.json').read_text())['results']
    expected = np.load(args.bridge_export / 'all_scores.npy', allow_pickle=False)
    # Fixed, label-blind coverage across the original ordered export.
    indices = np.linspace(0, len(historical) - 1, 24, dtype=int)
    records = []
    for index in indices:
        row = historical[index]
        scene, oid = row['scene'], row['bbox_id']
        path = args.output / 'bridge_objects' / f'{scene}_{oid}.json'
        if path.exists():
            record = json.loads(path.read_text())
        else:
            started = time.monotonic()
            score = classify(scene, oid, *load_cloud(args.scene_root / scene / 'scene.npz'))
            top = np.argsort(-score, kind='stable')[:5]
            record = {'scene': scene, 'bbox_id': oid, 'export_index': int(index),
                      'max_abs_score': float(np.max(np.abs(score - expected[index]))),
                      'top5_equal': [vocabulary[i] for i in top] == [h['category'] for h in row['top5']],
                      'elapsed_s': time.monotonic() - started}
            save(path, record)
        records.append(record)
        print(json.dumps({'bridge': record}), flush=True)
    passed = all(r['top5_equal'] and r['max_abs_score'] < .001 for r in records)
    save(args.output / 'CPU_BRIDGE.json', {'passed': passed, 'records': records,
                                        'criterion': '24 fixed label-blind objects; identical ordered Top-5 and all-score max_abs < 0.001'})
    if not passed:
        raise ValueError('Historical CPU bridge failed; new classifications are not authorized by this protocol')
    if not args.generate:
        return
    if args.coverage is None:
        parser.error('--generate needs --coverage')
    scenes = json.loads(args.coverage.read_text())['missing_category_scenes']
    rows, matrices, sources = [], [], []
    for scene in scenes:
        cloud = args.scene_root / scene / 'scene.npz'
        xyz, rgb, ids = load_cloud(cloud)
        sources.append({'scene': scene, 'cloud_sha256': digest(cloud)})
        for oid in sorted(set(map(int, np.unique(ids))) - {-1}):
            path = args.output / 'objects' / f'{scene}_{oid}.json'
            if path.exists():
                item = json.loads(path.read_text())
                if item['cloud_sha256'] != sources[-1]['cloud_sha256']:
                    raise ValueError(f'Cloud changed: {scene}')
            else:
                started = time.monotonic()
                score = classify(scene, oid, xyz, rgb, ids)
                low, high = xyz[ids == oid].min(0), xyz[ids == oid].max(0)
                top = np.argsort(-score, kind='stable')[:5]
                result = {'scene': scene, 'bbox_id': oid, 'status': 'completed',
                          'predicted_object': vocabulary[top[0]],
                          'top5': [{'category': vocabulary[i], 'score': float(score[i]), 'rank': rank}
                                   for rank, i in enumerate(top, 1)],
                          'bbox_3d': np.r_[(low + high) / 2, high - low].tolist(),
                          'point_count': int((ids == oid).sum()), 'score_type': SCORE_TYPE}
                item = {'result': result, 'scores': score.tolist(), 'cloud_sha256': sources[-1]['cloud_sha256'],
                        'elapsed_s': time.monotonic() - started}
                save(path, item)
            rows.append(item['result'])
            matrices.append(item['scores'])
            print(json.dumps({'classified': [scene, oid], 'completed': len(rows)}), flush=True)
    config = {'version': VERSION, 'score_type': SCORE_TYPE, 'sources': sources,
              'selection': selected, 'cpu_protocol_sha256': digest(config_path),
              'cpu_bridge_sha256': digest(args.output / 'CPU_BRIDGE.json')}
    save(args.output / 'config.json', config)
    save(args.output / 'predictions.json', {'version': VERSION, 'results': rows, 'summary': {
        'complete': True, 'errors': 0, 'count': len(rows), 'scenes': len(scenes), 'score_type': SCORE_TYPE}})
    save(args.output / 'vocabulary.json', vocabulary)
    np.save(args.output / 'all_scores.npy', np.asarray(matrices))
    save(args.output / 'EXPORT_VERIFIED.json', {'status': 'passed', 'files': {
        name: digest(args.output / name) for name in ('predictions.json', 'config.json', 'vocabulary.json', 'all_scores.npy')}})


if __name__ == '__main__':
    main()
