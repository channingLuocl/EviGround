"""Audit RGB assets and describe split differences without selecting a model."""
import argparse
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch

from eviground.categories.category_common_v82 import file_hash, load_cloud, normalized_category, run_lock
from eviground.geometry.shape_scale import frozen_logits, load_global, load_split
from eviground.evaluation.run_storage import write_json


normalize = lru_cache(None)(normalized_category)


def audit(args):
    prior = json.loads((args.prior / 'config.json').read_text())
    text, weight = load_global(args, 42, prior)
    reports = {}
    with run_lock(args.output):
        for split in ('train', 'dev', 'eval'):
            data = load_split(args, split, prior)
            manifest = data['manifest']
            base, _ = frozen_logits(data, text, weight)
            vocabulary = np.array([normalize(c) for c in manifest['vocabulary']])
            predicted = vocabulary[base.argmax(-1)]
            targets = np.array([normalize(r['category']) for r in manifest['samples']])
            hit = predicted == targets
            classes = {str(c): {'support': int((targets == c).sum()),
                               'accuracy': float(hit[targets == c].mean())} for c in np.unique(targets)}
            by_scene = defaultdict(list)
            for i, row in enumerate(manifest['samples']):
                by_scene[row['scene']].append(i)
            source_hashes = {r['scene']: r['cloud_sha256'] for r in manifest['sources']}
            counts, spans, per_scene = [], [], []
            rgb_missing, grayscale, bad_spans, annotation_differences = 0, 0, [], []
            raw_checked = 0
            for scene, indices in sorted(by_scene.items()):
                row = manifest['samples'][indices[0]]
                cloud_path = Path(row['cloud'])
                if file_hash(cloud_path) != source_hashes[scene]:
                    raise ValueError(f'Scene content changed: {scene}')
                xyz, rgb, ids = load_cloud(cloud_path)
                if rgb is not None:
                    grayscale += int(np.all(rgb[:, 0] == rgb[:, 1]) and np.all(rgb[:, 1] == rgb[:, 2]))
                annotations = {r['bbox_id']: normalize(r['target']) for r in
                               json.loads(cloud_path.with_name('objects_gt.json').read_text())}
                raw_path = args.feature_root / 'raw' / scene / f'{scene}.aggregation.json'
                raw = ({int(r['objectId']): normalize(r['label']) for r in
                        json.loads(raw_path.read_text())['segGroups']} if raw_path.exists() else None)
                order = np.argsort(ids, kind='stable')
                unique, start, number = np.unique(ids[order], return_index=True, return_counts=True)
                lookup = {int(oid): order[s:s+n] for oid, s, n in zip(unique, start, number)}
                scene_counts = []
                for i in indices:
                    row = manifest['samples'][i]
                    selected = lookup[row['bbox_id']]
                    points = xyz[selected]
                    extent = np.ptp(points, axis=0)
                    expected = data['meta']['records'][i]['bbox_3d'][3:]
                    if not np.allclose(extent, expected, rtol=1e-5, atol=1e-5):
                        bad_spans.append([scene, row['bbox_id']])
                    category = normalize(row['category'])
                    if annotations[row['bbox_id']] != category:
                        annotation_differences.append([scene, row['bbox_id'], 'prepared_GT'])
                    if raw is not None:
                        raw_checked += 1
                        if raw[row['bbox_id']] != category:
                            annotation_differences.append([scene, row['bbox_id'], 'raw_aggregation'])
                    rgb_missing += int(rgb is None)
                    counts.append(len(points))
                    spans.append(extent.tolist())
                    scene_counts.append(len(points))
                per_scene.append({'scene': scene, 'objects': len(indices), 'rgb_present': rgb is not None,
                                  'rgb_range': [float(rgb.min()), float(rgb.max())] if rgb is not None else None,
                                  'minimum_points': min(scene_counts), 'cloud_sha256': source_hashes[scene]})
                if len(per_scene) % 25 == 0:
                    print(json.dumps({'split': split, 'scenes_checked': len(per_scene), 'total': len(by_scene)}), flush=True)
            reports[split] = {'objects': len(targets), 'scenes': len(by_scene), 'base_accuracy': float(hit.mean()),
                             'known_accuracy': float(hit[data['labels'].numpy() >= 0].mean()),
                             'OOV': int((data['labels'].numpy() < 0).sum()), 'classes': classes,
                             'macro_accuracy': float(np.mean([c['accuracy'] for c in classes.values()])),
                             'raw_annotation_checked_objects': raw_checked,
                             'annotation_differences': annotation_differences, 'bbox_differences': bad_spans,
                             'RGB_missing_objects': rgb_missing, 'grayscale_scenes': grayscale,
                             'point_count_quantiles': np.quantile(counts, [0, .1, .5, .9, 1]).tolist(),
                             'span_quantiles': np.quantile(spans, [.1, .5, .9], axis=0).tolist(), 'scene_checks': per_scene}
            write_json(args.output / f'{split}.json', reports[split])
            if annotation_differences or bad_spans:
                raise ValueError(f'Input/label mismatch in {split}')
        common = sorted(set(reports['dev']['classes']) & set(reports['eval']['classes']))
        eval_counts = np.array([reports['eval']['classes'][c]['support'] for c in common])
        dev_counts = np.array([reports['dev']['classes'][c]['support'] for c in common])
        dev_acc = np.array([reports['dev']['classes'][c]['accuracy'] for c in common])
        eval_acc = np.array([reports['eval']['classes'][c]['accuracy'] for c in common])
        summary = {'status': 'passed', 'purpose': 'descriptive asset/distribution audit; no hyperparameter search',
                   'code_sha256': file_hash(Path(__file__)),
                   'splits': {s: {k: v for k, v in r.items() if k not in ('scene_checks', 'classes')} for s, r in reports.items()},
                   'common_class_diagnostic': {'classes': len(common), 'dev_objects': int(dev_counts.sum()),
                       'eval_objects': int(eval_counts.sum()),
                       'dev_accuracy_weighted_by_eval_class_counts': float(np.average(dev_acc, weights=eval_counts)),
                       'eval_accuracy_weighted_by_dev_class_counts': float(np.average(eval_acc, weights=dev_counts)),
                       'dev_macro_on_common_classes': float(dev_acc.mean()), 'eval_macro_on_common_classes': float(eval_acc.mean())},
                   'limitation': 'Room-disjoint for new heads; original backbone pretraining scene membership is not certified.'}
        write_json(args.output / 'summary.json', summary)
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('feature-root', 'prior', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    audit(args)
