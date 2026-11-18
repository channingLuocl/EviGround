"""Rerun only the released Mask3D category stage and retain foreground scores."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from category_metrics import associate, model_class_names, ranked_categories, summarize, unique_mask_indices


COLOR_MEAN = (.47793125906962, .4303257521323044, .3749598901421883)
COLOR_STD = (.2834475483823543, .27566157565723015, .27018971370874995)
CHECKPOINT_SHA256 = '6a0c4347616a27348bed32ff1ae225cab0087e0a0148f36d1023ca2d46f82534'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def load_model(source, checkpoint):
    import hydra
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0, str(source))
    config = OmegaConf.create({
        'general': {'num_targets': 201, 'train_on_segments': True},
        'data': {'in_channels': 3, 'num_labels': 200, 'voxel_size': .02},
        'model': OmegaConf.load(source / 'conf/model/mask3d.yaml')})
    config.model.num_queries = 150
    model = hydra.utils.instantiate(config.model, _recursive_=False)
    weights = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
    model.load_state_dict({k.removeprefix('model.'): v for k, v in weights.items()}, strict=True)
    from datasets.scannet200.scannet200_constants import CLASS_LABELS_200
    return model.cuda().eval(), model_class_names(CLASS_LABELS_200)


def predict(model, xyz, rgb, segments):
    import torch
    import MinkowskiEngine as ME
    from sklearn.cluster import DBSCAN
    from torch_scatter import scatter_mean
    # Match the released validation voxelization and color normalization.
    coordinates = np.floor(xyz / .02)
    _, unique, inverse = ME.utils.sparse_quantize(
        coordinates=coordinates, return_index=True, return_inverse=True)
    unique, inverse = np.asarray(unique), np.asarray(inverse)
    _, low_segments = np.unique(segments[unique], return_inverse=True)
    _, full_segments = np.unique(segments, return_inverse=True)
    features = ((rgb[unique].astype(np.float32) / 255 - np.asarray(COLOR_MEAN)) /
                np.asarray(COLOR_STD)).astype(np.float32)
    batch_coords = ME.utils.batched_coordinates([coordinates[unique].astype(np.int32)])
    sparse = ME.SparseTensor(torch.from_numpy(features).cuda(), coordinates=batch_coords.cuda())
    with torch.inference_mode():
        output = model(sparse, point2segment=[torch.from_numpy(low_segments).cuda()],
                       raw_coordinates=torch.from_numpy(xyz[unique]).cuda(), is_eval=True)
        probabilities = output['pred_logits'][0].softmax(-1)[..., :-1].cpu()
        low_logits = output['pred_masks'][0].cpu()[low_segments]
    del output, sparse
    # Every DBSCAN component retains the full vector of its parent query.
    split_logits, parent_ids = [], []
    for query in range(low_logits.shape[1]):
        positive = low_logits[:, query] > 0
        if not positive.any():
            continue
        clusters = DBSCAN(eps=.95, min_samples=1, n_jobs=4).fit_predict(xyz[unique][positive])
        for cluster in np.unique(clusters):
            keep = torch.zeros(len(unique), dtype=torch.bool)
            keep[positive] = torch.from_numpy(clusters == cluster)
            split_logits.append(low_logits[:, query] * keep)
            parent_ids.append(query)
    if not split_logits:
        return {'masks': np.empty((len(xyz), 0), dtype=bool), 'scores': np.empty(0),
                'class_probabilities': np.empty((0, 200)), 'query_ids': np.empty(0, dtype=int)}
    mask_logits = torch.stack(split_logits, 1)
    split_probabilities = probabilities[parent_ids]
    values, indices = split_probabilities.flatten().topk(min(750, split_probabilities.numel()), sorted=True)
    components, classes = indices // 200, indices % 200
    positive = mask_logits > 0
    quality = (mask_logits.sigmoid() * positive).sum(0) / (positive.sum(0) + 1e-6)
    scores = values * quality[components]
    # Query/class alternatives have the same geometry; keep its best label.
    selected = {}
    for i in scores.argsort(descending=True, stable=True).tolist():
        selected.setdefault(int(components[i]), i)
    chosen = list(selected.values())
    chosen_components = components[chosen]
    proposal_probabilities = split_probabilities[chosen_components].numpy()
    if not np.array_equal(classes[chosen].numpy(), proposal_probabilities.argmax(1)):
        raise ValueError('Selected class is not the original Top-1')
    if np.any(proposal_probabilities.argmax(1) >= 198):
        raise ValueError('An unused ScanNet200 output head was selected')
    masks = positive[:, chosen_components][inverse].float()
    masks = (scatter_mean(masks, torch.from_numpy(full_segments), dim=0) > .5)[full_segments].numpy()
    proposal_scores = scores[chosen].numpy()
    keep = unique_mask_indices(masks, proposal_scores)
    return {'masks': masks[:, keep], 'scores': proposal_scores[keep],
            'class_probabilities': proposal_probabilities[keep],
            'query_ids': np.asarray(parent_ids)[chosen_components.numpy()][keep]}


def evaluate(data, output, scenes):
    targets = json.loads((data / 'targets.json').read_text())
    rows = []
    for scene in scenes:
        prediction = json.loads((output / 'scenes' / f'{scene}.json').read_text())
        with np.load(output / 'scenes' / f'{scene}.npz') as saved:
            masks = np.unpackbits(saved['packed_masks'], axis=0, count=int(saved['vertices'])).astype(bool)
        with np.load(data / 'gt' / f'{scene}.npz') as saved:
            ids = saved['instance_ids']
        scene_targets = [r for r in targets if r['scene_id'] == scene]
        matches = associate(masks, ids, [r['object_id'] for r in scene_targets])
        for i, target in enumerate(scene_targets):
            row = dict(target, matched=i in matches, proposal_id=None, mask_iou=None, top5=[])
            if i in matches:
                j, iou = matches[i]
                row.update(proposal_id=j, mask_iou=iou, top5=prediction['proposals'][j]['top5'])
            rows.append(row)
    metrics = summarize(rows)
    metrics['scenes'] = len(scenes)
    metrics['query_label_metrics'] = summarize([dict(r, gt_label=r['query_label']) for r in rows])
    write_json(output / 'object_results.json', rows)
    write_json(output / 'metrics.json', metrics)
    print(json.dumps(metrics), flush=True)


def run(args):
    manifest = json.loads((args.data / 'manifest.json').read_text())
    scenes = [r['scene_id'] for r in manifest['scenes']]
    if args.limit:
        scenes = scenes[:args.limit]
    if args.evaluate_only:
        evaluate(args.data, args.output, scenes)
        return
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    if sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Checkpoint differs from pinned ScanNet200 validation weights')
    protocol = {
        'scope': 'Mask3D categories only; no VLM, language queries, training, or r8 classifier',
        'checkpoint_sha256': CHECKPOINT_SHA256, 'checkpoint_split': 'ScanNet200 validation',
        'mask3d_commit': '11bd5ff94477ff7194e9a7c52e9fae54d73ac3b5',
        'manifest_sha256': sha256(args.data / 'manifest.json'),
        'script_sha256': sha256(Path(__file__)),
        'metrics_code_sha256': sha256(Path(__file__).with_name('category_metrics.py')),
        'prediction': '150 queries; 0.02m voxels; segment masks; DBSCAN eps=.95 min1; top750 query/class pairs',
        'duplicate_geometry': 'Keep highest-confidence identical nonempty mask before matching',
        'seeground_proposal_filter': 'Drop proposals whose Top-1 is wall, floor, ceiling, object, or objects',
        'class_scores': 'Softmax over 201 logits, drop no-object; preserve all 200 foreground probabilities',
        'class_mapping': 'Official label_offset=2, chair clamp; 198 real classes, 2 unused heads excluded from ranks',
        'top5': 'Extension from same parent-query scores, not a metric originally reported by SeeGround',
        'association': 'Class-blind one-to-one mask IoU >= .5; maximize count, then IoU',
        'denominator': 'Distinct referenced GT objects; unmatched count wrong',
        'exact_labels': 'Original ScanNet aggregation strings; query-label results also saved; no aliases',
        'qualification': 'Released Mask3D rerun; authors original NPZ scores unavailable, not guaranteed identical proposals',
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(), 'seed': 42,
        'scenes': scenes}
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Protocol changed; choose a new output directory')
    write_json(protocol_path, protocol)
    model, names = load_model(args.source, args.checkpoint)
    for number, scene in enumerate(scenes, 1):
        destination = args.output / 'scenes' / f'{scene}.json'
        if destination.exists():
            continue
        started = time.monotonic()
        input_path = args.data / 'inputs' / f'{scene}.npz'
        expected = next(r['input_sha256'] for r in manifest['scenes'] if r['scene_id'] == scene)
        if sha256(input_path) != expected:
            raise ValueError(f'Input changed: {scene}')
        seed = int.from_bytes(scene.encode(), 'little') % (2**32 - 42) + 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        with np.load(input_path) as cloud:
            result = predict(model, cloud['xyz'], cloud['rgb'], cloud['segments'])
        indices, labels = ranked_categories(result['class_probabilities'], names)
        # Match SeeGround's ScanRefer lookup-table category filter exactly.
        keep = [j for j, row in enumerate(labels)
                if row[0] not in {'wall', 'floor', 'ceiling', 'object', 'objects'}]
        result['masks'] = result['masks'][:, keep]
        for key in ('scores', 'class_probabilities', 'query_ids'):
            result[key] = result[key][keep]
        indices, labels = indices[keep], [labels[j] for j in keep]
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination.with_suffix('.npz'),
                            packed_masks=np.packbits(result['masks'], axis=0), vertices=len(result['masks']),
                            scores=result['scores'], class_probabilities=result['class_probabilities'],
                            query_ids=result['query_ids'], top5_class_indices=indices)
        proposals = [{'proposal_id': j, 'query_id': int(result['query_ids'][j]),
                      'score': float(result['scores'][j]), 'top1': names[int(indices[j, 0])],
                      'top5': label, 'top5_scores': result['class_probabilities'][j, indices[j]].tolist()}
                     for j, label in enumerate(labels)]
        write_json(destination, {'scene_id': scene, 'proposals': proposals,
                                'seconds': time.monotonic() - started,
                                'arrays_sha256': sha256(destination.with_suffix('.npz'))})
        status = {'completed_scenes': number, 'total_scenes': len(scenes), 'scene': scene,
                  'proposals': len(proposals), 'seconds': time.monotonic() - started}
        print(json.dumps(status), flush=True)
        write_json(args.output / 'status.json', status)
        del result
        torch.cuda.empty_cache()
    evaluate(args.data, args.output, scenes)
    write_json(args.output / 'DONE.json', {'scenes': len(scenes)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--evaluate-only', action='store_true')
    run(parser.parse_args())
