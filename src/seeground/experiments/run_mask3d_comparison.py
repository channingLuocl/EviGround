"""Official ScanNet200 Mask3D frontend, with fixed-object geometric association."""
from seeground.paths import source_file
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from seeground.categories.category_common_v82 import file_hash, normalized_category, run_lock
from seeground.classifiers.mask3d_matching import associate, unique_predictions
from seeground.evaluation.run_storage import write_json


COLOR_MEAN = (.47793125906962, .4303257521323044, .3749598901421883)
COLOR_STD = (.2834475483823543, .27566157565723015, .27018971370874995)


def load_model(source, checkpoint):
    import hydra
    from omegaconf import OmegaConf
    sys.path.insert(0, str(source))
    config = OmegaConf.create({'general': {'num_targets': 201, 'train_on_segments': True},
        'data': {'in_channels': 3, 'num_labels': 200, 'voxel_size': .02},
        'model': OmegaConf.load(source/'conf/model/mask3d.yaml')})
    config.model.num_queries = 150
    model = hydra.utils.instantiate(config.model, _recursive_=False)
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict']
    model.load_state_dict({k.removeprefix('model.'): v for k, v in weights.items()}, strict=True)
    from datasets.scannet200.scannet200_constants import CLASS_LABELS_200
    return model.cuda().eval(), list(CLASS_LABELS_200)


def predict(model, xyz, rgb, segments):
    import MinkowskiEngine as ME
    from sklearn.cluster import DBSCAN
    from torch_scatter import scatter_mean
    xyz = np.asarray(xyz, dtype=np.float32)
    _, unique, inverse = ME.utils.sparse_quantize(coordinates=xyz/.02,
                                                 return_index=True, return_inverse=True)
    unique, inverse = np.asarray(unique), np.asarray(inverse)
    _, low_segments = np.unique(segments[unique], return_inverse=True)
    _, full_segments = np.unique(segments, return_inverse=True)
    features = ((rgb[unique].astype(np.float32)/255-np.asarray(COLOR_MEAN)) /
                np.asarray(COLOR_STD)).astype(np.float32)
    coordinates = ME.utils.batched_coordinates([np.floor(xyz[unique]/.02).astype(np.int32)])
    sparse = ME.SparseTensor(torch.from_numpy(features).cuda(), coordinates=coordinates.cuda())
    with torch.inference_mode():
        output = model(sparse, point2segment=[torch.from_numpy(low_segments).cuda()],
                       raw_coordinates=torch.from_numpy(xyz[unique]).cuda(), is_eval=True)
        probabilities = output['pred_logits'][0].softmax(-1)[..., :-1].cpu()
        low_logits = output['pred_masks'][0].cpu()[low_segments]
    del output, sparse
    # Official DBSCAN split, eps=.95, min_samples=1, prior to top-750 query/class selection.
    split_logits, split_probabilities = [], []
    for query in range(low_logits.shape[1]):
        positive = low_logits[:, query] > 0
        if not positive.any():
            continue
        labels = DBSCAN(eps=.95, min_samples=1, n_jobs=4).fit_predict(xyz[unique][positive])
        for cluster in np.unique(labels):
            keep = torch.zeros(len(unique), dtype=torch.bool)
            keep[positive] = torch.from_numpy(labels == cluster)
            split_logits.append(low_logits[:, query]*keep)
            split_probabilities.append(probabilities[query])
    if not split_logits:
        return np.empty((len(xyz), 0), dtype=bool), np.empty(0), np.empty(0, dtype=int)
    mask_logits = torch.stack(split_logits, 1)
    probabilities = torch.stack(split_probabilities)
    values, indices = probabilities.flatten().topk(min(750, probabilities.numel()), sorted=True)
    queries, classes = indices//200, indices % 200
    positive = mask_logits > 0
    quality = (mask_logits.sigmoid()*positive).sum(0)/(positive.sum(0)+1e-6)
    scores = values*quality[queries]
    # Avoid expanding the same query geometry for each of its class alternatives.
    selected = {}
    for i in scores.argsort(descending=True, stable=True).tolist():
        selected.setdefault(int(queries[i]), i)
    chosen = list(selected.values())
    masks = positive[:, queries[chosen]][inverse].float()
    masks = (scatter_mean(masks, torch.from_numpy(full_segments), dim=0) > .5)[full_segments].numpy()
    return unique_predictions(masks, scores[chosen].numpy(), classes[chosen].numpy())


def experiment(args):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    manifest = json.loads(args.manifest.read_text())
    scenes = manifest['scenes'][:args.limit] if args.limit else manifest['scenes']
    scenes = scenes[args.shard[0]::args.shard[1]]
    by_scene = defaultdict(list)
    for row in manifest['samples']:
        by_scene[row['scene']].append(row)
    with run_lock(args.output):
        protocol = {'version': 'mask3d-fixed-object-r1', 'seed': 42,
            'checkpoint_sha256': file_hash(args.checkpoint), 'manifest_sha256': file_hash(args.manifest),
            'source_commit': '11bd5ff94477ff7194e9a7c52e9fae54d73ac3b5',
            'model_source_sha256': {str(p.relative_to(args.source)): file_hash(p)
                for directory in ('models', 'conf/model') for p in sorted((args.source/directory).rglob('*'))
                if p.is_file() and p.suffix in ('.py', '.yaml')},
            'script_sha256': file_hash(Path(__file__)),
            'unlabeled_input_sha256': {str(p.relative_to(args.inputs)): file_hash(p)
                for scene in scenes for p in sorted((args.inputs/scene).iterdir()) if p.is_file()},
            'matching_sha256': file_hash(source_file('mask3d_matching.py')),
            'scenes': scenes, 'scene_shard': args.shard, 'color_mean': COLOR_MEAN, 'color_std': COLOR_STD,
            'checkpoint': 'official ScanNet200 validation weights, not benchmark train+val weights',
            'prediction': '150 queries, .02m voxels, segment masks, DBSCAN .95m min1, top750',
            'association': 'identical masks collapsed by confidence, class-blind one-to-one IoU>=.5; maximize match count then IoU',
            'denominator': 'all fixed objects; unmatched count wrong; no GT labels in model or association',
            'top5': 'not reported for proposal association',
            'evaluation_status': 'previously inspected validation; frontend comparison, not AgentGrounder reproduction'}
        saved = args.output/'protocol.json'
        if saved.exists() and json.loads(saved.read_text()) != json.loads(json.dumps(protocol)):
            raise ValueError('Protocol changed; use a new output directory')
        write_json(saved, protocol)
        write_json(args.output/'status.json', {'stage': 'loading_model', 'scenes': len(scenes)})
        model, labels = load_model(args.source, args.checkpoint)
        for number, scene in enumerate(scenes, 1):
            destination = args.output/f'scenes/{scene}.json'
            if destination.exists():
                continue
            started = time.monotonic()
            rows = by_scene[scene]
            scene_seed = int.from_bytes(scene.encode(), 'little') % (2**32-42)+42
            torch.manual_seed(scene_seed)
            np.random.seed(scene_seed)
            expected = next(r['cloud_sha256'] for r in manifest['sources'] if r['scene'] == scene)
            if file_hash(rows[0]['cloud']) != expected:
                raise ValueError('Evaluation cloud changed')
            directory = args.inputs/scene
            segments = np.asarray(json.loads((directory/(scene+'_vh_clean_2.0.010000.segs.json')).read_text())['segIndices'])
            from plyfile import PlyData
            vertices = PlyData.read(directory/(scene+'_vh_clean_2.ply'))['vertex']
            xyz = np.column_stack([vertices[k] for k in ('x', 'y', 'z')])
            rgb = np.column_stack([vertices[k] for k in ('red', 'green', 'blue')])
            if len(segments) != len(xyz):
                raise ValueError('Superpoint/vertex mismatch')
            masks, scores, classes = predict(model, xyz, rgb, segments)
            destination.parent.mkdir(parents=True, exist_ok=True)
            mask_path = destination.with_suffix('.masks.npz')
            np.savez_compressed(mask_path, packed_masks=np.packbits(masks, axis=0),
                                vertices=len(xyz), scores=scores, classes=classes)
            # GT instance IDs are read only after model inference.
            with np.load(rows[0]['cloud'], allow_pickle=False) as data:
                ids = data['ids']
            matches = associate(masks, ids, [r['bbox_id'] for r in rows])
            result = []
            for i, row in enumerate(rows):
                entry = {'scene': scene, 'bbox_id': row['bbox_id'], 'matched': i in matches,
                         'predicted_object': None, 'mask_iou': None, 'score': None}
                if i in matches:
                    j, iou = matches[i]
                    # Reproduce ScanNet200's chair special case and label_offset=2.
                    index = 1 if int(classes[j]) == 0 else int(classes[j])+2
                    name = labels[index] if index < len(labels) else f'unmapped_model_class_{classes[j]}'
                    entry.update(predicted_object=name, mask_iou=iou, score=float(scores[j]))
                result.append(entry)
            write_json(destination, {'results': result, 'unique_proposals': len(scores),
                                     'masks_sha256': file_hash(mask_path),
                                     'seconds': time.monotonic()-started})
            print(json.dumps({'scene': scene, 'progress': [number, len(scenes)],
                              'seconds': time.monotonic()-started}), flush=True)
            write_json(args.output/'status.json', {'stage': 'predicting', 'scene': scene,
                                                  'completed_scenes': number, 'scenes': len(scenes)})
            del masks
            torch.cuda.empty_cache()
        predictions = [r for scene in scenes for r in json.loads((args.output/f'scenes/{scene}.json').read_text())['results']]
        targets = [r for scene in scenes for r in by_scene[scene]]
        if not args.limit and args.shard == [0, 1] and (len(predictions) != 3413 or len(scenes) != 130):
            raise ValueError('Fixed 3413-object / 130-scene protocol changed')
        hits, exact, class_hits = [], [], defaultdict(list)
        for prediction, target in zip(predictions, targets):
            assert (prediction['scene'], prediction['bbox_id']) == (target['scene'], target['bbox_id'])
            name = prediction['predicted_object']
            correct = name is not None and normalized_category(name) == normalized_category(target['category'])
            hits.append(correct)
            exact.append(name == target['category'])
            class_hits[normalized_category(target['category'])].append(correct)
        matched = sum(r['matched'] for r in predictions)
        summary = {'count': len(hits), 'correct': sum(hits), 'accuracy': float(np.mean(hits)),
            'exact_accuracy': float(np.mean(exact)), 'top5_accuracy': None,
            'class_mean_accuracy': float(np.mean([np.mean(x) for x in class_hits.values()])),
            'errors': 0, 'matched': matched, 'coverage': matched/len(hits),
            'matched_accuracy': sum(hits)/matched if matched else None}
        write_json(args.output/'predictions.json', {'results': predictions})
        write_json(args.output/'results.json', {'protocol': protocol, 'summary': summary})
        write_json(args.output/'FROZEN.json', {p: file_hash(args.output/p) for p in
                   ('protocol.json', 'predictions.json', 'results.json')})
        write_json(args.output/'status.json', {'stage': 'completed', 'objects': len(predictions)})
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'source', 'checkpoint', 'inputs', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--shard', type=int, nargs=2, default=[0, 1], metavar=('INDEX', 'COUNT'))
    args = parser.parse_args()
    if not 0 <= args.shard[0] < args.shard[1]:
        parser.error('Expected 0 <= shard INDEX < COUNT')
    try:
        experiment(args)
    except BaseException as exc:
        write_json(args.output/'status.json', {'stage': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise
