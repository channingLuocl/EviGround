"""Import the verified SeeGround Mask3D run without reading GT association files."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from eviground.categories.hypotheses import MASK3D_SCORE_TYPE, gate_categories, sha256
from eviground.categories.export import NATIVE_EXPORT_VERSION
from eviground.evaluation.run_storage import write_json


def import_scene(scene, data, predictions, alignment_root, output, record, raw_record):
    source = predictions / 'scenes' / f'{scene}.json'
    value = json.loads(source.read_text())
    arrays = source.with_suffix('.npz')
    if value['scene_id'] != scene or sha256(arrays) != value['arrays_sha256']:
        raise ValueError(f'Prediction scene/hash mismatch: {scene}')
    cloud = data / 'inputs' / f'{scene}.npz'
    if sha256(cloud) != record['input_sha256']:
        raise ValueError(f'Original prediction cloud changed: {scene}')
    alignment = alignment_root / scene / f'{scene}.txt'
    expected = next(r['sha256'] for r in raw_record['raw_files'] if r['file'] == alignment.name)
    if sha256(alignment) != expected:
        raise ValueError(f'Alignment metadata changed: {scene}')
    line = next(s for s in alignment.read_text().splitlines() if s.startswith('axisAlignment'))
    transform = np.asarray(list(map(float, line.split('=', 1)[1].split()))).reshape(4, 4)
    if not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError(f'Invalid coordinate transform: {scene}')
    with np.load(cloud, allow_pickle=False) as saved:
        xyz, rgb = saved['xyz'], saved['rgb']
    with np.load(arrays, allow_pickle=False) as saved:
        vertices = int(saved['vertices'])
        packed = saved['packed_masks']
        probabilities, indices = saved['class_probabilities'], saved['top5_class_indices']
        scores, queries = saved['scores'], saved['query_ids']
        masks = np.unpackbits(packed, axis=0, count=vertices).astype(bool)
    proposals = value['proposals']
    if (len(xyz) != vertices or rgb.shape != xyz.shape or not np.isfinite(xyz).all()
            or masks.shape != (vertices, len(proposals)) or probabilities.shape != (len(proposals), 200)
            or indices.shape != (len(proposals), 5) or len(scores) != len(proposals)
            or len(queries) != len(proposals)):
        raise ValueError(f'Prediction arrays are not aligned: {scene}')
    # Names are pinned to the same Mask3D head mapping as the category run.
    from baselines.seeground.upstream.data.scannet200_constants import CLASS_LABELS_200
    names = [CLASS_LABELS_200[1], *CLASS_LABELS_200[3:]]
    ranking = np.argsort(-probabilities[:, :198], axis=1, kind='stable')[:, :5]
    if not np.array_equal(ranking, indices):
        raise ValueError(f'Original category ranking changed: {scene}')
    aligned = (np.column_stack([xyz, np.ones(len(xyz))]) @ transform.T)[:, :3]
    objects, selected, rows = [], [], []
    for j, proposal in enumerate(proposals):
        top5 = [dict(category=names[int(i)], score=float(probabilities[j, i]), rank=k)
                for k, i in enumerate(indices[j], 1)]
        gate_categories(top5, score_type=MASK3D_SCORE_TYPE)
        if (proposal['proposal_id'] != j or proposal['query_id'] != int(queries[j])
                or proposal['top5'] != [r['category'] for r in top5]
                or proposal['top5_scores'] != [r['score'] for r in top5]
                or proposal['top1'] != top5[0]['category'] or proposal['score'] != float(scores[j])
                or not 0 <= proposal['score'] <= 1):
            raise ValueError(f'Proposal class/score/ID mismatch: {scene}/{j}')
        points = np.flatnonzero(masks[:, j])
        if len(points) < 3 or np.any(np.ptp(aligned[points], axis=0) <= 0):
            raise ValueError(f'Degenerate source mask; no silent filtering: {scene}/{j}')
        low, high = aligned[points].min(0), aligned[points].max(0)
        box = np.r_[(low + high) / 2, high - low].tolist()
        objects.append(dict(bbox_id=j, bbox_3d=box, target=proposal['top1'],
                            source_mask_index=j, point_count=len(points)))
        selected.append(points)
        rows.append(dict(scene=scene, bbox_id=j, bbox_3d=box, status='completed',
            predicted_object=proposal['top1'], top5=top5, score_type=MASK3D_SCORE_TYPE,
            proposal_detection_score=proposal['score']))
    if not objects:
        raise ValueError(f'No predicted instances: {scene}')
    background = np.flatnonzero(~masks.any(1))
    order = np.concatenate([background, *selected])
    ids = np.concatenate([np.full(len(background), -1, dtype=np.int32), *[
        np.full(len(points), j, dtype=np.int32) for j, points in enumerate(selected)]])
    directory = output / 'scenes' / scene
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / 'scene.npz', xyz=aligned[order], rgb=rgb[order], ids=ids)
    shutil.copyfile(arrays, directory / 'masks.npz')
    write_json(directory / 'scene_instances.json', objects)
    provenance = dict(scene=scene, source_prediction_sha256=sha256(source),
        input_sha256=record['input_sha256'], alignment_sha256=expected,
        arrays_sha256=value['arrays_sha256'], proposal_ids_preserved=True,
        overlap='shared vertices duplicated for every original mask; background=-1',
        gt_read=False, association_read=False, instances=len(objects))
    write_json(directory / 'provenance.json', provenance)
    return rows, dict(scene=scene, cloud_sha256=sha256(directory / 'scene.npz'),
        instances_sha256=sha256(directory / 'scene_instances.json'),
        masks_sha256=sha256(directory / 'masks.npz'), provenance_sha256=sha256(directory / 'provenance.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--alignment-root', type=Path, required=True)
    parser.add_argument('--raw-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a fresh output directory; incomplete imports cannot masquerade as complete')
    manifest = json.loads((args.data / 'manifest.json').read_text())
    protocol = json.loads((args.predictions / 'protocol.json').read_text())
    if (sha256(args.data / 'manifest.json') != protocol['manifest_sha256']
            or json.loads((args.predictions / 'DONE.json').read_text())['scenes'] != len(manifest['scenes'])
            or protocol['scenes'] != [r['scene_id'] for r in manifest['scenes']]):
        raise ValueError('Need the complete corresponding Mask3D prediction run')
    raw = {r['scene']: r for r in json.loads(args.raw_manifest.read_text())['scenes']}
    rows, sources = [], []
    for record in manifest['scenes']:
        scene = record['scene_id']
        items, source = import_scene(scene, args.data, args.predictions, args.alignment_root,
                                     args.output, record, raw[scene])
        rows.extend(items)
        sources.append(source)
        print(json.dumps(dict(scene=scene, completed=len(sources), total=len(manifest['scenes']),
                              proposals=len(items))), flush=True)
    export = args.output / 'categories'
    export.mkdir()
    config = dict(version=NATIVE_EXPORT_VERSION, category_source='seeground_mask3d_scannet200',
        score_type=MASK3D_SCORE_TYPE, sources=sources,
        frontend_protocol_sha256=sha256(args.predictions / 'protocol.json'))
    write_json(export / 'config.json', config)
    write_json(export / 'predictions.json', dict(version=NATIVE_EXPORT_VERSION,
        summary=dict(complete=True, errors=0, count=len(rows), scenes=len(sources),
                     score_type=MASK3D_SCORE_TYPE), results=rows))
    write_json(export / 'EXPORT_VERIFIED.json', dict(status='passed', files={
        name: sha256(export / name) for name in ('config.json', 'predictions.json')}))
    write_json(args.output / 'IMPORT_COMPLETE.json', dict(scenes=len(sources), instances=len(rows),
        gt_read=False, association_read=False, config_sha256=sha256(export / 'config.json')))


if __name__ == '__main__':
    main()
