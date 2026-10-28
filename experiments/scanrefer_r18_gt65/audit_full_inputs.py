"""Read-only audit of complete inputs; does not run inference or select answers."""
import json
from pathlib import Path

import numpy as np

from scanrefer_metrics import prepare
from types import SimpleNamespace
from verify import ROOT, digest


def main():
    dataset = Path('/Volumes/KESU/datasets/ScanRefer')
    manifest = ROOT / 'inputs/full_manifest.json'
    # Recompute official uniqueness and exact descriptions, refusing any difference.
    recomputed = ROOT / 'work/audit_official_manifest.json'
    prepare(SimpleNamespace(annotations=dataset / 'ScanRefer_filtered_val.json',
                            labels=dataset / 'scannetv2-labels.combined.tsv', output=recomputed))
    assert json.loads(recomputed.read_text()) == json.loads(manifest.read_text())
    rows = json.loads(manifest.read_text())['rows']
    assert len(rows) == 9508 and len({r['scene'] for r in rows}) == 141
    export = ROOT / 'inputs/r8_predictions'
    marker = json.loads((export / 'EXPORT_VERIFIED.json').read_text())
    assert marker['status'] == 'passed'
    for name, checksum in marker['files'].items():
        assert digest(export / name) == checksum, name
    predictions = json.loads((export / 'predictions.json').read_text())['results']
    by_key = {(r['scene'], r['bbox_id']): r for r in predictions}
    assert len(by_key) == len(predictions)
    original = json.loads((ROOT / 'inputs/r8_predictions_original/predictions.json').read_text())['results']
    assert all(by_key[(r['scene'], r['bbox_id'])] == r for r in original)
    geometry_count = 0
    for scene in sorted({r['scene'] for r in rows}):
        directory = ROOT / 'inputs/scenes' / scene
        with np.load(directory / 'scene.npz', allow_pickle=False) as cloud:
            assert set(cloud.files) == {'xyz', 'rgb', 'ids'}, scene
            ids = set(map(int, np.unique(cloud['ids']))) - {-1}
            assert np.isfinite(cloud['xyz']).all(), scene
        table = json.loads((directory / 'scene_instances.json').read_text())
        boxes = json.loads((directory / 'objects_gt.json').read_text())
        assert {o['bbox_id'] for o in table} == {o['bbox_id'] for o in boxes} == ids, scene
        assert all(o['target'] == 'object' for o in table), scene
        evaluation_boxes = {o['bbox_id']: o['bbox_3d'] for o in boxes}
        assert all(np.allclose(o['bbox_3d'], evaluation_boxes[o['bbox_id']], atol=1e-5, rtol=0)
                   for o in table), scene
        assert all(r['target_id'] in ids and (scene, r['target_id']) in by_key
                   for r in rows if r['scene'] == scene), scene
        geometry_count += len(ids)
    assert all(r['status'] == 'completed' and len(r['top5']) == 5 for r in predictions)
    record = {'queries': len(rows), 'scenes': 141,
              'unique': sum(r['unique'] for r in rows),
              'multiple': sum(not r['unique'] for r in rows),
              'all_original_descriptions_and_unique_labels_verified': True,
              'original_r8_rows_unchanged': len(original),
              'total_predicted_instances': len(predictions),
              'geometry_instances': geometry_count,
              'all_targets_have_geometry_and_predicted_top5': True,
              'cloud_fields': ['xyz', 'rgb', 'ids'],
              'geometry_category_strings': ['object'],
              'objects_gt_json': 'evaluation only; original cached GT names retained, never loaded as inference OLT',
              'manifest_sha256': digest(manifest),
              'export_marker_sha256': digest(export / 'EXPORT_VERIFIED.json'),
              'audit_script_sha256': digest(Path(__file__))}
    path = ROOT / 'provenance/FULL_INPUT_AUDIT.json'
    path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
