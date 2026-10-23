"""Finite vertical-wall proxies fitted from labeled, aligned scene points."""
import numpy as np
from seeground.grounding.evidence import retrieval_category


def fit_wall(points):
    xy = np.asarray(points, dtype=float)[:, :2]
    record = {'source': 'aligned instance points; XY PCA vertical-wall proxy',
              'point_count': len(xy), 'max_normal_rms_m': 0.10,
              'max_variance_ratio': 0.05, 'valid': False}
    if len(xy) < 20:
        return {**record, 'reason': 'too few wall points (minimum 20)'}
    origin = xy.mean(0)
    delta = xy - origin
    values, vectors = np.linalg.eigh(delta.T @ delta / len(xy))
    if values[1] <= 1e-6:
        return {**record, 'reason': 'degenerate wall horizontal span'}
    tangent = vectors[:, 1]
    normal = np.array([-tangent[1], tangent[0]])
    offsets = delta @ tangent
    rms = float(np.sqrt(max(values[0], 0.)))
    ratio = float(max(values[0], 0.) / values[1])
    record.update(origin_xy_m=origin.tolist(), tangent_xy=tangent.tolist(),
                  normal_xy=normal.tolist(), tangent_extent_m=[float(offsets.min()), float(offsets.max())],
                  normal_rms_m=rms, variance_ratio=ratio)
    if rms > record['max_normal_rms_m'] or ratio > record['max_variance_ratio']:
        return {**record, 'reason': 'wall points do not support one thin vertical plane'}
    return {**record, 'valid': True}


def attach_wall_geometry(objects, xyz, ids):
    return [{**obj, 'wall_geometry': fit_wall(xyz[ids == obj['bbox_id']])}
            if retrieval_category(obj['target']) == 'wall' else obj for obj in objects]


def wall_reference_point(wall, other):
    """Closest point on the finite fitted XY wall segment to the other anchor."""
    origin = np.asarray(wall['origin_xy_m'])
    tangent = np.asarray(wall['tangent_xy'])
    coordinate = float((np.asarray(other['bbox_3d'][:2]) - origin) @ tangent)
    bounded = float(np.clip(coordinate, *wall['tangent_extent_m']))
    return origin + bounded * tangent, coordinate != bounded


def main():
    import argparse
    import json
    from pathlib import Path
    from seeground.grounding.diagnostic import diagnose, validate_structure
    from seeground.query.query_rewrite import rewrite_query
    from seeground.grounding.real_scene import object_table_path, structural_objects
    from seeground.rendering.run import load_objects

    parser = argparse.ArgumentParser(description='Offline wall-between evidence from a saved run; no API calls')
    parser.add_argument('--scene-dir', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = json.loads(args.run.read_text())
    table = run['object_table']
    objects = load_objects(object_table_path(args.scene_dir, table))
    with np.load(args.scene_dir/'scene.npz', allow_pickle=False) as cloud:
        xyz, ids = cloud['xyz'], cloud['ids']
    geometry_objects = attach_wall_geometry(objects + structural_objects(args.scene_dir, objects), xyz, ids)
    parsed = validate_structure(run['query'], run['parsed_query'])
    parsed['query_rewrite'] = rewrite_query(run['query'], parsed)
    candidate_ids = set(run['review_proposal']['ids'])
    diagnosis = diagnose(parsed, geometry_objects,
                         [obj for obj in objects if obj['bbox_id'] in candidate_ids],
                         (xyz.min(0)+xyz.max(0))/2,
                         run.get('diagnosis', {}).get('checks', []), trusted_categories=table == 'gt')
    groups = {}
    for triplet in diagnosis['geometry']['between_triplets']:
        key = ','.join(map(str, triplet['anchor_ids']))
        group = groups.setdefault(key, {'wall_pass_ids': [], 'center_pass_ids': [], 'unknown_ids': []})
        if triplet['passes'] is True:
            group['wall_pass_ids'].append(triplet['target_id'])
        elif triplet['passes'] is None:
            group['unknown_ids'].append(triplet['target_id'])
        if triplet.get('center_point_comparison', {}).get('passes'):
            group['center_pass_ids'].append(triplet['target_id'])
    print(json.dumps({'network_request': False, 'query': run['query'],
                      'status': diagnosis['status'], 'missing': diagnosis['missing'],
                      'candidate_ids': diagnosis['candidate_ids'],
                      'stable_candidate_ids': diagnosis['stable_candidate_ids'],
                      'between_pass_ids': diagnosis['geometry']['between_pass_ids'],
                      'between_unknown_ids': diagnosis['geometry']['between_unknown_ids'],
                      'candidate_shape_metrics': diagnosis['geometry']['candidate_metrics'],
                      'wall_fits': {str(obj['bbox_id']): obj['wall_geometry'] for obj in geometry_objects
                                    if 'wall_geometry' in obj},
                      'reference_groups': groups}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
