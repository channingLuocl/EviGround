"""Auditable box-size, vertical-position and separate target/reference evidence."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from evidence import canonical_category, match_retrieval_categories

DESCRIPTORS=(r'\b(smallest|largest|biggest|tallest|shortest|lowest|highest|'
             r'larger|smaller|bigger|taller|shorter|lower|upper|tall|short|'
             r'small|large|skinny|narrow|thin|wide|thick)\b')
CONNECTORS=r'next to|beside|near|by|above|below|under|beneath|away from|nearest(?: to)?|closest to|farthest from|furthest(?: away)? from'
METRICS=('volume_m3','footprint_m2','height_m','center_z_m','bottom_z_m','top_z_m')

# Fixed during development, in meters. These checks reject unstable observed
# bounds; they do not certify scan completeness from the number of points.
SIZE_GEOMETRY_POLICY = {
    'minimum_extent_m': 0.02,
    'boundary_quantiles': [0.01, 0.99],
    'maximum_boundary_loss_m': 0.05,
    'maximum_boundary_loss_fraction': 0.10,
    'minimum_hull_rectangle_coverage': 0.50,
    'maximum_vertical_plane_normal_z': 0.15,
    'maximum_plane_thickness_fraction': 0.20,
    'plane_thickness_quantiles': [0.05, 0.95],
}


def _convex_hull(points):
    """Monotone-chain hull of a two-dimensional point set."""
    ordered = sorted(map(tuple, np.unique(np.asarray(points, dtype=float), axis=0)))
    if len(ordered) < 3:
        return np.asarray(ordered, dtype=float)
    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    lower, upper = [], []
    for sequence, result in ((ordered, lower), (reversed(ordered), upper)):
        for point in sequence:
            while len(result) >= 2 and cross(result[-2], result[-1], point) <= 0:
                result.pop()
            result.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _hull_area(hull):
    if len(hull) < 3:
        return 0.
    return float(abs(np.sum(hull[:, 0] * np.roll(hull[:, 1], -1) -
                            hull[:, 1] * np.roll(hull[:, 0], -1))) / 2.)


def _horizontal_rectangle(points):
    hull = _convex_hull(points[:, :2])
    if len(hull) < 3 or _hull_area(hull) <= 1e-12:
        return None
    best = None
    for edge in np.roll(hull, -1, axis=0) - hull:
        axis = edge / np.linalg.norm(edge)
        axes = np.array([axis, [-axis[1], axis[0]]]).T
        projected = hull @ axes
        extents = np.ptp(projected, axis=0)
        area = float(np.prod(extents))
        if best is None or area < best['area_m2'] - 1e-12:
            long_index = int(np.argmax(extents))
            best = {'area_m2': area, 'short_m': float(min(extents)),
                    'long_m': float(max(extents)),
                    'long_axis_xy': axes[:, long_index].tolist(),
                    'short_axis_xy': axes[:, 1 - long_index].tolist()}
    best['hull_rectangle_coverage'] = _hull_area(hull) / best['area_m2']
    return best


def _boundary_reason(values):
    span = float(np.ptp(values))
    if not np.isfinite(span) or span < SIZE_GEOMETRY_POLICY['minimum_extent_m']:
        return 'observed extent is nonfinite, missing or below minimum extent'
    robust = float(np.diff(np.quantile(values, SIZE_GEOMETRY_POLICY['boundary_quantiles']))[0])
    tolerance = max(SIZE_GEOMETRY_POLICY['maximum_boundary_loss_m'],
                    SIZE_GEOMETRY_POLICY['maximum_boundary_loss_fraction'] * span)
    if span - robust > tolerance:
        return 'isolated extrema make observed boundaries unstable'
    return None


def _oriented_measurements(obj, points):
    """Rotation-invariant observed dimensions with predicate-specific quality."""
    metrics, quality = {}, {}
    def record(name, value, algorithm, units, reasons):
        quality[name] = {'status': 'unknown' if reasons else 'usable',
                         'algorithm': algorithm, 'units': units, 'reasons': reasons}
        metrics[name] = None if reasons else float(value)
    height = float(np.ptp(points[:, 2]))
    height_reasons = [reason for reason in [_boundary_reason(points[:, 2])] if reason]
    record('height_m', height, 'observed instance world-Z span; gravity aligned', 'm', height_reasons)
    rectangle = _horizontal_rectangle(points)
    reasons = []
    if rectangle is None:
        reasons.append('horizontal instance support is missing or collinear')
    else:
        if rectangle['hull_rectangle_coverage'] <= SIZE_GEOMETRY_POLICY['minimum_hull_rectangle_coverage']:
            reasons.append('horizontal support does not cover the bounding rectangle')
    long_reasons, short_reasons = list(reasons), list(reasons)
    if rectangle is not None:
        for axis, axis_reasons in (('long_axis_xy', long_reasons),
                                   ('short_axis_xy', short_reasons)):
            reason = _boundary_reason(points[:, :2] @ rectangle[axis])
            if reason:
                axis_reasons.append(reason)
    footprint_reasons = list(dict.fromkeys(long_reasons + short_reasons))
    algorithm = 'gravity-aligned XY convex hull minimum-area enclosing rectangle'
    long = rectangle['long_m'] if rectangle else 0.
    short = rectangle['short_m'] if rectangle else 0.
    area = rectangle['area_m2'] if rectangle else 0.
    record('horizontal_oriented_long_extent_m', long, algorithm, 'm', long_reasons)
    record('horizontal_oriented_short_extent_m', short, algorithm, 'm', short_reasons)
    record('oriented_footprint_m2', area, algorithm, 'm2', footprint_reasons)
    record('oriented_volume_m3', area * height, algorithm + ' multiplied by observed Z span',
           'm3', list(dict.fromkeys(footprint_reasons + height_reasons)))
    face_reasons = []
    details = {'horizontal_rectangle': rectangle,
               'height_semantics': 'observed assigned-instance span; not a fitted desk tabletop',
               'scan_completeness': 'not independently certified; point count is not completeness evidence',
               'policy': dict(SIZE_GEOMETRY_POLICY)}
    width = 0.
    if canonical_category(obj['target']) not in ('door', 'window'):
        face_reasons.append('vertical-face size proxy requires a door or window structure')
    elif len(points) < 3:
        face_reasons.append('insufficient independent support for a vertical face')
    else:
        centered = points - points.mean(axis=0)
        values, axes = np.linalg.eigh(centered.T @ centered / len(points))
        normal = axes[:, 0]
        if values[1] <= 1e-12 or abs(float(normal[2])) > SIZE_GEOMETRY_POLICY['maximum_vertical_plane_normal_z']:
            face_reasons.append('instance points do not establish one vertical face')
        horizontal_normal = normal[:2]
        norm = np.linalg.norm(horizontal_normal)
        if norm > 1e-9:
            tangent = np.array([-horizontal_normal[1], horizontal_normal[0]]) / norm
            projected = np.column_stack((points[:, :2] @ tangent, points[:, 2]))
            width = float(np.ptp(projected[:, 0]))
            thickness = float(np.diff(np.quantile(centered @ normal,
                SIZE_GEOMETRY_POLICY['plane_thickness_quantiles']))[0])
            coverage = _hull_area(_convex_hull(projected)) / (width * height) if width * height > 1e-12 else 0.
            details['vertical_face'] = {'normal_xyz': normal.tolist(), 'width_axis_xy': tangent.tolist(),
                'observed_width_m': width, 'observed_height_m': height,
                'normal_90_percent_span_m': thickness, 'hull_rectangle_coverage': coverage}
            face_reasons.extend(reason for reason in [
                _boundary_reason(projected[:, 0]), _boundary_reason(projected[:, 1])] if reason)
            if min(width, height) <= 0. or thickness > SIZE_GEOMETRY_POLICY['maximum_plane_thickness_fraction'] * min(width, height):
                face_reasons.append('instance thickness does not support a thin face')
            if coverage <= SIZE_GEOMETRY_POLICY['minimum_hull_rectangle_coverage']:
                face_reasons.append('face support is visibly incomplete or degenerate')
        else:
            face_reasons.append('vertical face has no horizontal width direction')
    algorithm = 'instance PCA thin vertical plane; horizontal tangent width times world-Z span'
    record('vertical_face_width_m', width, algorithm, 'm', face_reasons)
    record('vertical_face_height_m', height, algorithm, 'm', face_reasons)
    record('vertical_face_area_m2', width * height, algorithm, 'm2', face_reasons)
    # Preserve the observed AABB height even when its quality is unknown.
    metrics.pop('height_m')
    return metrics, quality, details


def category_matches(name, objects):
    # ScanNet often calls kitchen cabinets simply cabinet. Keep this local to evidence.
    normalized=canonical_category(name)
    # Descriptors belong to the linguistic role, not to the ScanNet category.
    normalized=' '.join(word for word in normalized.split()
                        if word not in {'smallest','largest','biggest','tallest','shortest','lowest','highest',
                                        'larger','smaller','bigger','taller','shorter','lower','upper',
                                        'tall','short','small','large',
                                        'skinny','narrow','thin','wide','thick'})
    if normalized=='kitchen cabinet':normalized='cabinet'
    return match_retrieval_categories([normalized],objects)


def describe_role(phrase, objects):
    descriptors=[{'word':m.group().lower(),'phrase':m.group()} for m in re.finditer(DESCRIPTORS,phrase,re.I)]
    name=re.sub(DESCRIPTORS,'',phrase,flags=re.I).strip()
    name=re.sub(r'^(?:the|a|an)\s+','',name,flags=re.I).strip()
    matches=category_matches(name,objects)
    return {'phrase':phrase,'category':name,'descriptors':descriptors,
            'ids':[o['bbox_id'] for o in matches]}


def box_measurements(objects):
    rows=[]
    for obj in objects:
        x,y,z,w,length,h=map(float, obj['bbox_3d'])
        short, long = min(w, length), max(w, length)
        rows.append({'id':obj['bbox_id'],'volume_m3':w*length*h,'footprint_m2':w*length,
                     'height_m':h,'center_z_m':z,'bottom_z_m':z-h/2,'top_z_m':z+h/2,
                     'extent_x_m':w, 'extent_y_m':length, 'extent_z_m':h,
                     'horizontal_short_extent_m':short,
                     'horizontal_long_extent_m':long,
                     'horizontal_aspect_ratio':long / short if short > 1e-9 else None,
                     'height_to_horizontal_long_ratio':h / long if long > 1e-9 else None,
                     'source':'annotated_box', 'status':'known',
                     'measurement_reliability': 'usable' if min(w, length, h) > 1e-9 else 'unreliable'})
    rankings={}
    for metric in METRICS:
        ordered=sorted(rows,key=lambda r:(r[metric],r['id']))
        rankings[metric]={'ascending_ids':[r['id'] for r in ordered],
                          'minimum_ids':[r['id'] for r in ordered if abs(r[metric]-ordered[0][metric])<=1e-6],
                          'maximum_ids':[r['id'] for r in ordered if abs(r[metric]-ordered[-1][metric])<=1e-6]} if ordered else {}
    return {'measurements':rows,'rankings':rankings}


def point_cloud_measurements(objects, xyz, instance_ids, scene_objects=None):
    """Measure instance AABBs directly from the prepared scene point cloud.

    Missing instance points are reported as unknown instead of inheriting the
    table box, because a fabricated measurement would look like evidence.
    """
    rows = []
    if xyz is None or instance_ids is None:
        point_ids = None
    else:
        point_ids = np.asarray(instance_ids)
        points = np.asarray(xyz, dtype=float)
        if len(points) != len(point_ids):
            raise ValueError('scene xyz and instance ids must have equal length')
    for obj in objects:
        object_id = obj['bbox_id']
        object_points = None if point_ids is None else points[point_ids == object_id]
        row = {'id': object_id, 'point_count': 0,
               'source': 'instance_point_cloud_aabb', 'status': 'unknown',
               'measurement_reliability': 'unknown'}
        from v4_policy import enabled
        if enabled('size'):
            row['annotated_box_metrics'] = box_measurements([obj])['measurements'][0]
            row['provenance'] = {'instance_id': object_id, 'coordinate_frame': 'prepared scene world; Z gravity',
                                 'box_format': 'cx,cy,cz,dx,dy,dz',
                                 'box_source': 'supplied object table (GT/pred recorded at run level)',
                                 'independence': 'box and assigned-instance points are not independent evidence'}
        if object_points is None or not len(object_points):
            row['unknown_reason'] = 'no instance points in scene.npz'
            rows.append(row)
            continue
        if not np.isfinite(object_points).all():
            row['point_count'] = int(len(object_points))
            row['unknown_reason'] = 'nonfinite instance points prevent reliable measurement'
            rows.append(row)
            continue
        lower = np.min(object_points, axis=0).astype(float)
        upper = np.max(object_points, axis=0).astype(float)
        extent = upper - lower
        horizontal_short = min(float(extent[0]), float(extent[1]))
        horizontal_long = max(float(extent[0]), float(extent[1]))
        # A point-cloud AABB is a size proxy. Sparse or degenerate instances
        # remain visible in the audit, but are not safe for ranking.
        reliability = 'usable'
        reliability_reason = None
        if len(object_points) < 2:
            reliability = 'limited'
            reliability_reason = 'fewer than two instance points'
        elif min(float(value) for value in extent) <= 1e-9:
            reliability = 'unreliable'
            reliability_reason = 'degenerate point-cloud extent'
        row.update({
            'point_count': int(len(object_points)),
            'min_xyz': lower.tolist(), 'max_xyz': upper.tolist(),
            'extent_x_m': float(extent[0]), 'extent_y_m': float(extent[1]),
            'extent_z_m': float(extent[2]),
            'volume_m3': float(np.prod(extent)),
            'footprint_m2': float(extent[0] * extent[1]),
            'height_m': float(extent[2]),
            'horizontal_short_extent_m': horizontal_short,
            'horizontal_long_extent_m': horizontal_long,
            'horizontal_aspect_ratio': (horizontal_long / horizontal_short
                                        if horizontal_short > 1e-9 else None),
            'height_to_horizontal_long_ratio': (float(extent[2]) / horizontal_long
                                                 if horizontal_long > 1e-9 else None),
            'status': 'known',
            'measurement_reliability': reliability,
        })
        if reliability_reason:
            row['reliability_reason'] = reliability_reason
        oriented, metric_quality, geometry_details = _oriented_measurements(obj, object_points)
        row.update(oriented)
        row['metric_reliability'] = metric_quality
        row['geometry_details'] = geometry_details
        if enabled('size'):
            row['boundary_stability'] = {axis: {
                'raw_span_m': float(np.ptp(object_points[:, index])),
                'robust_span_m': float(np.diff(np.quantile(object_points[:, index], [.01, .99]))[0]),
                'raw_bounds_m': [float(v) for v in (lower[index], upper[index])],
                'robust_bounds_m': np.quantile(object_points[:, index], [.01, .99]).tolist(),
                'status': 'unstable' if _boundary_reason(object_points[:, index]) else 'stable'}
                for index, axis in enumerate(('x', 'y', 'z'))}
            row['observed_completeness'] = observed_completeness(object_points)
            robust_bottom, robust_top = np.quantile(object_points[:, 2], [.05, .95])
            row['body_height_diagnostics'] = {
                'instance_total_span_m': float(extent[2]),
                'central_90_percent_span_m': float(robust_top - robust_bottom),
                'upper_tail_points': int(np.sum(object_points[:, 2] > robust_top)),
                'lower_tail_points': int(np.sum(object_points[:, 2] < robust_bottom)),
                'body_height_m': None,
                'reason': 'quantile span is a diagnostic, not body segmentation; attached contents and missing body remain unresolved'}
            for metric in ('volume_m3', 'footprint_m2'):
                axes = ('x', 'y', 'z') if metric == 'volume_m3' else ('x', 'y')
                reasons = [axis + ' boundary is unstable' for axis in axes
                           if row['boundary_stability'][axis]['status'] != 'stable']
                row['metric_reliability'][metric] = {'status': 'unknown' if reasons else 'usable',
                    'reasons': reasons, 'units': 'm3' if metric == 'volume_m3' else 'm2',
                    'algorithm': 'observed assigned-instance AABB'}
            if canonical_category(obj['target']) in ('table', 'desk'):
                row.update(tabletop_measurement(obj, object_points, scene_objects or objects, points, point_ids))
                row['metric_reliability']['tabletop_height_m'] = {
                    'status': row['tabletop_evidence']['status'], 'units': 'm',
                    'algorithm': row['tabletop_evidence']['source'],
                    'reasons': [] if row['tabletop_height_m'] is not None else [row['tabletop_evidence']['reason']]}
        rows.append(row)
    return rows


def observed_completeness(points):
    """A surface-only observation is a risk flag, not a missing-object verdict."""
    centered = points - points.mean(0)
    values, axes = np.linalg.eigh(centered.T @ centered / len(points))
    normal = axes[:, 0]
    thickness = float(np.diff(np.quantile(centered @ normal, [.05, .95]))[0])
    planar = values[1] > 1e-10 and values[0] / values[1] < .015
    horizontal = abs(float(normal[2])) > .95
    surface_only = bool(planar and horizontal)
    return {'status': 'unknown', 'surface_only_risk': surface_only,
            'normal_xyz': normal.tolist(), 'surface_thickness_90_percent_m': thickness,
            'observations': ['points predominantly form one horizontal surface; sides/bottom not established']
                            if surface_only else ['multiple observed dimensions; complete physical body still unverified'],
            'alternative_explanations': ['partial scan of a top surface', 'a genuinely thin object'] if surface_only else [],
            'affected_metrics': ['volume_m3', 'oriented_volume_m3', 'height_m'] if surface_only else [],
            'requires_visual_check': surface_only,
            'policy': 'boundary stability, thickness and point count cannot certify completeness'}


def tabletop_measurement(obj, object_points, objects, points, point_ids):
    """Fit a horizontal upper surface and a nearby *labelled* floor independently."""
    floor_ids = [o['bbox_id'] for o in objects if canonical_category(o['target']) in ('floor', 'ground')]
    box = np.asarray(obj['bbox_3d'], float)
    local = np.all(np.abs(points[:, :2] - box[:2]) <= box[3:5] / 2 + .30, axis=1)
    floor = points[local & np.isin(point_ids, floor_ids)]
    result = {'tabletop_height_m': None, 'tabletop_evidence': {'status': 'unknown',
              'source': 'assigned horizontal surface and local labelled floor', 'floor_instance_ids': floor_ids}}
    evidence = result['tabletop_evidence']
    if len(floor) < 30 or len(object_points) < 30:
        evidence['reason'] = 'local labelled floor or tabletop plane support unavailable'
        return result
    floor_z = float(np.median(floor[:, 2]))
    if np.diff(np.quantile(floor[:, 2], [.05, .95]))[0] > .04:
        evidence['reason'] = 'local floor is not one stable horizontal plane'
        return result
    upper = object_points[object_points[:, 2] >= np.median(object_points[:, 2])]
    bins = np.floor(upper[:, 2] / .02).astype(int)
    keys, counts = np.unique(bins, return_counts=True)
    center_z = (keys[np.argmax(counts)] + .5) * .02
    plane = upper[np.abs(upper[:, 2] - center_z) <= .02]
    rectangle = _horizontal_rectangle(plane) if len(plane) >= 30 else None
    overall = _horizontal_rectangle(object_points)
    if (not rectangle or not overall or rectangle['area_m2'] < .50 * overall['area_m2']
            or rectangle['hull_rectangle_coverage'] < .50):
        evidence['reason'] = 'upper horizontal patch lacks broad tabletop support'
        return result
    z = float(np.median(plane[:, 2]))
    centered = plane - plane.mean(0)
    _, axes = np.linalg.eigh(centered.T @ centered / len(plane))
    if abs(float(axes[2, 0])) < .98 or z - floor_z <= .10:
        evidence['reason'] = 'tabletop orientation or above-floor separation not established'
        return result
    result['tabletop_height_m'] = z - floor_z
    evidence.update(status='usable', tabletop_z_m=z, local_floor_z_m=floor_z,
                    tabletop_enclosing_area_m2=rectangle['area_m2'],
                    reason='broad horizontal assigned-instance patch above a stable local labelled floor')
    return result


def build_geometry(query, parsed, objects):
    evidence={'policy':'soft box evidence only; no candidate exclusion or unique reference certification',
              'units':'meters, square meters, cubic meters; world Z points up',
              'limitations':['AABB volume/footprint are alternative size proxies, not object volume.',
                             'Tall/lower/near/away from have no calibrated threshold; not scene-wide extremes.',
                             'Category pools describe the full supplied table; linguistic subsets remain unbound.',
                             'Box distances and vertical overlap do not prove support, contact or appearance.'],
              'category_pools':[],'roles':{},'binding_pairs':[],'missing':[]}
    names=[]
    for field in ('Target','Anchor'):
        value=parsed.get(field,[])
        names.extend([value] if isinstance(value,str) else value)
    for name in dict.fromkeys(names):
        pool=category_matches(name,objects)
        evidence['category_pools'].append({'category':name,'ids':[o['bbox_id'] for o in pool],
                                          **box_measurements(pool)})
    # Whole simple clauses only: preserve distinct noun phrases even for identical categories.
    clause=re.sub(r'^(?:find|choose|select)\s+','',query.strip(),flags=re.I).rstrip('.!?')
    match=re.fullmatch(r'(?P<target>[A-Za-z ]+?)\s+(?P<relation>'+CONNECTORS+r')\s+(?P<reference>[A-Za-z ]+)',clause,re.I)
    phrases={'target':match['target'].strip(),'reference':match['reference'].strip()} if match else {'target':clause}
    for role,phrase in phrases.items():
        info=describe_role(phrase,objects)
        evidence['roles'][role]={**info,**box_measurements([o for o in objects if o['bbox_id'] in info['ids']])}
        if not info['ids']:evidence['missing'].append('unbound '+role+' phrase: '+phrase)
    if not match:
        if parsed.get('Anchor'):evidence['missing'].append('reference phrase requires a simple explicit clause')
        return evidence
    evidence['relation_quote']=match['relation']
    by_id={o['bbox_id']:o for o in objects}
    for target_id in evidence['roles']['target']['ids']:
        for reference_id in evidence['roles']['reference']['ids']:
            if target_id==reference_id:continue
            a=np.asarray(by_id[target_id]['bbox_3d']);b=np.asarray(by_id[reference_id]['bbox_3d'])
            gap=np.maximum(np.abs(a[:3]-b[:3])-(a[3:]+b[3:])/2,0)
            overlap=np.maximum(np.minimum(a[:3]+a[3:]/2,b[:3]+b[3:]/2)
                               -np.maximum(a[:3]-a[3:]/2,b[:3]-b[3:]/2),0)
            evidence['binding_pairs'].append({'target_id':target_id,'reference_id':reference_id,
                'center_distance_m':float(np.linalg.norm(a[:3]-b[:3])),
                'aabb_gap_m':float(np.linalg.norm(gap)),'horizontal_aabb_gap_m':float(np.linalg.norm(gap[:2])),
                'center_z_delta_m':float(a[2]-b[2]),'vertical_overlap_m':float(overlap[2]),
                'horizontal_overlap_m2':float(overlap[0]*overlap[1]),
                'target_bottom_minus_reference_top_m':float(a[2]-a[5]/2-b[2]-b[5]/2),
                'reference_bottom_minus_target_top_m':float(b[2]-b[5]/2-a[2]-a[5]/2)})
    if not evidence['binding_pairs']:evidence['missing'].append('no distinct target/reference pair')
    return evidence


def main():
    from run import load_objects
    from real_scene import object_table_path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir',type=Path,required=True);parser.add_argument('--query',required=True)
    parser.add_argument('--target',required=True);parser.add_argument('--anchor',default='')
    parser.add_argument('--object-table',choices=('pred','gt'),default='pred')
    args=parser.parse_args()
    objects=load_objects(object_table_path(args.scene_dir,args.object_table))
    evidence=build_geometry(args.query,{'Target':[args.target],'Anchor':[args.anchor] if args.anchor else []},objects)
    print(json.dumps({'network_request':False,'object_table':args.object_table,'geometry_conditions':evidence},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
