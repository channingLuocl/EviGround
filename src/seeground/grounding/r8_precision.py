"""Named dimensions and conditional observer frames, using prediction geometry only."""
import numpy as np

from seeground.grounding.evidence import distance_pair_metrics, match_retrieval_categories
from seeground.grounding.r8_grounding import generic_target_pool, matching_hypotheses


INSTRUCTIONS = '''
r3 measurement contract:
objects use explicit height_m, extent_x_m, extent_y_m and center coordinates. Compare height_m for
tall/short, never a horizontal extent. numeric_reference_orders lists EACH reference category's
measurements; bind the described reference and its appearance before comparing the targets.
relative_geometry contains signed target-minus-reference coordinates, vertical surface gap and
XY overlap. A target supported ON a reference usually has overlapping XY and bottom near its top;
zero AABB gap alone is NOT proof of support. A box attached to the side need not be on its top.
observer_frames are CONDITIONAL coordinate transforms, not inferred facts about the speaker.
For each frame, positive right_m is speaker right and positive forward_m is along their gaze.
Use only a frame whose origin AND facing agree with the original description. When no observer
is specified, a room-center-facing-reference frame is an explicit assumption, not world-X left.
Never select a frame because its ordering happens to support a preferred object. For an explicit
observer at a door facing the targets, use that door origin and target-group facing frame.
If appearance supports only one target, do not overturn it using an unsupported coordinate frame.
Explain any essential frame assumption in unresolved. Preserve all required clauses and negation.
Classifier hypotheses are rank priors, not probabilities; score magnitudes are omitted from this
compact view. Primary-match candidates get preference only when other evidence is comparable.
'''


def frame_coordinates(origin, facing, candidates):
    origin, facing = np.asarray(origin, float)[:2], np.asarray(facing, float)[:2]
    forward = facing-origin
    if np.linalg.norm(forward) < 1e-6:
        return None
    forward /= np.linalg.norm(forward)
    right = np.array([forward[1], -forward[0]])
    rows = []
    for item in candidates:
        delta = np.asarray(item['bbox_3d'][:2])-origin
        rows.append([item['bbox_id'], round(float(delta@right), 4),
                     round(float(delta@forward), 4)])
    return {'origin_xy_m': origin.tolist(), 'forward_xy': forward.tolist(),
            'right_xy': right.tolist(), 'columns': ['target_id', 'right_m', 'forward_m'],
            'rows_left_to_right': sorted(rows, key=lambda r: (r[1], r[0]))}


def precision_evidence(query, parsed, objects, candidates):
    targets, anchors = parsed.get('Target', []), parsed.get('Anchor', [])
    table = []
    for item in objects:
        x, y, z, dx, dy, height = map(float, item['bbox_3d'])
        hypotheses = [h for h in item.get('category_hypotheses', []) if generic_target_pool(targets) or h['rank'] == 1
                      or matching_hypotheses(targets+anchors,
                          {**item, 'category_hypotheses': [h]})]
        table.append({'id': item['bbox_id'], 'category': item['target'],
            'category_hypotheses': [{'category': h['category'], 'rank': h['rank']} for h in hypotheses],
            **{k: round(v, 4) for k, v in zip(
                ('center_x_m', 'center_y_m', 'center_z_m', 'extent_x_m', 'extent_y_m',
                 'height_m', 'bottom_z_m', 'top_z_m'), (x, y, z, dx, dy, height, z-height/2, z+height/2))}})
    orders = []
    for category in anchors:
        pool = match_retrieval_categories([category], objects)
        orders.append({'category': category, 'height_ascending': [
            [o['bbox_id'], round(float(o['bbox_3d'][5]), 4),
             min((h['rank'] for h in matching_hypotheses([category], o)), default=None)]
            for o in sorted(pool, key=lambda o: (o['bbox_3d'][5], o['bbox_id']))],
            'columns': ['reference_id', 'height_m', 'best_category_rank']})
    relative = []
    for reference in match_retrieval_categories(anchors, objects):
        box = np.asarray(reference['bbox_3d'], float)
        rows = []
        for item in candidates:
            if item['bbox_id'] == reference['bbox_id']:
                continue
            target = np.asarray(item['bbox_3d'], float)
            overlap = np.maximum(0, np.minimum(target[:2]+target[3:5]/2, box[:2]+box[3:5]/2)
                                 - np.maximum(target[:2]-target[3:5]/2, box[:2]-box[3:5]/2))
            metrics = distance_pair_metrics(reference, item)
            values = [*list(target[:3]-box[:3]), metrics['horizontal_center'],
                metrics['horizontal_gap'], target[2]-target[5]/2-(box[2]+box[5]/2),
                float(np.prod(overlap)/np.prod(target[3:5]))]
            rows.append([item['bbox_id'], *[round(float(v), 4) for v in values]])
        relative.append({'reference_id': reference['bbox_id'],
            'columns': ['target_id', 'delta_x_m', 'delta_y_m', 'delta_z_m', 'xy_center_m',
                        'xy_box_gap_m', 'target_bottom_minus_reference_top_m', 'target_xy_overlap_fraction'],
            'rows': rows})
    boxes = np.array([o['bbox_3d'] for o in objects], float)
    room_center = ((boxes[:, :3]-boxes[:, 3:]/2).min(0)
                   +(boxes[:, :3]+boxes[:, 3:]/2).max(0))/2
    frames = []
    for reference in match_retrieval_categories(anchors, objects):
        coords = frame_coordinates(room_center, reference['bbox_3d'][:3], candidates)
        if coords:
            frames.append({'assumption': 'room_center_facing_reference',
                           'reference_id': reference['bbox_id'], **coords})
    observer = parsed.get('Observer', {})
    origin = observer.get('origin', {})
    facing = observer.get('facing', {})
    if (origin.get('category') and origin.get('phrase')
            and origin['phrase'].casefold() in query.casefold()):
        origins = match_retrieval_categories([origin['category']], objects)
        explicit_facing = (facing.get('category') and facing.get('phrase')
                           and facing['phrase'].casefold() in query.casefold())
        facing_pool = match_retrieval_categories([facing['category']], objects) if explicit_facing else candidates
        if facing_pool:
            at = np.mean([o['bbox_3d'][:3] for o in facing_pool], axis=0)
            for item in origins:
                coords = frame_coordinates(item['bbox_3d'][:3], at, candidates)
                if coords:
                    frames.append({'assumption': 'parsed_observer_facing_category_group',
                        'origin_id': item['bbox_id'], 'origin_quote': origin['phrase'],
                        'facing_category': facing['category'] if explicit_facing else targets,
                        'facing_quote': facing['phrase'] if explicit_facing else '', **coords})
    for frame in frames:
        frame['frame_id'] = (f"room_center_facing_reference_{frame['reference_id']}" if 'reference_id' in frame
                             else f"observer_{frame['origin_id']}_facing_category_group")
    return {'objects': table, 'numeric_reference_orders': orders,
            'relative_geometry': relative, 'observer_frames': frames}
