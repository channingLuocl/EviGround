"""Query matching and measured geometry; no ground-truth labels are used."""
import re
from categories_v9 import category_names
import numpy as np

ALIASES = {
    'television': 'tv', 'televisions': 'tv', 'screen': 'tv', 'screens': 'tv',
    'monitor': 'tv', 'monitors': 'tv',
    'trashcan': 'trash can', 'trashcans': 'trash can', 'trash cans': 'trash can',
    'garbage bin': 'trash can', 'garbage can': 'trash can',
    'fridge': 'refrigerator', 'refrigerators': 'refrigerator',
    'front doors': 'door', 'front door': 'door',
    'double doors': 'door', 'double door': 'door', 'doors': 'door',
    'windows': 'window', 'chairs': 'chair', 'tables': 'table', 'cabinets': 'cabinet',
    'curtains': 'curtain',
    'carts': 'cart', 'photocopiers': 'copier', 'photocopier': 'copier', 'copiers': 'copier',
    'blackboards': 'blackboard',
    'countertop': 'counter', 'countertops': 'counter',
}

# Distance is evidence only when the query supplies a comparison operator.
# Keep the metric names explicit so a center distance is never mistaken for a
# surface, horizontal, or observer distance.
DISTANCE_METRIC_ALIASES = {
    'center': 'center',
    'center_distance': 'center',
    'euclidean': 'center',
    'euclidean_center': 'center',
    'aabb': 'aabb_gap',
    'aabb_gap': 'aabb_gap',
    'box_gap': 'aabb_gap',
    'surface': 'surface_gap',
    'surface_gap': 'surface_gap',
    'surface_distance': 'surface_gap',
    'horizontal': 'horizontal_center',
    'horizontal_center': 'horizontal_center',
    'horizontal_distance': 'horizontal_center',
    'horizontal_gap': 'horizontal_gap',
    'viewpoint': 'viewpoint',
    'observer': 'viewpoint',
    'observer_distance': 'viewpoint',
    'plane': 'plane',
    'plane_distance': 'plane',
}
DISTANCE_TIE_TOLERANCE_M = 0.01

# A classifier miss is most recoverable when the description already gives a
# spatial or visual relation.  Keep this gate lexical and conservative: the
# fallback is a small geometry neighborhood, never the whole scene.
_VISUAL_FALLBACK_TERMS = re.compile(
    r'\b(?:both|group|one|another|left|right|front|behind|back|above|below|under|'
    r'near|next\s+to|beside|alongside|closest|nearest|farthest|furthest|'
    r'between|against|inside|outside|middle|corner|end|row|stack|largest|'
    r'smallest|tallest|lowest|highest|white|black|red|blue|green|brown|'
    r'wooden|plain|open|closed|flat|pane|floor-length)\b', re.IGNORECASE)
_SHELL_CATEGORIES = {'wall', 'floor', 'ground', 'ceiling'}


def normalize_distance_metric(value):
    """Return a stable metric name, or ``None`` for an unsupported metric."""
    if value is None:
        return 'center'
    if not isinstance(value, str):
        return None
    return DISTANCE_METRIC_ALIASES.get(value.strip().casefold())


def _distance_metric_for_relation(relation):
    raw = relation.get('distance_metric', relation.get('metric'))
    if raw is None:
        scope = relation.get('distance_scope', relation.get('scope', ''))
        if isinstance(scope, str) and scope.casefold() in ('horizontal', 'horizontal_distance'):
            raw = 'horizontal_center'
        elif isinstance(scope, str) and scope.casefold() in ('viewpoint', 'observer'):
            raw = 'viewpoint'
    return normalize_distance_metric(raw)


def _viewpoint_origin(viewpoint):
    """Extract a numeric observer origin without inventing one from a category."""
    if viewpoint is None:
        return None
    value = viewpoint
    if isinstance(value, dict):
        if 'origin' in value and not any(key in value for key in ('xyz', 'point', 'position')):
            value = value.get('origin')
        if isinstance(value, dict):
            for key in ('xyz', 'point', 'position', 'origin_xyz'):
                if key in value:
                    value = value[key]
                    break
    try:
        point = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if point.size < 3 or not np.isfinite(point[:3]).all():
        return None
    return point[:3]


def distance_pair_metrics(target, anchor, viewpoint=None):
    """Compute separately named pair distances for one bound target/anchor.

    ``surface_gap`` is the AABB gap proxy.  A finite fitted wall can provide a
    ``plane`` distance; otherwise that metric remains unknown instead of using
    the wall center as a surrogate.  A viewpoint distance is only present when
    a numeric observer origin is supplied.
    """
    a = np.asarray(target['bbox_3d'], dtype=float)
    b = np.asarray(anchor['bbox_3d'], dtype=float)
    a_min, a_max = a[:3] - a[3:] / 2., a[:3] + a[3:] / 2.
    b_min, b_max = b[:3] - b[3:] / 2., b[:3] + b[3:] / 2.
    gap_xyz = np.maximum(np.abs(a[:3] - b[:3]) - (a[3:] + b[3:]) / 2., 0.)
    gap_xy = np.maximum(np.abs(a[:2] - b[:2]) - (a[3:5] + b[3:5]) / 2., 0.)
    target_center = a[:3]
    anchor_center = b[:3]
    metrics = {
        'center': float(np.linalg.norm(target_center - anchor_center)),
        'aabb_gap': float(np.linalg.norm(gap_xyz)),
        # Surface and AABB gap are intentionally distinct fields in the API;
        # this implementation uses the documented box-gap proxy for surfaces.
        'surface_gap': float(np.linalg.norm(gap_xyz)),
        'horizontal_center': float(np.linalg.norm(target_center[:2] - anchor_center[:2])),
        'horizontal_gap': float(np.linalg.norm(gap_xy)),
        'viewpoint': None,
        'plane': None,
    }
    observer = _viewpoint_origin(viewpoint)
    if observer is not None:
        metrics['viewpoint'] = float(np.linalg.norm(target_center - observer))
    wall = anchor.get('wall_geometry')
    if isinstance(wall, dict) and wall.get('valid'):
        try:
            origin = np.asarray(wall['origin_xy_m'], dtype=float)
            normal = np.asarray(wall['normal_xy'], dtype=float)
            tangent = np.asarray(wall['tangent_xy'], dtype=float)
            extent = np.asarray(wall['tangent_extent_m'], dtype=float)
            tangent_coordinate = float((target_center[:2] - origin) @ tangent)
            if extent[0] <= tangent_coordinate <= extent[1]:
                metrics['plane'] = float(abs((target_center[:2] - origin) @ normal))
        except (KeyError, TypeError, ValueError):
            metrics['plane'] = None
    return metrics


def canonical_category(name):
    text = ' '.join(re.findall(r'[a-z]+', name.lower()))
    from v4_policy import v7_enabled
    aliases = {**ALIASES, **({'chalk board': 'blackboard', 'chalk boards': 'blackboard',
        'chalkboard': 'blackboard', 'chalkboards': 'blackboard'} if v7_enabled() else {})}
    for source in sorted(aliases, key=len, reverse=True):
        text = re.sub(r'\b' + re.escape(source) + r'\b', aliases[source], text)
    return text


def match_categories(names, objects):
    if isinstance(names, str):
        names = [names]
    from v4_policy import r8_retrieval_enabled
    aliases = ({'sofa': 'couch', 'sofas': 'couch', 'loveseat': 'couch',
                'bookcase': 'bookshelf', 'bookcases': 'bookshelf',
                'cupboard': 'cabinet', 'cupboards': 'cabinet',
                'filing cabinet': 'file cabinet',
                'painting': 'picture', 'paintings': 'picture', 'artwork': 'picture',
                'picture frame': 'picture', 'picture frames': 'picture'}
               if r8_retrieval_enabled() else {})
    def words(name):
        text = canonical_category(name)
        return aliases.get(text, text).split()
    categories = [words(n) for n in names if n.strip()]
    matched = []
    for obj in objects:
        labels = [words(name) for name in category_names(obj)]
        if any(category and words and category[-1] == words[-1]
               and all(w in words for w in category)
               for category in categories for words in labels):
            matched.append(obj)
    return matched


def distance_evidence(targets, anchors, metric=None, viewpoint=None):
    """Return independent geometry metrics for every target/anchor instance pair."""
    pairs = []
    for target in targets:
        for anchor in anchors:
            if target['bbox_id'] == anchor['bbox_id']:
                continue
            values = distance_pair_metrics(target, anchor, viewpoint=viewpoint)
            pair = {'target_id': target['bbox_id'], 'anchor_id': anchor['bbox_id'],
                    'center_distance_m': round(values['center'], 4),
                    'aabb_gap_m': round(values['aabb_gap'], 4),
                    'surface_gap_m': round(values['surface_gap'], 4),
                    'horizontal_center_distance_m': round(values['horizontal_center'], 4),
                    'horizontal_gap_m': round(values['horizontal_gap'], 4),
                    'viewpoint_distance_m': (round(values['viewpoint'], 4)
                                             if values['viewpoint'] is not None else None),
                    'plane_distance_m': (round(values['plane'], 4)
                                         if values['plane'] is not None else None),
                    'metric_status': {name: ('known' if value is not None else 'unknown')
                                      for name, value in values.items()}}
            if metric is not None:
                metric_name = normalize_distance_metric(metric)
                pair['selected_metric'] = metric_name
                pair['selected_distance_m'] = (round(values[metric_name], 4)
                                                if metric_name and values.get(metric_name) is not None
                                                else None)
            pairs.append(pair)
    selected = normalize_distance_metric(metric) if metric is not None else None
    return {'units': 'meters', 'world_axes': 'X/Y horizontal, Z up; these are not camera axes',
            'distance_definition': ('selected metric: ' + selected if selected else
                                    'all metrics reported separately; center distance is not a surface distance'),
            'metric_definitions': {
                'center': 'Euclidean distance between object centers',
                'aabb_gap': 'Euclidean gap between axis-aligned bounding boxes',
                'surface_gap': 'AABB gap proxy for visible surface separation',
                'horizontal_center': 'Euclidean distance between center XY coordinates; Z ignored',
                'horizontal_gap': 'AABB gap in XY coordinates; Z ignored',
                'viewpoint': 'Euclidean distance from target center to explicit observer origin',
                'plane': 'Perpendicular distance to a valid finite fitted wall plane',
            },
            'selected_metric': selected, 'pairs': pairs}


def retrieval_category(name):
    """Remove common appearance modifiers only for category retrieval."""
    text = canonical_category(name)
    if text in {'wall', 'floor', 'ground', 'ceiling'}:
        return text
    modifiers = {'large', 'small', 'big', 'little', 'black', 'white', 'red', 'blue',
                 'green', 'gray', 'grey', 'brown', 'wooden', 'wall', 'mounted'}
    return ' '.join(word for word in text.split() if word not in modifiers)


def visual_fallback_candidates(query, parsed, objects, targets, anchors, limit=6):
    """Return a bounded geometry neighborhood for visual classifier misses.

    The returned objects retain their predicted Top-5 hypotheses, but callers
    must label them as fallback candidates because their category did not match
    the parsed target.  This function never reads evaluation labels or target
    IDs.  It is intentionally disabled for descriptions without a relation or
    appearance clue, where adding unrelated objects would add noise.
    """
    if limit < 1 or not _VISUAL_FALLBACK_TERMS.search(query or ''):
        return []
    seeds = []
    seen_seed_ids = set()
    for item in [*targets, *anchors]:
        if item['bbox_id'] not in seen_seed_ids:
            seen_seed_ids.add(item['bbox_id'])
            seeds.append(item)
    if not seeds:
        return []
    seed_ids = {item['bbox_id'] for item in seeds}
    ranked = []
    for item in objects:
        oid = item['bbox_id']
        if oid in seed_ids:
            continue
        hypotheses = item.get('category_hypotheses', [])
        categories = {retrieval_category(h.get('category', '')) for h in hypotheses}
        if categories and categories.issubset(_SHELL_CATEGORIES):
            continue
        gap = min(distance_pair_metrics(item, seed)['aabb_gap'] for seed in seeds)
        center = min(distance_pair_metrics(item, seed)['center'] for seed in seeds)
        ranked.append((gap, center, oid, item))
    ranked.sort(key=lambda row: (row[0], row[1], row[2]))
    by_id = {item['bbox_id']: item for _, _, _, item in ranked}
    selected = []

    def add(item, reason):
        if item is None or item['bbox_id'] in seed_ids or item['bbox_id'] in selected:
            return
        selected.append(item['bbox_id'])
        row = dict(item)
        row['candidate_role'] = 'visual_fallback'
        row['fallback_reason'] = reason
        pair = next((entry for entry in ranked if entry[2] == item['bbox_id']), None)
        if pair is not None:
            row['fallback_distance_to_seed_m'] = round(float(pair[0]), 4)
            row['fallback_center_distance_to_seed_m'] = round(float(pair[1]), 4)
        return row

    # Preserve the React candidate pool while making explicit extreme queries
    # visible even when the classifier's top-5 category misses the target.
    lower = (query or '').casefold()
    extreme = []
    if re.search(r'\b(?:farthest|furthest)\b', lower):
        relation_anchors = []
        for relation in parsed.get('Relations', []):
            if not isinstance(relation, dict) or relation.get('relation') != 'farthest':
                continue
            name = relation.get('anchor_category')
            if isinstance(name, str) and name.strip():
                relation_anchors.extend(match_retrieval_categories([name], objects))
        relation_anchors = list({item['bbox_id']: item for item in relation_anchors}.values()) or list(anchors)
        if relation_anchors:
            distance_rows = []
            for item in by_id.values():
                distances = [distance_pair_metrics(item, anchor)['aabb_gap']
                             for anchor in relation_anchors if item['bbox_id'] != anchor['bbox_id']]
                if distances:
                    distance_rows.append((max(distances), item['bbox_id'], item))
            extreme.extend(item for _, _, item in sorted(distance_rows, reverse=True)[:3])
    if re.search(r'\b(?:lowest|shortest)\b', lower):
        extreme.extend(sorted(by_id.values(), key=lambda item: (
            float(item['bbox_3d'][2] - item['bbox_3d'][5] / 2), item['bbox_id']))[:10])
    if re.search(r'\b(?:highest|tallest)\b', lower):
        if 'highest' in lower:
            key = lambda item: float(item['bbox_3d'][2] + item['bbox_3d'][5] / 2)
        else:
            key = lambda item: float(item['bbox_3d'][5])
        extreme.extend(sorted(by_id.values(), key=lambda item: (-key(item), item['bbox_id']))[:10])
    if re.search(r'\b(?:largest|smallest)\b', lower):
        volumes = lambda item: float(np.prod(np.asarray(item['bbox_3d'][3:], dtype=float)))
        reverse = 'largest' in lower
        extreme.extend(sorted(by_id.values(), key=lambda item: ((-1 if reverse else 1) * volumes(item), item['bbox_id']))[:10])

    result = []
    for item in extreme:
        row = add(item, 'query_extreme')
        if row is not None:
            result.append(row)
        if len(result) >= limit:
            return result
    for gap, center, _, item in ranked:
        row = add(item, 'geometry_neighborhood')
        if row is not None:
            result.append(row)
        if len(result) >= limit:
            break
    return result


def match_retrieval_categories(names, objects):
    if isinstance(names, str):
        names = [names]
    return match_categories([retrieval_category(name) for name in names], objects)


def superlative_rejection(query, relation):
    kind = relation.get('relation')
    phrase = relation.get('phrase', '')
    pattern = r'\b(nearest|closest)\b' if kind == 'nearest' else r'\b(farthest|furthest)\b'
    if (not isinstance(phrase, str) or not phrase.strip() or
            phrase.casefold() not in query.casefold() or not re.search(pattern, phrase, re.I)):
        return 'no matching superlative quote in original query'
    if re.search(r"\b(no|not|never|without|except)\b|n['’]t\b", query, re.I):
        return 'negation/exclusion requires further parsing'
    return None


def validate_distance_relations(query, parsed):
    """Suppress unsupported distance operators while retaining raw API calls separately."""
    relations = parsed.get('Relations', [])
    if not isinstance(relations, list) or any(not isinstance(r, dict) for r in relations):
        raise ValueError('Relations must be a list of objects')
    valid, rejected = [], []
    for relation in relations:
        reason = superlative_rejection(query, relation) if relation.get('relation') in ('nearest','farthest') else None
        if reason:
            rejected.append({'relation': relation, 'reason': reason})
            valid.append({**relation, 'relation': 'other'})
        else:
            valid.append(relation)
    return {**parsed, 'Relations': valid, 'rejected_distance_relations': rejected}


def distance_rankings(query, parsed, targets, anchors, viewpoint=None,
                      tie_tolerance_m=DISTANCE_TIE_TOLERANCE_M):
    """Rank explicit superlatives separately for each bound anchor instance.

    Near/far are intentionally returned as ignored evidence unless a future
    calibrated predicate is added.  A near-equal best distance is represented
    by multiple ``best_ids`` instead of being broken by object ID.
    """
    rankings, ignored = [], list(parsed.get('rejected_distance_relations', []))
    relations = parsed.get('Relations', [])
    if not isinstance(relations, list):
        return {'rankings': [], 'ignored': [{'reason': 'Relations must be a list'}]}
    for relation in relations:
        if not isinstance(relation, dict):
            ignored.append({'reason': 'invalid relation object'})
            continue
        kind = relation.get('relation')
        if kind in ('near', 'far'):
            ignored.append({'relation': relation,
                            'reason': kind + ' has no calibrated distance threshold; evidence is unknown'})
            continue
        if kind not in ('nearest', 'farthest'):
            continue
        phrase = relation.get('phrase', '')
        reason = superlative_rejection(query, relation)
        if reason:
            ignored.append({'relation': relation, 'reason': reason})
            continue
        target_name, anchor_name = relation.get('target_category'), relation.get('anchor_category')
        if not isinstance(target_name, str) or not isinstance(anchor_name, str) or not target_name or not anchor_name:
            ignored.append({'relation': relation, 'reason': 'missing category binding'})
            continue
        metric = _distance_metric_for_relation(relation)
        if metric is None:
            ignored.append({'relation': relation,
                            'reason': 'unsupported distance metric: ' + str(
                                relation.get('distance_metric', relation.get('metric')))})
            continue
        if metric in ('viewpoint',) and _viewpoint_origin(viewpoint) is None:
            ignored.append({'relation': relation,
                            'reason': 'viewpoint distance requires an explicit numeric observer origin'})
            continue
        try:
            tie_tolerance = float(relation.get('tie_tolerance_m', tie_tolerance_m))
        except (TypeError, ValueError):
            tie_tolerance = tie_tolerance_m
        if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
            ignored.append({'relation': relation, 'reason': 'tie_tolerance_m must be finite and non-negative'})
            continue
        selected_targets = match_retrieval_categories([target_name], targets)
        selected_anchors = match_retrieval_categories([anchor_name], anchors)
        target_instance_id = relation.get('target_id', relation.get('target_instance_id'))
        anchor_instance_id = relation.get('anchor_id', relation.get('anchor_instance_id'))
        if type(target_instance_id) is int:
            selected_targets = [obj for obj in selected_targets
                                if obj['bbox_id'] == target_instance_id]
        if type(anchor_instance_id) is int:
            selected_anchors = [obj for obj in selected_anchors
                                if obj['bbox_id'] == anchor_instance_id]
        if not selected_targets or not selected_anchors:
            ignored.append({'relation': relation, 'reason': 'unmatched target/reference category'})
            continue
        for anchor in selected_anchors:
            distances, unknown_targets = [], []
            for target in selected_targets:
                if target['bbox_id'] == anchor['bbox_id']:
                    continue
                values = distance_pair_metrics(target, anchor, viewpoint=viewpoint)
                distance = values.get(metric)
                if distance is None or not np.isfinite(distance):
                    unknown_targets.append({'target_id': target['bbox_id'],
                                            'distance_m': None, 'metric': metric,
                                            'distance_metrics': values,
                                            'status': 'unknown',
                                            'reason': 'selected distance metric is unavailable'})
                    continue
                distances.append({'target_id': target['bbox_id'],
                                  'distance_m': float(distance),
                                  'metric': metric,
                                  'distance_metrics': values})
            distances.sort(key=lambda value: value['distance_m'], reverse=kind == 'farthest')
            if not distances:
                ignored.append({'relation': relation, 'anchor_id': anchor['bbox_id'],
                                'reason': 'no finite target distances for selected metric',
                                'metric': metric})
            best = distances[0]['distance_m'] if distances else None
            tied = sorted(value['target_id'] for value in distances
                          if abs(value['distance_m'] - best) <= tie_tolerance)
            exact_tied = sorted(value['target_id'] for value in distances
                                if abs(value['distance_m'] - best) <= 1e-6)
            rankings.append({'relation': kind, 'phrase': phrase, 'anchor_id': anchor['bbox_id'],
                             'ordered_targets': distances, 'unknown_targets': unknown_targets,
                             'unknown_target_ids': sorted(value['target_id'] for value in unknown_targets),
                             'comparison_status': 'unknown' if unknown_targets else 'known',
                             'best_ids': [] if unknown_targets else tied,
                             'measured_best_ids': tied,
                             'exact_best_ids': exact_tied,
                             'near_equal': len(tied) > len(exact_tied),
                             'tie_tolerance_m': tie_tolerance,
                             'metric': metric,
                             'metric_definition': distance_evidence([], [], metric=metric)['metric_definitions'][metric],
                             'anchor_binding': {'anchor_id': anchor['bbox_id'],
                                                'anchor_category': retrieval_category(anchor_name),
                                                'status': 'instance_bound'},
                             'scope': 'target category candidates; each anchor instance ranked independently'})
    return {'rankings': rankings, 'ignored': ignored,
            'policy': 'explicit nearest/farthest comparisons; no cross-anchor averaging; '
                      'near/far remain unknown without calibration',
            'metric_definitions': distance_evidence([], [], metric=None)['metric_definitions']}


def propose_category_checks(objects, targets, anchors, limit):
    """Check label matches first, then spatial neighbors; no fixed object IDs."""
    if limit < 1:
        raise ValueError('review-limit must be positive')
    anchor_ids = {obj['bbox_id'] for obj in anchors}
    selected = targets[:limit]
    selected_ids = {obj['bbox_id'] for obj in selected}
    remaining = [obj for obj in objects if obj['bbox_id'] not in selected_ids | anchor_ids]
    if anchors:
        remaining.sort(key=lambda obj: min(float(np.linalg.norm(np.asarray(obj['bbox_3d'][:3]) - anchor['bbox_3d'][:3]))
                                          for anchor in anchors))
    for obj in remaining[:max(0, limit-len(selected))]:
        selected.append(obj)
    return selected
