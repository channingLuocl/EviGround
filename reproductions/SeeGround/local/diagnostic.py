"""Scoped constraint diagnosis, with shared observer/reference hypotheses, never GT."""
from itertools import product
from math import prod
import re
from math import ceil
import numpy as np
from evidence import match_retrieval_categories, retrieval_category
from query_rewrite import CONTAINMENT_PATTERN


VERTICAL_WORDS = {
    'above': 'above', 'over': 'above', 'below': 'below',
    'under': 'below', 'beneath': 'below',
}

# A vertical relation can be expressed by adjacent boxes whose XY projections
# do not overlap exactly. Keep this tolerance small enough to reject objects
# elsewhere in the room, while covering ordinary ScanNet box noise.
HORIZONTAL_ALIGNMENT_GAP_M = 0.5
VERTICAL_CLEARANCE_TOLERANCE_M = 0.05


PARSER_INSTRUCTIONS = '''
Field definitions within the SAME top-level object; do not omit Target or Anchor:
"Observer":{"origin":{"kind":"room_center|category|unspecified",
"category":"category or empty","phrase":"exact quote or empty"},
"facing":{"category":"category or empty","group":false,"phrase":"exact quote or empty"}},
"Attributes":[{"category":"category","phrase":"exact appearance/size quote"}],
"Unhandled":["exact quotes for conditions not represented in these fields"].
Represent EVERY query condition, including appearance, corner, on-ground and negation.
Do not silently omit conditions; put unsupported ones in Unhandled.
Use room_center ONLY for an explicit middle/center of room statement.
Facing an object without a stated origin leaves origin unspecified.
Use facing.group=true ONLY when the description explicitly faces a plural/group.
Direction relations use left|right|front|back, plus scope=relative_anchor|side.
relative_anchor means target is on a named reference's side in the observer frame;
side means a signed side of the observer, NOT leftmost/rightmost among peers.
Use between with anchor_categories for a target located between two named references.
Represent smallest/largest/biggest/tallest/shortest/highest/lowest and the
comparatives larger/smaller/bigger/taller/shorter as target Attributes, not Unhandled.
Adjective phrases such as tall skinny are Attributes, never Relations. Preserve every modifier.
Use other+Unhandled for unsupported extremes, object-intrinsic axes, ordinal or unsupported relations.
Includes/contains a named object is a containment/composition relation, not appearance;
use other+Unhandled because this relation has no implemented predicate.
Separate facing references from target relations; include their categories in Anchor.
Resolve pronouns to an actual category in the description, never use a pronoun as a category.
Every relation and observer phrase must be a nonempty EXACT quote.
For Observer use the full standing/facing clause; do not invent the speaker's position.
'''


def quoted(query, phrase):
    return isinstance(phrase, str) and bool(phrase.strip()) and phrase.casefold() in query.casefold()


def recover_distance_relations(query, parsed):
    """Recover only a complete, simple distance clause with explicit bindings."""
    parsed = dict(parsed)
    relations = list(parsed.get('Relations') or [])
    targets = parsed.get('Target') if isinstance(parsed.get('Target'), list) else []
    anchors = parsed.get('Anchor') if isinstance(parsed.get('Anchor'), list) else []
    if len(targets) != 1 or len(anchors) != 1:
        return parsed
    if any(isinstance(r, dict) and r.get('relation') in ('nearest', 'farthest') for r in relations):
        return parsed
    target, anchor = (name.strip() for name in (targets[0], anchors[0]))
    if not target or not anchor:
        return parsed
    pattern = (r'(?:it is\s+)?(?:the\s+)?' + re.escape(target) + r's?\s+(?:is\s+)?'
               r'(?P<operator>nearest|closest|farthest|furthest)\s+(?:away\s+)?'
               r'(?:to\s+|from\s+)?(?:the\s+)?' + re.escape(anchor) + r's?[.!?]?')
    match = re.fullmatch(pattern, query.strip(), re.I)
    if match:
        kind = 'nearest' if match['operator'].lower() in ('nearest', 'closest') else 'farthest'
        recovered = {'target_category': target, 'anchor_category': anchor,
                     'relation': kind, 'scope': '', 'phrase': query.strip()}
        parsed['Relations'] = relations + [recovered]
        parsed['recovered_distance_relations'] = [recovered]
    return parsed


def recover_vertical_relations(query, parsed):
    """Recover one explicit vertical clause when the parser omitted or mislabeled it."""
    parsed = dict(parsed)
    relations = [dict(r) for r in (parsed.get('Relations') or []) if isinstance(r, dict)]
    targets = parsed.get('Target') if isinstance(parsed.get('Target'), list) else []
    anchors = parsed.get('Anchor') if isinstance(parsed.get('Anchor'), list) else []
    if len(targets) != 1 or len(anchors) != 1:
        return parsed

    # Correct the common Qwen failure: "right above" returned as near/other.
    for relation in relations:
        phrase = relation.get('phrase', '')
        match = re.search(r'\b(?:right\s+above|directly\s+above|above|over|below|directly\s+under|under|beneath)\b',
                          phrase, re.I)
        if match:
            relation['relation'] = VERTICAL_WORDS[re.search(
                r'\b(above|over|below|under|beneath)\b', match.group(), re.I).group().lower()]
            relation['scope'] = 'relative_anchor'
            relation['target_category'] = relation.get('target_category') or targets[0]
            relation['anchor_category'] = relation.get('anchor_category') or anchors[0]

    if any(r.get('relation') in ('above', 'below') for r in relations):
        parsed['Relations'] = relations
        return parsed

    vertical = re.search(
        r'\b(?:right\s+above|directly\s+above|above|over|below|directly\s+under|under|beneath)\b',
        query, re.I)
    if not vertical:
        parsed['Relations'] = relations
        return parsed
    operator = re.search(r'\b(above|over|below|under|beneath)\b', vertical.group(), re.I).group().lower()
    # Keep the quoted clause short and exact: operator plus the following noun phrase.
    tail = query[vertical.end():]
    tail_match = re.match(r'\s+(?:the|a|an)\s+[A-Za-z][A-Za-z -]*', tail)
    end = vertical.end() + (tail_match.end() if tail_match else 0)
    phrase = query[vertical.start():end].strip().rstrip('.,!?')
    relations.append({'target_category': targets[0], 'anchor_category': anchors[0],
                      'relation': 'above' if operator in ('above', 'over') else 'below',
                      'scope': 'relative_anchor', 'phrase': phrase})
    parsed['Relations'] = relations
    return parsed


def recover_between_relations(query, parsed):
    """Recover a two-reference horizontal relation omitted by the parser."""
    parsed = dict(parsed)
    relations = [dict(r) for r in (parsed.get('Relations') or []) if isinstance(r, dict)]
    targets = parsed.get('Target') if isinstance(parsed.get('Target'), list) else []
    anchors = parsed.get('Anchor') if isinstance(parsed.get('Anchor'), list) else []
    if len(targets) != 1 or len(anchors) < 2:
        parsed['Relations'] = relations
        return parsed
    for relation in relations:
        if relation.get('relation') == 'between':
            names = relation.get('anchor_categories')
            if (not isinstance(names, list) or len(names) != 2) and len(anchors) >= 2:
                relation['anchor_categories'] = [name.strip() for name in anchors[:2]
                                                 if isinstance(name, str) and name.strip()]
            relation.setdefault('scope', 'relative_anchors')
            parsed['Relations'] = relations
            return parsed
    match = re.search(r'\bbetween\b', query, re.I)
    if not match:
        parsed['Relations'] = relations
        return parsed
    anchor_categories = [name.strip() for name in anchors[:2]
                         if isinstance(name, str) and name.strip()]
    if len(anchor_categories) != 2:
        parsed['Relations'] = relations
        return parsed
    phrase = query[match.start():].strip().rstrip('.,!?')
    relations.append({'target_category': targets[0],
                      'anchor_categories': anchor_categories,
                      'relation': 'between', 'scope': 'relative_anchors',
                      'phrase': phrase})
    parsed['Relations'] = relations
    parsed['recovered_between_relations'] = [relations[-1]]
    return parsed


def covered_attribute_request(query, parsed):
    targets = parsed.get('Target', [])
    targets = [targets] if isinstance(targets, str) else targets
    if len(targets) != 1 or parsed.get('Anchor') or parsed.get('Relations'):
        return False
    words = []
    allowed = set(('small smallest smaller large largest larger big biggest bigger little '
                   'skinny narrow wide thin thick tall tallest taller high highest short '
                   'shortest shorter low lowest dark light black white red blue green gray '
                   'grey brown wooden mounted square round').split())
    for attribute in parsed.get('Attributes', []):
        if retrieval_category(attribute.get('category', '')) != retrieval_category(targets[0]):
            return False
        tokens = attribute.get('phrase', '').casefold().split()
        if not tokens or not set(tokens).issubset(allowed):
            return False
        words.extend(tokens)
    if not words:
        return False
    match = re.fullmatch(r'\s*(?:(?:choose|select|find)\s+|it is\s+)?(?:the\s+)?'
                         r'(?P<attributes>[a-z ]+)\s+' + re.escape(targets[0]) +
                         r'\s*[.!?]?\s*', query, re.I)
    return bool(match and sorted(match['attributes'].casefold().split()) == sorted(words))


def validate_structure(query, parsed):
    parsed = recover_between_relations(query, recover_vertical_relations(query, dict(parsed)))
    # Recover only a complete, simple size request; compound conditions stay untouched.
    targets = parsed.get('Target', [])
    targets = [targets] if isinstance(targets, str) else targets
    if len(targets) == 1:
        match = re.fullmatch(
            r'\s*(?:(?:choose|select|find|pick|click)\s+|it\s+is\s+)?(?:the\s+)?'
            r'(smallest|largest|biggest|tallest|shortest|highest|lowest|'
            r'larger|smaller|bigger|taller|shorter)\s+'
            + re.escape(targets[0]) + r'\s*[.!?]?\s*', query, re.I)
        if (match and not parsed.get('Anchor') and not parsed.get('Relations')
                and isinstance(parsed.get('Unhandled'), list)
                and all(isinstance(p, str) for p in parsed['Unhandled'])):
            parsed['Attributes'] = [{'category': targets[0], 'phrase': match.group(1)}]
            parsed['Unhandled'] = [phrase for phrase in parsed.get('Unhandled', [])
                                   if phrase.strip().rstrip('.!?').casefold() not in
                                   (query.strip().rstrip('.!?').casefold(), match.group(1).casefold())]
    missing = []
    relations = parsed.get('Relations', [])
    valid = []
    for relation in relations:
        kind = relation.get('relation')
        phrase = relation.get('phrase')
        valid_quote = quoted(query, phrase)
        target_name = relation.get('target_category')
        anchor_name = relation.get('anchor_category', '')
        anchor_names = relation.get('anchor_categories', []) if kind == 'between' else [anchor_name]
        valid_quote = valid_quote and isinstance(target_name, str) and bool(target_name.strip())
        if kind == 'between':
            valid_quote = (valid_quote and isinstance(anchor_names, list) and len(anchor_names) == 2
                           and all(isinstance(name, str) and bool(name.strip()) for name in anchor_names)
                           and bool(re.search(r'\bbetween\b', phrase, re.I)))
        elif kind in ('nearest', 'farthest', 'near', 'above', 'below') or relation.get('scope') == 'relative_anchor':
            valid_quote = valid_quote and isinstance(anchor_name, str) and bool(anchor_name.strip())
        if kind in ('left', 'right', 'front', 'back'):
            words = {'left': r'\bleft\b', 'right': r'\bright\b',
                     'front': r'\b(front|ahead)\b', 'back': r'\b(back|behind)\b'}
            valid_quote = valid_quote and bool(re.search(words[kind], phrase, re.I))
            valid_quote = valid_quote and relation.get('scope') in ('relative_anchor', 'side')
        if kind in ('above', 'below'):
            valid_quote = valid_quote and bool(re.search(r'\b(above|over|below|under|beneath)\b', phrase, re.I))
        if not valid_quote or kind not in ('nearest', 'farthest', 'near', 'left', 'right', 'front', 'back', 'above', 'below', 'between'):
            missing.append('unsupported relation: ' + phrase if quoted(query, phrase)
                           else 'parser produced an unquoted relation; not a query condition')
        else:
            valid.append(relation)
    parsed['Relations'] = valid
    observer = parsed.get('Observer', {})
    if not isinstance(observer, dict):
        observer = {}
    origin, facing = observer.get('origin', {}), observer.get('facing', {})
    if not isinstance(origin, dict):
        origin = {}
    if not isinstance(facing, dict):
        facing = {}
    if origin.get('kind') == 'room_center':
        phrase = origin.get('phrase', '')
        if (not quoted(query, phrase) or not re.search(r'\b(middle|cent(?:er|re))\b', phrase, re.I)
                or not re.search(r'\broom\b', phrase, re.I)):
            missing.append('room-center origin has no supporting quote')
            origin = {}
    elif origin.get('kind') == 'category':
        if not quoted(query, origin.get('phrase')) or not isinstance(origin.get('category'), str):
            missing.append('named origin has no supporting quote/category')
            origin = {}
    else:
        origin = {}
    if facing.get('category'):
        phrase = facing.get('phrase', '')
        if (not quoted(query, phrase) or not re.search(r'\b(fac(?:e|ing)|look(?:ing)?)\b', phrase, re.I)
                or not isinstance(facing.get('category'), str) or type(facing.get('group', False)) is not bool):
            missing.append('facing direction has no supporting quote/category')
            facing = {}
        elif facing.get('group') and not re.search(r'\b' + re.escape(retrieval_category(facing['category'])) + r's\b', phrase, re.I):
            # Irregular plurals require an explicit group quote instead of guessing a centroid.
            if not re.search(r'\b(group|all|both)\b', phrase, re.I):
                missing.append('group-facing has no explicit plural/group quote')
                facing = {}
    parsed['Observer'] = {'origin': origin, 'facing': facing}
    attrs = parsed.get('Attributes')
    if not isinstance(attrs, list) or any(not isinstance(a, dict) for a in attrs):
        missing.append('attribute coverage not supplied by parser')
        attrs = []
    bound_categories = set()
    for key in ('Target', 'Anchor'):
        names = parsed.get(key, [])
        names = names if isinstance(names, list) else [names]
        bound_categories.update(retrieval_category(name) for name in names
                                if isinstance(name, str) and retrieval_category(name))
    for relation in parsed['Relations']:
        for key in ('target_category', 'anchor_category'):
            name = relation.get(key, '')
            if isinstance(name, str) and retrieval_category(name):
                bound_categories.add(retrieval_category(name))
        for name in relation.get('anchor_categories', []):
            if isinstance(name, str) and retrieval_category(name):
                bound_categories.add(retrieval_category(name))
    observer = parsed['Observer']
    for role in ('origin', 'facing'):
        name = observer.get(role, {}).get('category', '')
        if isinstance(name, str) and retrieval_category(name):
            bound_categories.add(retrieval_category(name))
    valid_attrs = []
    for attribute in attrs:
        category_name = attribute.get('category')
        if (quoted(query, attribute.get('phrase')) and isinstance(category_name, str)
                and retrieval_category(category_name) in bound_categories):
            valid_attrs.append(attribute)
        elif quoted(query, attribute.get('phrase')):
            missing.append('attribute category is not a supplied target/reference: ' + category_name)
            missing.append('appearance condition not represented: ' + attribute.get('phrase', ''))
    parsed['Attributes'] = valid_attrs
    if len(parsed['Attributes']) != len(attrs):
        missing.append('unquoted or unbound appearance attribute')
    covered = ' '.join(a['phrase'] for a in parsed['Attributes'])
    for word in re.findall(r'\b(smallest|largest|biggest|tallest|shortest|highest|lowest|'
                           r'larger|smaller|bigger|taller|shorter|large|small|big|little|'
                           r'skinny|narrow|wide|thin|thick|dark|light|black|white|red|blue|'
                           r'green|gr[ae]y|brown|wooden|mounted)\b', query, re.I):
        if not re.search(r'\b' + re.escape(word) + r'\b', covered, re.I):
            missing.append('appearance condition not represented: ' + word)
    unhandled = parsed.get('Unhandled')
    if not isinstance(unhandled, list) or any(not isinstance(x, str) for x in unhandled):
        missing.append('condition coverage not supplied by parser')
        parsed['Unhandled'] = []
    else:
        parsed['Unhandled'] = [p for p in unhandled if quoted(query, p)]
        if len(parsed['Unhandled']) != len(unhandled):
            missing.append('parser produced unquoted Unhandled entries; these are not query conditions')
        if covered_attribute_request(query, parsed):
            parsed['Unhandled'] = [p for p in parsed['Unhandled']
                                   if p.strip().rstrip('.!?').casefold() !=
                                   query.strip().rstrip('.!?').casefold()]
        missing.extend('unhandled condition: ' + p for p in parsed['Unhandled'])
    targets = parsed['Target'] if isinstance(parsed['Target'], list) else [parsed['Target']]
    if any(retrieval_category(name) in ('one','correct one','it','object','thing') for name in targets):
        missing.append('target category is an unresolved pronoun or generic noun')
    if re.search(r"\b(no|not|nothing|never|without|except)\b|n['’]t\b", query, re.I):
        missing.append('negation/exclusion is unsupported')
        parsed['Relations'] = []
    direction_query = re.sub(r'\bright\s+(?:above|over|below|under|beneath)\b', '', query, flags=re.I)
    if re.search(r'\b(left|right|ahead|behind)\b', direction_query, re.I) and not any(
            r['relation'] in ('left', 'right', 'front', 'back') for r in parsed['Relations']):
        missing.append('direction condition not represented')
    for pattern, kinds, label in ((r'\b(nearest|closest|farthest|furthest)\b', ('nearest','farthest'), 'distance extreme'),
                                  (r'\b(next to|beside|near)\b', ('near',), 'proximity')):
        if re.search(pattern, query, re.I) and not any(r['relation'] in kinds for r in parsed['Relations']):
            missing.append(label + ' condition not represented')
    if re.search(r'\b(corner|leftmost|rightmost|second|third)\b', query, re.I):
        missing.append('corner/ordinal/support/extreme conditions require additional predicates')
    if re.search(r'\b(above|over|below|under|underneath|beneath)\b', query, re.I) and not any(
            r['relation'] in ('above', 'below') for r in parsed['Relations']):
        missing.append('vertical condition not represented')
    for match in re.finditer(CONTAINMENT_PATTERN, query, re.I):
        missing.append('unsupported containment condition: ' + match.group().rstrip())
    # A quoted attribute plus its bound category is a noun phrase, not a relation.
    covered_nouns = {a['phrase'].casefold() + ' ' + a['category'].casefold()
                     for a in parsed['Attributes']}
    missing = [item for item in missing if not (
        item.startswith('unsupported relation: ') and
        item[len('unsupported relation: '):].casefold() in covered_nouns)]
    parsed['diagnostic_missing'] = missing
    return parsed


def observer_frames(parsed, objects, room_center):
    origin, facing = parsed['Observer']['origin'], parsed['Observer']['facing']
    if not origin or not facing:
        return [], ['observer origin or facing direction is unspecified']
    origins = ([{'bbox_id': None, 'bbox_3d': list(room_center)}] if origin['kind'] == 'room_center'
               else match_retrieval_categories([origin['category']], objects))
    faces = match_retrieval_categories([facing['category']], objects)
    if not origins or not faces:
        return [], ['observer origin/facing category has no detected instance']
    groups = [faces] if facing.get('group') else [[face] for face in faces]
    frames = []
    missing = []
    for start, group in product(origins, groups):
        at = np.mean([face['bbox_3d'][:3] for face in group], axis=0)
        forward = at - np.asarray(start['bbox_3d'][:3])
        forward[2] = 0
        norm = np.linalg.norm(forward)
        if norm <= 1e-6:
            missing.append('degenerate observer origin/facing pair')
            continue
        forward /= norm
        right = np.cross(forward, [0., 0., 1.])
        frames.append({'origin_id': start['bbox_id'], 'facing_ids': [o['bbox_id'] for o in group],
                       'origin': list(start['bbox_3d'][:3]), 'forward': forward.tolist(), 'right': right.tolist()})
    return frames, missing


def attributes_for(parsed, category):
    return [a['phrase'] for a in parsed['Attributes'] if retrieval_category(a['category']) == retrieval_category(category)]


def _box_metrics(obj):
    x, y, z, width, length, height = map(float, obj['bbox_3d'])
    return {'volume_m3': width * length * height, 'footprint_m2': width * length,
            'height_m': height, 'center_z_m': z,
            'bottom_z_m': z - height / 2., 'top_z_m': z + height / 2.,
            'horizontal_short_extent_m': min(width, length),
            'horizontal_long_extent_m': max(width, length),
            'horizontal_aspect_ratio': max(width, length) / min(width, length),
            'height_to_horizontal_long_ratio': height / max(width, length)}


def _pair_geometry(target, anchor):
    a = np.asarray(target['bbox_3d'], dtype=float)
    b = np.asarray(anchor['bbox_3d'], dtype=float)
    a_min, a_max = a[:3] - a[3:] / 2., a[:3] + a[3:] / 2.
    b_min, b_max = b[:3] - b[3:] / 2., b[:3] + b[3:] / 2.
    aabb_gap_xyz = np.maximum(np.abs(a[:3] - b[:3]) - (a[3:] + b[3:]) / 2., 0.)
    horizontal_gap_xy = np.maximum(np.abs(a[:2] - b[:2]) - (a[3:5] + b[3:5]) / 2., 0.)
    horizontal_overlap = np.maximum(np.minimum(a_max[:2], b_max[:2]) - np.maximum(a_min[:2], b_min[:2]), 0.)
    vertical_overlap = max(0., min(a_max[2], b_max[2]) - max(a_min[2], b_min[2]))
    return {'target_id': target['bbox_id'], 'anchor_id': anchor['bbox_id'],
            'center_z_delta_m': float(a[2] - b[2]),
            'target_bottom_minus_anchor_top_m': float(a_min[2] - b_max[2]),
            'anchor_bottom_minus_target_top_m': float(b_min[2] - a_max[2]),
            'aabb_gap_m': float(np.linalg.norm(aabb_gap_xyz)),
            'horizontal_aabb_gap_m': float(np.linalg.norm(horizontal_gap_xy)),
            'horizontal_gap_m': float(np.linalg.norm(horizontal_gap_xy)),
            'horizontal_overlap_m2': float(horizontal_overlap[0] * horizontal_overlap[1]),
            'vertical_overlap_m': float(vertical_overlap),
            'center_distance_m': float(np.linalg.norm(a[:3] - b[:3]))}


def _proximity_records(targets, anchor):
    """Return candidate-relative AABB-gap evidence for one near reference.

    Proximity has no calibrated scene-wide threshold, so this helper only ranks
    the live target candidates for the bound anchor.  The records are evidence
    for review; callers must not use ``relative_rank`` as a hard predicate.
    """
    records = [_pair_geometry(target, anchor) for target in targets
               if target['bbox_id'] != anchor['bbox_id']]
    records.sort(key=lambda item: (item['aabb_gap_m'], item['target_id']))
    candidate_count = len(records)
    best_gap = records[0]['aabb_gap_m'] if records else None
    best_ids = ([item['target_id'] for item in records
                 if abs(item['aabb_gap_m'] - best_gap) <= 1e-6]
                if best_gap is not None else [])
    for record in records:
        # Equal gaps keep equal ranks: overlapping AABBs cannot distinguish
        # contact, adjacency or physical shape, even if centers differ.
        rank = 1 + sum(item['aabb_gap_m'] < record['aabb_gap_m'] - 1e-6
                       for item in records)
        record.update({
            'relation': 'near',
            'passes': None,
            'soft_metric': 'aabb_gap_m',
            'relative_rank': rank,
            'candidate_count': candidate_count,
            'best_ids': best_ids,
            'relative_percentile': (float(rank - 1) / float(candidate_count - 1)
                                    if candidate_count > 1 else 0.0),
            'policy': 'candidate-relative AABB evidence; near is not nearest; '
                      'no calibrated threshold or exclusion; zero gap does not prove contact',
        })
    return records


def _between_geometry(target, anchor_a, anchor_b):
    """Measure whether a target footprint lies in the horizontal corridor A-B."""
    target_box = np.asarray(target['bbox_3d'], dtype=float)
    a_box = np.asarray(anchor_a['bbox_3d'], dtype=float)
    b_box = np.asarray(anchor_b['bbox_3d'], dtype=float)
    start, end, point = a_box[:2], b_box[:2], target_box[:2]
    walls = [anchor for anchor in (anchor_a, anchor_b) if retrieval_category(anchor['target']) == 'wall']
    wall_record = {}
    if walls:
        center_comparison = _between_geometry(target,
                           {**anchor_a, 'target': 'reference point'},
                           {**anchor_b, 'target': 'reference point'})
        if len(walls) != 1 or not walls[0].get('wall_geometry', {}).get('valid'):
            return {'target_id': target['bbox_id'], 'anchor_ids': [anchor_a['bbox_id'], anchor_b['bbox_id']],
                    'relation': 'between', 'passes': None, 'reference_policy': 'finite wall projection',
                    'reason': 'between needs exactly one wall with a valid fitted vertical plane',
                    'wall_fits': {str(w['bbox_id']): w.get('wall_geometry') for w in walls},
                    'center_point_comparison': center_comparison}
        from wall_geometry import wall_reference_point
        wall = walls[0]
        other = anchor_b if wall is anchor_a else anchor_a
        endpoint, clamped = wall_reference_point(wall['wall_geometry'], other)
        wall_record = {'reference_policy': 'other anchor projected onto finite wall segment',
                       'wall_id': wall['bbox_id'], 'wall_reference_xy_m': endpoint.tolist(),
                       'wall_projection_clamped': clamped, 'wall_fit': wall['wall_geometry'],
                       'center_point_comparison': center_comparison}
        if wall is anchor_a:
            start = endpoint
        else:
            end = endpoint
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 1e-6:
        return {'target_id': target['bbox_id'], 'anchor_ids': [anchor_a['bbox_id'], anchor_b['bbox_id']],
                'relation': 'between', 'projection_t': None, 'perpendicular_distance_m': None,
                'lateral_tolerance_m': None, 'within_segment': False, 'passes': False,
                'reason': 'reference endpoints are coincident', **wall_record}
    unit = direction / length
    normal = np.asarray([-unit[1], unit[0]])
    projection_t = float(np.dot(point - start, direction) / (length * length))
    projected = start + projection_t * direction
    perpendicular_distance = float(np.linalg.norm(point - projected))
    # Project the target's XY AABB onto the corridor normal. The small margin
    # covers reconstruction noise without allowing a separate nearby object.
    half_lateral_extent = 0.5 * (abs(normal[0]) * target_box[3] +
                                 abs(normal[1]) * target_box[4])
    lateral_tolerance = max(0.15, float(half_lateral_extent) + 0.10)
    within_segment = 0.0 <= projection_t <= 1.0
    return {'target_id': target['bbox_id'], 'anchor_ids': [anchor_a['bbox_id'], anchor_b['bbox_id']],
            'relation': 'between', 'projection_t': projection_t,
            'perpendicular_distance_m': perpendicular_distance,
            'lateral_tolerance_m': lateral_tolerance,
            'within_segment': within_segment,
            'passes': bool(within_segment and perpendicular_distance <= lateral_tolerance),
            'reference_center_distance_m': float(np.linalg.norm(a_box[:2] - b_box[:2])),
            'reference_endpoint_distance_m': length, **wall_record}


def _vertical_match(pair, relation):
    # Height difference alone would accept unrelated objects elsewhere in the room.
    # A small negative clearance tolerates annotation noise while rejecting
    # boxes whose vertical ranges substantially intersect.
    aligned = (pair['horizontal_overlap_m2'] > 0.0 or
               pair['horizontal_gap_m'] <= HORIZONTAL_ALIGNMENT_GAP_M)
    tolerance = VERTICAL_CLEARANCE_TOLERANCE_M
    if relation == 'above':
        return (pair['center_z_delta_m'] > 0.05 and aligned and
                pair['target_bottom_minus_anchor_top_m'] >= -tolerance)
    if relation == 'below':
        return (pair['center_z_delta_m'] < -0.05 and aligned and
                pair['anchor_bottom_minus_target_top_m'] >= -tolerance)
    return True


def _floor_reference(objects):
    bottoms = sorted(_box_metrics(obj)['bottom_z_m'] for obj in objects)
    if not bottoms:
        return 0.0
    # Use the median of the lowest fifth rather than one potentially corrupt box.
    count = max(1, ceil(len(bottoms) * 0.2))
    return float(np.median(bottoms[:count]))


def _size_ids(objects, category, phrase):
    text = phrase.casefold()
    pool = [o for o in objects if match_retrieval_categories([category], [o])]
    if not pool:
        return set(), None
    if 'smallest' in text:
        metric, choose = 'volume_m3', min
    elif 'largest' in text or 'biggest' in text:
        metric, choose = 'volume_m3', max
    elif 'tallest' in text or 'highest' in text:
        metric, choose = 'height_m', max
    elif 'shortest' in text or 'lowest' in text:
        metric, choose = 'height_m', min
    else:
        return set(), None
    values = {o['bbox_id']: _box_metrics(o)[metric] for o in pool}
    best = choose(values.values())
    return {oid for oid, value in values.items() if abs(value - best) <= 1e-6}, {
        'category': category, 'phrase': phrase, 'metric': metric,
        'candidate_ids': sorted(values), 'selected_ids': sorted(oid for oid, value in values.items()
                                                                 if abs(value - best) <= 1e-6),
        'proxy': 'AABB volume' if metric == 'volume_m3' else 'AABB height'}


def _candidate_metrics(objects, category, selected):
    """Expose comparable box measurements without turning adjectives into filters."""
    pool = match_retrieval_categories([category], objects)
    volume_order = sorted(pool, key=lambda obj: _box_metrics(obj)['volume_m3'])
    height_order = sorted(pool, key=lambda obj: _box_metrics(obj)['height_m'])
    volume_rank = {obj['bbox_id']: index + 1 for index, obj in enumerate(volume_order)}
    height_rank = {obj['bbox_id']: index + 1 for index, obj in enumerate(height_order)}
    records = []
    for obj in selected:
        metrics = _box_metrics(obj)
        records.append({'id': obj['bbox_id'], 'category': category,
                        **metrics, 'volume_rank': volume_rank.get(obj['bbox_id']),
                        'height_rank': height_rank.get(obj['bbox_id']),
                        'pool_size': len(pool)})
    return records


def diagnose(parsed, objects, candidates, room_center, checks=(), trusted_categories=False,
             apply_size_constraints=True):
    """Unknown category evidence stays eligible; predicates are ANDed per interpretation."""
    rewritten = parsed.get('query_rewrite') if isinstance(parsed, dict) else None
    if isinstance(rewritten, dict):
        from query_rewrite import constraints_as_relations
        normalized_relations = constraints_as_relations(rewritten)
        parsed = dict(parsed)
        if normalized_relations:
            parsed['Relations'] = normalized_relations
        attrs = list(parsed.get('Attributes') or [])
        target_names = parsed.get('Target') if isinstance(parsed.get('Target'), list) else []
        target_category = retrieval_category(target_names[0]) if len(target_names) == 1 else ''
        for constraint in rewritten.get('constraints', []):
            if constraint.get('type') in ('size_min', 'size_max', 'height_min', 'height_max',
                                          'bottom_z_near_floor'):
                phrase = constraint.get('phrase', '')
                if not any(a.get('phrase') == phrase for a in attrs if isinstance(a, dict)):
                    attrs.append({'category': target_category, 'phrase': phrase})
        parsed['Attributes'] = attrs
    target_names = parsed['Target'] if isinstance(parsed['Target'], list) else [parsed['Target']]
    names = {retrieval_category(n) for n in target_names if retrieval_category(n)}
    missing = list(parsed['diagnostic_missing'])
    if len(names) != 1:
        missing.append('requires one bound target category')
    category = next(iter(names)) if len(names) == 1 else ''
    checks_by_key = {(c['id'], c['category']): c for c in checks}
    by_id = {o['bbox_id']:o for o in objects}
    geometry = {
        'size_constraints': [], 'floor_constraint': None, 'relation_pairs': [],
        'between_triplets': [], 'proximity_pairs': [],
        'vertical_policy': {
            'horizontal_overlap_required': False,
            'horizontal_gap_max_m': HORIZONTAL_ALIGNMENT_GAP_M,
            'vertical_clearance_tolerance_m': VERTICAL_CLEARANCE_TOLERANCE_M,
            'requires_vertical_separation': True,
        },
        'size_policy': {
            'metric': 'AABB volume, footprint and height',
            'adjective_policy': 'small/large are soft evidence; only explicit superlatives filter',
            'shape_policy': 'XY extents and ratios are soft AABB proxies, not intrinsic object width/depth; '
                            'skinny/narrow is not a minimum-short-edge rule',
        },
    }
    def verdict(oid, name):
        if trusted_categories and not match_retrieval_categories([name], [by_id[oid]]):
            return 'no'
        checked = checks_by_key.get((oid, retrieval_category(name)))
        if checked:
            return checked.get('verdict', 'uncertain')
        if trusted_categories and not attributes_for(parsed, name):
            return 'yes'
        return 'uncertain'
    eligible = [o for o in candidates if verdict(o['bbox_id'], category) != 'no']
    # Explicit superlatives and "on the floor" are deterministic box predicates.
    size_allowed = None
    for attribute in parsed.get('Attributes', []):
        if retrieval_category(attribute.get('category', '')) != category:
            continue
        ids, record = _size_ids(objects, category, attribute.get('phrase', ''))
        if record:
            geometry['size_constraints'].append(record)
            size_allowed = ids if size_allowed is None else size_allowed & ids
        if re.search(r'\b(on the floor|on floor|on the ground|grounded)\b',
                     attribute.get('phrase', ''), re.I):
            floor_z = _floor_reference(objects)
            floor_ids = {o['bbox_id'] for o in objects
                         if _box_metrics(o)['bottom_z_m'] <= floor_z + 0.15}
            geometry['floor_constraint'] = {'floor_z_m': floor_z, 'tolerance_m': 0.15,
                                            'selected_ids': sorted(floor_ids)}
            size_allowed = floor_ids if size_allowed is None else size_allowed & floor_ids
    if size_allowed is not None and apply_size_constraints:
        eligible = [o for o in eligible if o['bbox_id'] in size_allowed]
    directions = any(r['relation'] in ('left', 'right', 'front', 'back') for r in parsed['Relations'])
    observer = parsed['Observer']
    frames, frame_missing = observer_frames(parsed, objects, room_center) if directions else ([None], [])
    if directions:
        frames = [frame for frame in frames if
                  (frame['origin_id'] is None or verdict(frame['origin_id'], observer['origin'].get('category', '')) != 'no')
                  and all(verdict(oid, observer['facing'].get('category', '')) != 'no' for oid in frame['facing_ids'])]
        if not frames and not frame_missing:
            frame_missing.append('observer categories contradicted by RGB evidence')
        missing.extend(frame_missing)
    # References sharing a category share an instance within each interpretation.
    reference_names = set()
    for relation in parsed['Relations']:
        if retrieval_category(relation.get('target_category', '')) != category:
            missing.append('relation target is not the bound target category')
            continue
        if relation['relation'] == 'near':
            missing.append('near has no calibrated predicate; distances are evidence only')
        if relation['relation'] == 'between':
            reference_names.update(retrieval_category(name)
                                   for name in relation.get('anchor_categories', []))
        elif relation['relation'] in ('nearest', 'farthest', 'near') or relation.get('scope') == 'relative_anchor':
            reference_names.add(retrieval_category(relation.get('anchor_category', '')))
    references = {}
    for name in sorted(reference_names):
        matches = [o for o in match_retrieval_categories([name], objects) if verdict(o['bbox_id'], name) != 'no']
        # A descriptor on a different reference category can be bound independently.
        # When target and reference share a category, the parser has not told us which
        # occurrence owns the descriptor, so retain all instances and record the ambiguity.
        if name != category:
            allowed = None
            for attribute in parsed.get('Attributes', []):
                if retrieval_category(attribute.get('category', '')) != name:
                    continue
                ids, record = _size_ids(objects, name, attribute.get('phrase', ''))
                if record:
                    geometry['size_constraints'].append({**record, 'role': 'reference'})
                    allowed = ids if allowed is None else allowed & ids
            if allowed is not None:
                matches = [o for o in matches if o['bbox_id'] in allowed]
        if not name or not matches:
            missing.append('missing reference category: ' + name)
        else:
            references[name] = matches
    geometry['candidate_metrics'] = _candidate_metrics(objects, category, eligible) if category else []
    geometry['reference_metrics'] = [
        metric for name, matches in references.items()
        for metric in _candidate_metrics(objects, name, matches)
    ]
    proximity_by_pair = {}
    for index, relation in enumerate(parsed['Relations']):
        if (relation['relation'] != 'near' or
                retrieval_category(relation.get('target_category', '')) != category):
            continue
        name = retrieval_category(relation.get('anchor_category', ''))
        for anchor in references.get(name, []):
            for pair in _proximity_records(eligible, anchor):
                pair.update({'phrase': relation.get('phrase', ''), 'anchor_category': name})
                geometry['proximity_pairs'].append(pair)
                proximity_by_pair[(index, pair['target_id'], pair['anchor_id'])] = pair
    anchor_names = parsed.get('Anchor', [])
    if isinstance(anchor_names, str):
        anchor_names = [anchor_names]
    represented = reference_names | {retrieval_category(c.get('category', '')) for c in observer.values()}
    if any(retrieval_category(name) not in represented for name in anchor_names):
        missing.append('parsed reference has no represented relation/observer role')
    if any(retrieval_category(a['category']) not in represented | {category} for a in parsed['Attributes']):
        missing.append('appearance attribute has no represented target/reference role')
    hypotheses = []
    choices = list(references)
    combinations = prod(len(references[n]) for n in choices) * max(len(frames), 1)
    if combinations > 256:
        missing.append('hypothesis budget exceeded (256); no truncated stability claim')
    interpretations = product(frames or [None], product(*(references[n] for n in choices))) if combinations <= 256 else []
    for frame, refs in interpretations:
        binding = dict(zip(choices, refs))
        if frame:
            origin_name = retrieval_category(observer['origin'].get('category', ''))
            facing_name = retrieval_category(observer['facing'].get('category', ''))
            if origin_name in binding and binding[origin_name]['bbox_id'] != frame['origin_id']:
                continue
            if (not observer['facing'].get('group') and facing_name in binding
                    and binding[facing_name]['bbox_id'] != frame['facing_ids'][0]):
                continue
        possible, measured = [], []
        for target in eligible:
            oid = target['bbox_id']
            passed = True
            for index, relation in enumerate(parsed['Relations']):
                if retrieval_category(relation.get('target_category', '')) != category:
                    continue
                kind = relation['relation']
                relation_anchor_names = (relation.get('anchor_categories', [])
                                         if kind == 'between' else
                                         [relation.get('anchor_category', '')])
                relation_anchors = [binding.get(retrieval_category(name))
                                    for name in relation_anchor_names]
                anchor = relation_anchors[0] if relation_anchors else None
                if anchor and anchor['bbox_id'] == oid:
                    passed = False
                    break
                if kind == 'between':
                    if len(relation_anchors) != 2 or any(item is None for item in relation_anchors):
                        passed = False
                        continue
                    if any(item['bbox_id'] == oid for item in relation_anchors):
                        passed = False
                        break
                    between = _between_geometry(target, relation_anchors[0], relation_anchors[1])
                    geometry['between_triplets'].append(between)
                    measured.append(between)
                    if between['passes'] is None:
                        missing.append(between['reason'])
                    else:
                        passed = passed and between['passes']
                elif kind == 'near' and anchor:
                    # Keep proximity as candidate-relative evidence only.  A
                    # scene-wide threshold is not calibrated, so this branch
                    # deliberately leaves ``passed`` unchanged and preserves
                    # the existing missing/uncertainty policy.
                    proximity = proximity_by_pair.get((index, oid, anchor['bbox_id']))
                    if proximity is not None:
                        measured.append(proximity)
                elif kind in ('nearest', 'farthest') and anchor:
                    distances = [(o['bbox_id'], float(np.linalg.norm(np.asarray(o['bbox_3d'][:3])-anchor['bbox_3d'][:3])))
                                 for o in eligible if o['bbox_id'] != anchor['bbox_id']]
                    best = (min if kind == 'nearest' else max)(d for _, d in distances)
                    distance = next(d for i, d in distances if i == oid)
                    passed = passed and abs(distance-best) <= 1e-6
                    measured.append({'target_id': oid, 'anchor_id': anchor['bbox_id'], 'relation': kind, 'distance_m': distance})
                elif kind in ('above', 'below') and anchor:
                    pair = _pair_geometry(target, anchor)
                    pair['relation'] = kind
                    pair['horizontal_alignment_passes'] = (
                        pair['horizontal_overlap_m2'] > 0.0 or
                        pair['horizontal_gap_m'] <= HORIZONTAL_ALIGNMENT_GAP_M)
                    if kind == 'above':
                        pair['vertical_separation_passes'] = (
                            pair['center_z_delta_m'] > 0.05 and
                            pair['target_bottom_minus_anchor_top_m'] >= -VERTICAL_CLEARANCE_TOLERANCE_M)
                    else:
                        pair['vertical_separation_passes'] = (
                            pair['center_z_delta_m'] < -0.05 and
                            pair['anchor_bottom_minus_target_top_m'] >= -VERTICAL_CLEARANCE_TOLERANCE_M)
                    pair['passes'] = _vertical_match(pair, kind)
                    geometry['relation_pairs'].append(pair)
                    measured.append(pair)
                    passed = passed and pair['passes']
                elif kind in ('left', 'right', 'front', 'back') and frame:
                    if relation.get('scope') == 'relative_anchor' and not anchor:
                        continue
                    base = anchor['bbox_3d'][:3] if relation.get('scope') == 'relative_anchor' else frame['origin']
                    delta = np.asarray(target['bbox_3d'][:3]) - base
                    axis = frame['right'] if kind in ('left', 'right') else frame['forward']
                    value = float(np.dot(delta, axis))
                    passed = passed and (value < -1e-6 if kind in ('left', 'back') else value > 1e-6)
                    measured.append({'target_id': oid, 'relation': kind, 'scope': relation['scope'], 'signed_m': value})
            if passed:
                possible.append(oid)
        hypotheses.append({'observer': frame, 'references': {n: o['bbox_id'] for n, o in binding.items()},
                           'candidate_ids': possible, 'measurements': measured})
    sets = [set(h['candidate_ids']) for h in hypotheses]
    possible = sorted(set.union(*sets)) if sets else []
    stable = sorted(set.intersection(*sets)) if sets else []
    between_by_id = {}
    for triplet in geometry['between_triplets']:
        entry = between_by_id.setdefault(triplet['target_id'], {
            'target_id': triplet['target_id'], 'passing_reference_pairs': [],
            'unknown_reference_pairs': [], 'failing_reference_pairs': []})
        key = ('unknown_reference_pairs' if triplet['passes'] is None else
               'passing_reference_pairs' if triplet['passes'] else 'failing_reference_pairs')
        if triplet['anchor_ids'] not in entry[key]:
            entry[key].append(triplet['anchor_ids'])
    geometry['between_candidate_evidence'] = [between_by_id[oid] for oid in sorted(between_by_id)]
    geometry['between_pass_ids'] = sorted(oid for oid, item in between_by_id.items() if item['passing_reference_pairs'])
    geometry['between_unknown_ids'] = sorted(oid for oid, item in between_by_id.items() if item['unknown_reference_pairs'])
    geometry['between_evidence_policy'] = ('Per-predicate geometry only, not votes or full-query validation. '
        'Unknown pairs preserve recall but do not provide passing evidence. '
        'An ID may have both passing and unknown reference pairs.')
    # VLM support covers category AND its supplied appearance attributes, not identity proof.
    needs = []
    def need(oid, name, role):
        name = retrieval_category(name)
        key = (oid, name)
        if verdict(oid, name) == 'uncertain' and not any((v['id'], v['category']) == key for v in needs):
            needs.append({'id': oid, 'category': name, 'attributes': attributes_for(parsed, name), 'role': role})
    for target in eligible:
        need(target['bbox_id'], category, 'target')
    for hypothesis in hypotheses:
        for name, oid in hypothesis['references'].items():
            need(oid, name, 'reference')
        frame = hypothesis['observer']
        if frame:
            observer = parsed['Observer']
            if frame['origin_id'] is not None:
                need(frame['origin_id'], observer['origin']['category'], 'observer_origin')
            for oid in frame['facing_ids']:
                need(oid, observer['facing']['category'], 'observer_facing')
    if needs:
        missing.append('category/appearance evidence is uncertain or unreviewed')
    missing = list(dict.fromkeys(missing))
    if not eligible or (sets and not possible and not missing):
        status = 'conflict'
    elif missing:
        status = 'missing'
    elif len(possible) != 1 or any(len(s) != 1 for s in sets):
        status = 'ambiguous'
    else:
        status = 'resolved'
    return {'status': status, 'missing': missing, 'candidate_ids': possible, 'stable_candidate_ids': stable,
            'hypotheses': hypotheses, 'visual_needs': needs, 'checks': list(checks),
            'geometry': geometry,
            'scope': 'bounded proposed instances, center-distance geometry, parser and VLM evidence; not linguistic uniqueness',
            'room_center_definition': 'center of point-cloud AABB, approximation only',
            'hypotheses_are_votes': False, 'category_source': 'annotated' if trusted_categories else 'predicted',
            'category_policy': 'annotated categories trusted; appearance still requires evidence' if trusted_categories
                               else 'category and appearance require visual support'}
