"""V7 source-bound box proxies and tentative, unweighted evidence selection.

No evaluation answers enter this module. Box/context preferences never delete
candidates or certify unresolved predicates; verified contradictions still veto.
"""
import math
import re

from evidence import match_retrieval_categories, retrieval_category
from object_geometry import box_measurements

COLORS = {'black', 'white', 'red', 'blue', 'green', 'grey', 'gray', 'brown',
          'yellow', 'orange', 'purple', 'pink'}


def recover_part_category(query, parsed):
    match = re.search(r'\b(?:section|part|piece)s?\s+of\s+(?:the\s+|a\s+)?([a-z]+)', query, re.I)
    if not match or len(parsed.get('Target', [])) != 1:
        return
    original = parsed['Target'][0]
    mother = match[1]
    if retrieval_category(original) not in ('section', 'part', 'piece', retrieval_category(mother)):
        return
    parsed['Target'] = [mother]
    if len(re.findall(r'\b' + re.escape(mother) + r's?\b', query, re.I)) == 1:
        parsed['Anchor'] = [a for a in parsed.get('Anchor', []) if retrieval_category(a) != retrieval_category(mother)]
    parsed['category_recovery'] = {'original': match[0].split()[0], 'canonical': retrieval_category(mother),
                                  'source_quote': match[0], 'reason': 'explicit part-of mother category'}
    for row in parsed.get('Attributes', []):
        if retrieval_category(row.get('category', '')) == retrieval_category(original):
            row['category'] = mother
    for row in parsed.get('Relations', []):
        if retrieval_category(row.get('target_category', '')) == retrieval_category(original):
            row['target_category'] = mother


def bind_negative_attributes(query, parsed, exclusions):
    from query_graph import polarity
    from appearance_evidence import attribute_units
    rows = []
    for attribute in parsed.get('Attributes', []):
        for phrase in attribute_units(attribute.get('phrase', '')):
            value = polarity(query, phrase, exclusions, attribute=True)
            if value == 'positive' or (value == 'exclude' and phrase.casefold() in COLORS):
                row = {**attribute, 'phrase': phrase, 'polarity': value}
                clause = next((c for c in exclusions if c['source_span'][0] <= query.casefold().find(phrase.casefold())
                               < c['source_span'][1]), None)
                if value == 'exclude':
                    if not clause or not re.fullmatch(r'(?:not|except)\s+(?:the\s+)?' +
                            re.escape(phrase) + r'(?:\s+one)?[.!?]?', clause['source_quote'], re.I):
                        continue  # qualified absence is not bare negative color
                if value == 'exclude' and clause:
                    row['negation_quote'] = clause['source_quote']
                    # Explicit postnominal "not the white one" corefers to a
                    # single named target, not a newly introduced reference.
                    if len(parsed.get('Target', [])) == 1 and re.fullmatch(
                            r'(?:not|except)\s+(?:the\s+)?' + re.escape(phrase) + r'\s+one[.!?]?',
                            clause['source_quote'], re.I):
                        row.update(category=parsed['Target'][0], role='target',
                                   binding_source='quoted contrastive one coreference')
                if row not in rows:
                    rows.append(row)
    # Recover only explicit bare color contrasts missed by the parser. More
    # complex negation/absence keeps its unresolved obligation.
    for match in re.finditer(r'\b(?:not|except)\s+(?:the\s+)?(' + '|'.join(sorted(COLORS)) + r')\s+one\b', query, re.I):
        if len(parsed.get('Target', [])) == 1 and not any(r['phrase'].casefold() == match[1].casefold()
                                                      and r['polarity'] == 'exclude' for r in rows):
            rows.append({'category': parsed['Target'][0], 'role': 'target', 'phrase': match[1],
                         'polarity': 'exclude', 'negation_quote': match[0],
                         'binding_source': 'quoted contrastive one coreference'})
    parsed['Attributes'] = rows
    covered = {r['negation_quote'] for r in rows if r.get('negation_quote')}
    parsed['_v7_executed_negations'] = sorted(covered)
    parsed['Unhandled'] = [p for p in parsed.get('Unhandled', []) if p.rstrip('.!?') not in
                           {c.rstrip('.!?') for c in covered}]
    if covered and all(c['source_quote'] in covered for c in exclusions):
        parsed['diagnostic_missing'] = [m for m in parsed.get('diagnostic_missing', [])
            if m != 'negation/exclusion is unsupported' and not any(m == 'unhandled condition: ' + c for c in covered)]


def box_size_spec(spec, category, query):
    spec = {**spec, 'measurement_policy': 'provided_instance_box',
            'metric_choice_source': 'v7 declared category/word semantics'}
    category = retrieval_category(category)
    if spec['word'] in {'tall', 'tallest', 'taller', 'short', 'shortest', 'shorter', 'high', 'highest', 'low', 'lowest'}:
        explicit_tabletop = re.search(r'\b(?:tabletop|table top|desktop|desk top)\b.*\b(?:floor|ground)\b', query, re.I)
        metric = 'tabletop_height_m' if explicit_tabletop and category in ('table', 'desk') else 'height_m'
        if metric == 'tabletop_height_m':
            spec.pop('measurement_policy')  # retain the existing local-floor executor
    elif spec['word'] == 'longer':
        metric = 'horizontal_long_extent_m'
    elif category in ('door', 'window'):
        metric = 'vertical_box_face_area_m2'
    elif category in ('whiteboard', 'blackboard', 'mirror'):
        metric = 'box_plane_area_m2'
    elif category in ('table', 'desk') and not re.search(r'\bvolume\b', query, re.I):
        metric = 'footprint_m2'
    elif category == 'book':
        # Single book versus stack is not inferable from a category label.
        # Preserve the observed volume; log alternative face area below.
        metric = 'volume_m3'
    else:
        metric = 'volume_m3'
    return {**spec, 'metric': metric, 'tentative_metric': True,
            'metric_reason': 'provided instance box ' + metric + '; observed geometry proxy, not completed physical object'}


def box_rows(objects, point_rows=()):
    point_by_id = {r['id']: r for r in point_rows}
    rows = box_measurements(objects)['measurements']
    for row in rows:
        dimensions = sorted((row['extent_x_m'], row['extent_y_m'], row['extent_z_m']))
        row.update(box_plane_area_m2=dimensions[1] * dimensions[2],
                   vertical_box_face_area_m2=row['horizontal_long_extent_m'] * row['height_m'],
                   metric_reliability={}, observed_completeness={},
                   box_quality='observed_bounds_may_be_incomplete_or_inflated',
                   point_geometry=point_by_id.get(row['id'], {}))
        if row['point_geometry']:
            row['observed_completeness'] = row['point_geometry'].get('observed_completeness', {})
            row['tabletop_height_m'] = row['point_geometry'].get('tabletop_height_m')
            row['tabletop_evidence'] = row['point_geometry'].get('tabletop_evidence', {})
            row['metric_reliability']['tabletop_height_m'] = row['point_geometry'].get('metric_reliability', {}).get('tabletop_height_m', {})
        if not all(math.isfinite(row[k]) and row[k] > 0 for k in ('extent_x_m', 'extent_y_m', 'extent_z_m')):
            row['measurement_reliability'] = 'unreliable'
    return rows


def evaluate_box_size(spec, measurements, comparison_pool_ids, *, scope_bound,
                      candidate_ids=None, scope_source='full_category', scope_unknown_ids=()):
    pool = sorted(set(comparison_pool_ids))
    candidate_ids = pool if candidate_ids is None else list(candidate_ids)
    by_id = {r['id']: r for r in measurements}
    metric = spec['metric']
    usable = {oid: by_id[oid][metric] for oid in pool if oid in by_id
              and by_id[oid].get('source') == 'annotated_box'
              and by_id[oid].get('measurement_reliability') == 'usable'
              and isinstance(by_id[oid].get(metric), (int, float))
              and math.isfinite(by_id[oid][metric]) and by_id[oid][metric] > 0}
    result = {**spec, 'status': 'unknown', 'comparison_pool_ids': pool,
              'scope_status': 'bound' if scope_bound and not scope_unknown_ids else 'unknown',
              'scope_source': scope_source, 'comparison_pool_frozen': bool(scope_bound),
              'scope_unknown_ids': sorted(scope_unknown_ids), 'selected_ids': [], 'tie_ids': [],
              'tentative_selected_ids': [], 'unknown_measurement_ids': sorted(set(pool) - usable.keys()),
              'measurement_sources': {str(oid): by_id[oid].get('source') for oid in pool if oid in by_id},
              'reason': 'provided box ordering is provisional; unresolved scope and scan risks retained'}
    required = spec.get('required_member_count')
    # Incomplete membership is different from unknown modifier scope. A wrong
    # explicit cardinality or missing measurement cannot establish an extreme.
    if len(pool) >= 2 and len(usable) == len(pool) and (required is None or required == len(pool)):
        ordered = sorted(usable, key=usable.get, reverse=spec['direction'] == 'max')
        best, second = usable[ordered[0]], usable[ordered[1]]
        tolerance = max(1e-9, 1e-6 * max(best, second))
        tied = sorted(oid for oid in pool if abs(usable[oid] - best) <= tolerance)
        result.update(tie_ids=tied, measured_gap=abs(best-second), relative_gap=abs(best-second)/max(best, second),
                      certification_margin=.20 if spec['form'] == 'adjective' else .10)
        if len(tied) == 1 and spec['interpretation'] != 'reference_comparison':
            result['tentative_selected_ids'] = tied
        else:
            result['reason'] = 'box values tied or joint reference comparison needs bound reference evidence'
    result['candidate_evidence'] = [{'candidate_id': oid, 'status': 'unknown',
        'tentative_status': 'pass' if oid in result['tentative_selected_ids'] else 'unknown',
        'reason': result['reason'], 'metric': metric, 'value': usable.get(oid),
        'measurement_reliability': by_id.get(oid, {}).get('measurement_reliability', 'unknown')}
        for oid in candidate_ids]
    result['risks'] = {str(oid): by_id[oid].get('observed_completeness', {}) for oid in usable}
    return result


def relation_preferences(query, parsed, objects, candidates, checks):
    from v4_relations import relation_tasks, geometric_relation, pair_facts
    from diagnostic import _attribute_comparison_pool
    groups = []
    for task in relation_tasks(query, parsed):
        pool = match_retrieval_categories([task['target_category']], candidates)
        anchors = match_retrieval_categories([task.get('anchor_category', '')], objects)
        if task['kind'] != 'corner' and anchors:
            original = anchors
            anchors, binding = _attribute_comparison_pool(parsed, anchors, task['anchor_category'], checks, 'reference')
            if binding and binding.get('status') != 'bound':
                excluded = {p['candidate_id'] for p in binding.get('candidate_removal_audit', []) if p.get('removal_allowed')}
                anchors = [a for a in original if a['bbox_id'] not in excluded]
        facts, hypotheses = [], []
        if task['kind'] == 'near':
            for anchor in anchors:
                distances = [(o['bbox_id'], pair_facts(o, anchor)['aabb_gap_m']) for o in pool
                             if o['bbox_id'] != anchor['bbox_id']]
                best = min((d for _, d in distances), default=None)
                # Contextual rank is a preference, never a proof of near. A
                # distant nearest object is not even a useful near preference.
                plausible = best is not None and best <= max(1., max(anchor['bbox_3d'][3:]))
                ids = sorted(oid for oid, d in distances if plausible and abs(d-best) <= .01)
                hypotheses.append({'anchor_id': anchor['bbox_id'], 'preferred_ids': ids,
                                   'distances_m': dict(distances), 'plausible_band_m': max(1., max(anchor['bbox_3d'][3:]))})
            sets = [set(h['preferred_ids']) for h in hypotheses]
            preferred = sorted(set.intersection(*sets)) if sets and all(sets) else []
        else:
            for obj in pool:
                alternatives = [geometric_relation(task, obj, anchors)] if task['kind'] == 'corner' else [
                    geometric_relation({**task, 'binding_ambiguous': False}, obj, [a]) for a in anchors]
                states = [a['status'] for a in alternatives]
                facts.append({'candidate_id': obj['bbox_id'], 'alternatives': alternatives})
            preferred = [f['candidate_id'] for f in facts if f['alternatives'] and
                         all(a['status'] == 'pass' for a in f['alternatives'])]
        groups.append({'source_quote': task['source_quote'], 'kind': task['kind'],
                       'preferred_ids': preferred, 'anchor_hypotheses': hypotheses,
                       'geometry_evidence': facts, 'status': 'tentative',
                       'reason': 'context/box support; no hard exclusion or language threshold certification'})
    return groups


def supported_choice(query, parsed, objects, candidates, diagnosis, checks=()):
    from diagnostic import size_conditions, appearance_attributes_for
    from appearance_evidence import aggregate_visual_checks, required_visual_attributes
    live = set(diagnosis.get('candidate_ids', []))
    pool = [o for o in candidates if o['bbox_id'] in live]
    contradicted = {r['candidate_id'] for r in diagnosis.get('candidate_removal_audit', [])
                    if r.get('removal_allowed') and r.get('status') == 'fail' and r.get('target_role') == 'target'}
    supports = {oid: set() for oid in live}
    point_rows = [p for p in parsed.get('_v7_measurement_rows', []) if p.get('point_geometry')]
    rows = box_rows(candidates, [p['point_geometry'] for p in point_rows])
    relations = relation_preferences(query, parsed, objects, pool, checks)
    for relation in relations:
        for oid in relation['preferred_ids']:
            if oid in supports:
                supports[oid].add('relation:' + relation['source_quote'])
    constraints = []
    invalidated = set(parsed.get('_v4_invalidated_conditions', []))
    for category in parsed.get('Target', []):
        category_pool = match_retrieval_categories([category], candidates)
        expected = required_visual_attributes(appearance_attributes_for(parsed, category))
        for obj in pool:
            relevant = [c for c in checks if c.get('id') == obj['bbox_id']
                        and retrieval_category(c.get('category', '')) == retrieval_category(category)
                        and c.get('role', 'target') == 'target']
            visual = aggregate_visual_checks(relevant, expected)
            if visual:
                if visual['category_verdict'] == 'no':
                    contradicted.add(obj['bbox_id'])
                for attr in visual['attribute_checks']:
                    if attr['verdict'] == 'no':
                        contradicted.add(obj['bbox_id'])
                    elif attr['verdict'] == 'yes':
                        supports[obj['bbox_id']].add('appearance:' + attr['attribute'])
        for spec in size_conditions(parsed, category):
            if spec['original_quote'] in invalidated or spec.get('measurement_policy') != 'provided_instance_box':
                continue
            scoped = category_pool
            source = 'initial category box preview; modifier scope unresolved'
            # A restrictive corner/support modifier may define the comparison
            # subset. Geometry remains tentative, and its origin is recorded.
            scope_quotes = spec.get('scope_relation_quotes', [])
            if spec.get('scope_kind') in ('relation_subset', 'unbound_explicit_scope'):
                matching = [r for r in relations if r['kind'] in ('corner', 'support') and
                            (not scope_quotes or any(q.casefold() in r['source_quote'].casefold() or
                             r['source_quote'].casefold() in q.casefold() for q in scope_quotes))]
                if matching and all(r['preferred_ids'] for r in matching):
                    ids = set.intersection(*(set(r['preferred_ids']) for r in matching))
                    scoped = [o for o in scoped if o['bbox_id'] in ids]
                    source = 'tentative geometry modifier subset'
            constraint = evaluate_box_size(spec, rows, [o['bbox_id'] for o in scoped],
                scope_bound=False, candidate_ids=live, scope_source=source)
            constraints.append(constraint)
            for oid in constraint['tentative_selected_ids']:
                if oid in supports:
                    supports[oid].add('size:' + spec['original_quote'])
    eligible = {oid: proof for oid, proof in supports.items() if oid not in contradicted and proof}
    # Partial ordering of sets avoids arbitrary weights or repeated votes.
    # Incomparable support (e.g. color vs size disagree) requires more evidence.
    maxima = [oid for oid, proof in eligible.items() if not any(proof < other for other in eligible.values())]
    selected = maxima[0] if len(maxima) == 1 else None
    return {'selected_id': selected, 'status': 'tentative' if selected is not None else 'abstain',
            'candidate_support': {str(oid): sorted(proof) for oid, proof in sorted(supports.items())},
            'contradicted_ids': sorted(contradicted), 'size_constraints': constraints, 'relations': relations,
            'unresolved_conditions': list(diagnosis.get('missing', [])),
            'reason': 'unique undominated supported candidate' if selected is not None else
                      'no evidence distinguishes a unique uncontradicted candidate'}


def fitted_corner(target, walls):
    """Two nonparallel finite observed wall planes, without a room AABB proxy."""
    import numpy as np
    planes = [w for w in walls if w.get('wall_geometry', {}).get('valid')]
    pairs = []
    for index, first in enumerate(planes):
        a = first['wall_geometry']
        na, oa = np.asarray(a['normal_xy']), np.asarray(a['origin_xy_m'])
        for second in planes[index+1:]:
            b = second['wall_geometry']
            nb, ob = np.asarray(b['normal_xy']), np.asarray(b['origin_xy_m'])
            if abs(float(na @ nb)) > math.cos(math.radians(30)):
                continue  # parallel or nearly parallel walls do not form a corner
            intersection = np.linalg.solve(np.array([na, nb]), np.array([na @ oa, nb @ ob]))
            if any(not wall['tangent_extent_m'][0] - .15 <= float(
                    (intersection - np.asarray(wall['origin_xy_m'])) @ np.asarray(wall['tangent_xy']))
                    <= wall['tangent_extent_m'][1] + .15 for wall in (a, b)):
                continue  # distant infinite-plane intersection is not a room corner
            box = target['bbox_3d']
            gaps = []
            for wall in (a, b):
                normal = np.asarray(wall['normal_xy'])
                radius = float(np.abs(normal) @ np.asarray(box[3:5]) / 2)
                gaps.append(max(0., abs(float((np.asarray(box[:2])-np.asarray(wall['origin_xy_m'])) @ normal)) - radius))
            pairs.append({'anchor_ids': [first['bbox_id'], second['bbox_id']],
                          'gaps_m': gaps, 'intersection_xy_m': intersection.tolist()})
    near = [p for p in pairs if max(p['gaps_m']) <= .30]
    return {'status': 'pass' if near else 'unknown',
            'anchor_ids': sorted({oid for p in near for oid in p['anchor_ids']}),
            'corner_pairs': pairs, 'source': 'finite fitted wall planes',
            'reason': 'two intersecting wall planes support corner proximity; provisional without visual confirmation'}
