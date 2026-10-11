"""Quoted relation tasks shared by subset binding and position/distance evidence."""
import re
import numpy as np
from evidence import match_retrieval_categories, retrieval_category
from size_semantics import category_pattern


def relation_tasks(query, parsed):
    tasks = []
    for relation in parsed.get('Relations', []):
        if relation.get('relation') == 'near':
            tasks.append({**relation, 'kind': 'near', 'source_quote': relation['phrase']})
    phrases = list(parsed.get('Unhandled', []))
    phrases += [m['text'] for c in parsed.get('query_syntax', {}).get('comparisons', [])
                if c.get('attachment') == 'restrictive_subset' for m in c.get('modifiers', [])]
    phrases += [s[len('unsupported relation: '):] for s in parsed.get('diagnostic_missing', [])
                if s.startswith('unsupported relation: ')]
    for phrase in dict.fromkeys(phrases):
        if not phrase or phrase.casefold() not in query.casefold() or re.search(r'\bnot\b', phrase, re.I):
            continue
        kind = ('corner' if re.search(r'\bcorner\b', phrase, re.I) else
                'attachment' if re.search(r'\b(?:against|attached|hanging)\b', phrase, re.I) else
                'support' if re.search(r'\b(?:on|on top of)\b', phrase, re.I) else
                'near' if re.search(r'\bnear\b', phrase, re.I) else None)
        if not kind:
            continue
        if kind == 'support' and not re.match(r'^(?:(?:that|which)\s+(?:is|are)\s+)?(?:on\b|(?:sitting|resting|standing)\s+on\b)', phrase, re.I):
            continue  # "with a plant on top" puts the reference on the target, not vice versa.
        anchors = [name for name in parsed.get('Anchor', [])
                   if re.search(r'\b' + category_pattern(name) + r'\b', phrase, re.I)]
        if kind == 'corner' and anchors and not any(n in ('wall', 'room') for n in anchors):
            continue  # A desk corner is not a room corner.
        if kind == 'corner':
            anchors = ['wall']
        phrase = re.sub(r'^(?:that|which)\s+(?:is|are)\s+', '', phrase, flags=re.I)
        tasks.append({'kind': kind, 'source_quote': phrase,
                      'target_category': (parsed.get('Target') or [''])[0],
                      'anchor_category': anchors[0] if len(anchors) == 1 else '',
                      'binding_ambiguous': len(anchors) != 1 and kind != 'corner'})
    return list({(t['source_quote'], t['target_category']): t for t in tasks}.values())


def pair_facts(target, anchor):
    a, b = (np.asarray(o['bbox_3d'], float) for o in (target, anchor))
    gap = np.maximum(np.abs(a[:3] - b[:3]) - (a[3:] + b[3:]) / 2, 0)
    overlap = np.maximum(np.minimum(a[:3] + a[3:] / 2, b[:3] + b[3:] / 2) -
                         np.maximum(a[:3] - a[3:] / 2, b[:3] - b[3:] / 2), 0)
    return {'aabb_gap_m': float(np.linalg.norm(gap)),
            'horizontal_gap_m': float(np.linalg.norm(gap[:2])),
            'bottom_minus_top_m': float(a[2] - a[5] / 2 - b[2] - b[5] / 2),
            'overlap_fraction': float(np.prod(overlap[:2]) / max(np.prod(a[3:5]), 1e-9)),
            'anchor_horizontal_extent_m': float(max(b[3:5]))}


def geometric_relation(task, target, anchors):
    if task.get('binding_ambiguous') or not anchors:
        return {'status': 'unknown', 'reason': 'quoted reference is not uniquely bound'}
    if task['kind'] == 'corner':
        # A global scene bounding rectangle is not a room-corner annotation.
        walls = [a for a in anchors if min(a['bbox_3d'][3:5]) < .25]
        pairs = []
        for i, first in enumerate(walls):
            for second in walls[i + 1:]:
                if np.argmin(first['bbox_3d'][3:5]) == np.argmin(second['bbox_3d'][3:5]):
                    continue
                if pair_facts(first, second)['horizontal_gap_m'] > .15:
                    continue
                gaps = [pair_facts(target, wall)['horizontal_gap_m'] for wall in (first, second)]
                pairs.append({'anchor_ids': [first['bbox_id'], second['bbox_id']], 'gaps_m': gaps})
        near = [p for p in pairs if max(p['gaps_m']) <= .30]
        if len(near) == 1:
            return {'status': 'pass', 'anchor_ids': near[0]['anchor_ids'], 'corner_pairs': pairs,
                    'reason': 'two intersecting orthogonal supplied wall boxes within 0.30m; visual confirmation required'}
        return {'status': 'unknown', 'corner_pairs': pairs,
                'reason': 'two nearby intersecting wall instances are not uniquely established'}
    if len(anchors) != 1 or target['bbox_id'] == anchors[0]['bbox_id']:
        return {'status': 'unknown', 'reason': 'reference instance is not uniquely and distinctly bound'}
    facts = pair_facts(target, anchors[0])
    kind = task['kind']
    if kind == 'near':
        # Declared geometric support band, not nearest, not a universal language calibration.
        close = min(.40, max(.15, .25 * facts['anchor_horizontal_extent_m']))
        far = max(1., 2 * close)
        gap = facts['horizontal_gap_m']
        state = 'pass' if gap <= close else 'fail' if gap > far else 'unknown'
        facts.update(near_support_band_m=close, far_contradiction_band_m=far)
    elif kind == 'support':
        delta = facts['bottom_minus_top_m']
        state = ('pass' if abs(delta) <= .05 and facts['overlap_fraction'] >= .20 else
                 'fail' if abs(delta) > .20 or facts['overlap_fraction'] == 0 else 'unknown')
    else:
        gap = facts['aabb_gap_m']
        state = 'pass' if gap <= .04 else 'fail' if gap > .20 else 'unknown'
    return {'status': state, 'anchor_ids': [anchors[0]['bbox_id']], 'geometry': facts,
            'reason': 'observed box evidence; visible relation confirmation required'}


def bind_tasks(query, parsed, objects, candidates, checks=()):
    from diagnostic import _attribute_comparison_pool
    rows = []
    for task in relation_tasks(query, parsed):
        anchors = match_retrieval_categories([task['anchor_category']], objects) if task['anchor_category'] else []
        if task['kind'] != 'corner':
            anchors, scope = _attribute_comparison_pool(parsed, anchors, task['anchor_category'], checks, 'reference')
            if scope and scope.get('status') != 'bound':
                anchors = []
        for obj in candidates:
            if retrieval_category(obj['target']) != retrieval_category(task['target_category']):
                continue
            rows.append({**task, 'candidate_id': obj['bbox_id'],
                         'geometry_evidence': geometric_relation(task, obj, anchors),
                         'status': 'unknown'})
    return rows


def verified_members(parsed, quote, category):
    rows = [r for r in parsed.get('_v4_relation_evidence', [])
            if r['source_quote'].casefold() == quote.casefold() and
            retrieval_category(r['target_category']) == retrieval_category(category)]
    return {r['candidate_id']: r for r in rows}
