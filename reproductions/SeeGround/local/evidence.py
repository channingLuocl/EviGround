"""Query matching and measured geometry; no ground-truth labels are used."""
import re
import numpy as np

ALIASES = {
    'television': 'tv', 'televisions': 'tv', 'screen': 'tv', 'screens': 'tv',
    'monitor': 'tv', 'monitors': 'tv',
    'trashcan': 'trash can', 'trashcans': 'trash can', 'trash cans': 'trash can',
    'garbage bin': 'trash can', 'garbage can': 'trash can',
    'fridge': 'refrigerator', 'refrigerators': 'refrigerator',
    'front doors': 'door', 'front door': 'door', 'double door': 'door', 'doors': 'door',
    'windows': 'window', 'chairs': 'chair', 'tables': 'table', 'cabinets': 'cabinet',
    'blackboards': 'blackboard',
    'countertop': 'counter', 'countertops': 'counter',
}


def canonical_category(name):
    text = ' '.join(re.findall(r'[a-z]+', name.lower()))
    for source in sorted(ALIASES, key=len, reverse=True):
        text = re.sub(r'\b' + re.escape(source) + r'\b', ALIASES[source], text)
    return text


def match_categories(names, objects):
    if isinstance(names, str):
        names = [names]
    categories = [canonical_category(n).split() for n in names if n.strip()]
    matched = []
    for obj in objects:
        words = canonical_category(obj['target']).split()
        if any(category and all(w in words for w in category) for category in categories):
            matched.append(obj)
    return matched


def distance_evidence(targets, anchors):
    pairs = []
    for target in targets:
        for anchor in anchors:
            if target['bbox_id'] == anchor['bbox_id']:
                continue
            a, b = np.asarray(target['bbox_3d']), np.asarray(anchor['bbox_3d'])
            gap = np.maximum(np.abs(a[:3] - b[:3]) - (a[3:] + b[3:]) / 2, 0)
            pairs.append({'target_id': target['bbox_id'], 'anchor_id': anchor['bbox_id'],
                          'center_distance_m': round(float(np.linalg.norm(a[:3] - b[:3])), 4),
                          'aabb_gap_m': round(float(np.linalg.norm(gap)), 4)})
    return {'units': 'meters', 'world_axes': 'X/Y horizontal, Z up; these are not camera axes',
            'distance_definition': 'Euclidean center distance; AABB gap is a separate box approximation',
            'pairs': pairs}


def retrieval_category(name):
    """Remove common appearance modifiers only for category retrieval."""
    text = canonical_category(name)
    if text in {'wall', 'floor', 'ground', 'ceiling'}:
        return text
    modifiers = {'large', 'small', 'big', 'little', 'black', 'white', 'red', 'blue',
                 'green', 'gray', 'grey', 'brown', 'wooden', 'wall', 'mounted'}
    return ' '.join(word for word in text.split() if word not in modifiers)


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


def distance_rankings(query, parsed, targets, anchors):
    """Rank only explicit quoted superlatives, separately for each anchor."""
    rankings, ignored = [], list(parsed.get('rejected_distance_relations', []))
    relations = parsed.get('Relations', [])
    if not isinstance(relations, list):
        return {'rankings': [], 'ignored': [{'reason': 'Relations must be a list'}]}
    for relation in relations:
        if not isinstance(relation, dict):
            ignored.append({'reason': 'invalid relation object'})
            continue
        kind = relation.get('relation')
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
        selected_targets = match_retrieval_categories([target_name], targets)
        selected_anchors = match_retrieval_categories([anchor_name], anchors)
        if not selected_targets or not selected_anchors:
            ignored.append({'relation': relation, 'reason': 'unmatched target/reference category'})
            continue
        for anchor in selected_anchors:
            distances = [{'target_id': target['bbox_id'],
                          'distance_m': float(np.linalg.norm(np.asarray(target['bbox_3d'][:3]) - anchor['bbox_3d'][:3]))}
                         for target in selected_targets if target['bbox_id'] != anchor['bbox_id']]
            distances.sort(key=lambda value: value['distance_m'], reverse=kind == 'farthest')
            if not distances:
                continue
            best = distances[0]['distance_m']
            tied = [value['target_id'] for value in distances if abs(value['distance_m'] - best) <= 1e-6]
            rankings.append({'relation': kind, 'phrase': phrase, 'anchor_id': anchor['bbox_id'],
                             'ordered_targets': distances, 'best_ids': tied,
                             'metric': 'Euclidean center distance in meters',
                             'scope': 'predicted-category candidates only; reference not disambiguated'})
    return {'rankings': rankings, 'ignored': ignored,
            'policy': 'soft evidence; no hard filtering, no cross-anchor averaging'}


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
