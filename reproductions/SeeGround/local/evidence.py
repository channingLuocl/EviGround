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
