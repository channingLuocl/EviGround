"""Deterministic, auditable normalization of a structured NR3D query."""
import re

from evidence import canonical_category, retrieval_category


RELATION_TYPES = {
    'nearest': 'distance_min',
    'farthest': 'distance_max',
    'near': 'near',
    'left': 'left_of',
    'right': 'right_of',
    'front': 'front_of',
    'back': 'back_of',
    'above': 'above',
    'below': 'below',
}
VERTICAL_WORDS = {'over': 'above', 'above': 'above', 'below': 'below',
                  'under': 'below', 'beneath': 'below'}


def _category(value):
    if not isinstance(value, str) or not value.strip():
        return ''
    return retrieval_category(canonical_category(value)).strip()


def _quote(query, phrase):
    return isinstance(phrase, str) and bool(phrase.strip()) and phrase.casefold() in query.casefold()


def _attribute_constraint(attribute, query, target_category):
    phrase = attribute.get('phrase', '') if isinstance(attribute, dict) else ''
    category = _category(attribute.get('category', '')) if isinstance(attribute, dict) else ''
    if not _quote(query, phrase) or not category:
        return None
    text = phrase.casefold()
    if category != target_category:
        return None
    for words, kind, metric in (
        (('smallest',), 'size_min', 'volume_m3'),
        (('largest',), 'size_max', 'volume_m3'),
        (('tallest', 'highest'), 'height_max', 'height_m'),
        (('shortest', 'lowest'), 'height_min', 'height_m'),
    ):
        if any(re.search(r'\b' + re.escape(word) + r'\b', text) for word in words):
            return {'type': kind, 'target': 'target', 'reference': None,
                    'metric': metric, 'phrase': phrase, 'confidence': 'lexical'}
    if re.search(r'\b(on the floor|on floor|on the ground|grounded)\b', text):
        return {'type': 'bottom_z_near_floor', 'target': 'target', 'reference': None,
                'metric': 'bottom_z_m', 'phrase': phrase, 'confidence': 'lexical'}
    # Keep ordinary color/shape attributes auditable without pretending they are geometry.
    return {'type': 'attribute', 'target': 'target', 'reference': None,
            'value': phrase, 'phrase': phrase, 'confidence': 'lexical'}


def rewrite_query(query, parsed):
    """Compile parsed fields into stable constraints without choosing an object ID."""
    if not isinstance(query, str) or not isinstance(parsed, dict):
        raise ValueError('query and parsed query are required')
    raw_targets = parsed.get('Target', [])
    raw_targets = [raw_targets] if isinstance(raw_targets, str) else raw_targets
    targets = [{'category': _category(value), 'role': 'target', 'phrase': value}
               for value in raw_targets if _category(value)]
    target_category = targets[0]['category'] if len(targets) == 1 else ''

    raw_references = parsed.get('Anchor', [])
    raw_references = [raw_references] if isinstance(raw_references, str) else raw_references
    references = []

    def add_reference(value, role='anchor', phrase=''):
        category = _category(value)
        if not category:
            return
        key = (category, role)
        if not any((item['category'], item['role']) == key for item in references):
            references.append({'category': category, 'role': role, 'phrase': phrase or value})

    for value in raw_references or []:
        add_reference(value)
    observer = parsed.get('Observer') if isinstance(parsed.get('Observer'), dict) else {}
    for role, entry in (('observer_origin', observer.get('origin', {})),
                        ('observer_facing', observer.get('facing', {}))):
        if isinstance(entry, dict) and entry.get('category'):
            add_reference(entry['category'], role, entry.get('phrase', ''))

    constraints = []
    unresolved = []
    for relation in parsed.get('Relations', []) or []:
        if not isinstance(relation, dict):
            unresolved.append({'phrase': '', 'reason': 'invalid relation object'})
            continue
        relation_type = relation.get('relation')
        phrase = relation.get('phrase', '')
        target = _category(relation.get('target_category', ''))
        reference = _category(relation.get('anchor_category', ''))
        if relation_type not in RELATION_TYPES or not target or not reference or not _quote(query, phrase):
            unresolved.append({'phrase': phrase, 'reason': 'relation is incomplete or unquoted'})
            continue
        add_reference(relation.get('anchor_category', ''), 'anchor', phrase)
        constraints.append({'type': RELATION_TYPES[relation_type], 'target': target,
                            'reference': reference, 'scope': relation.get('scope', ''),
                            'phrase': phrase, 'confidence': 'lexical'})

    for attribute in parsed.get('Attributes', []) or []:
        compiled = _attribute_constraint(attribute, query, target_category)
        if compiled:
            constraints.append(compiled)
        elif isinstance(attribute, dict) and attribute.get('phrase'):
            unresolved.append({'phrase': attribute['phrase'], 'reason': 'attribute category is not uniquely bound'})

    for phrase in parsed.get('Unhandled', []) or []:
        if _quote(query, phrase):
            unresolved.append({'phrase': phrase, 'reason': 'parser marked condition unsupported'})

    # Keep lexical conditions that are commonly omitted from Attributes visible to the audit.
    if re.search(r'\b(on the floor|on floor|on the ground|grounded)\b', query, re.I) \
            and not any(c['type'] == 'bottom_z_near_floor' for c in constraints):
        phrase = re.search(r'\b(on the floor|on floor|on the ground|grounded)\b', query, re.I).group()
        constraints.append({'type': 'bottom_z_near_floor', 'target': 'target', 'reference': None,
                            'metric': 'bottom_z_m', 'phrase': phrase, 'confidence': 'lexical_recovered'})
    return {
        'query': query,
        'target': targets,
        'references': references,
        'constraints': constraints,
        'unresolved': unresolved,
        'policy': 'deterministic lexical normalization; no target ID selection',
    }


def constraints_as_relations(rewritten):
    """Convert relation constraints to the existing Diagnostic relation schema."""
    result = []
    for constraint in rewritten.get('constraints', []) if isinstance(rewritten, dict) else []:
        relation = next((name for name, kind in RELATION_TYPES.items()
                         if kind == constraint.get('type')), None)
        if relation is None or not constraint.get('reference'):
            continue
        result.append({'target_category': constraint.get('target', ''),
                       'anchor_category': constraint.get('reference', ''),
                       'relation': relation, 'scope': constraint.get('scope', ''),
                       'phrase': constraint.get('phrase', '')})
    return result
