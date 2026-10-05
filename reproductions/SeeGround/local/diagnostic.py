"""Scoped constraint diagnosis, with shared observer/reference hypotheses, never GT."""
from itertools import product
from math import prod
import re
import numpy as np
from evidence import match_retrieval_categories, retrieval_category


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
Use other+Unhandled for extremes, object-intrinsic axes, between, ordinal or unsupported relations.
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


def validate_structure(query, parsed):
    parsed = dict(parsed)
    missing = []
    relations = parsed.get('Relations', [])
    valid = []
    for relation in relations:
        kind = relation.get('relation')
        phrase = relation.get('phrase')
        valid_quote = quoted(query, phrase)
        target_name, anchor_name = relation.get('target_category'), relation.get('anchor_category', '')
        valid_quote = valid_quote and isinstance(target_name, str) and bool(target_name.strip())
        if kind in ('nearest', 'farthest', 'near') or relation.get('scope') == 'relative_anchor':
            valid_quote = valid_quote and isinstance(anchor_name, str) and bool(anchor_name.strip())
        if kind in ('left', 'right', 'front', 'back'):
            words = {'left': r'\bleft\b', 'right': r'\bright\b',
                     'front': r'\b(front|ahead)\b', 'back': r'\b(back|behind)\b'}
            valid_quote = valid_quote and bool(re.search(words[kind], phrase, re.I))
            valid_quote = valid_quote and relation.get('scope') in ('relative_anchor', 'side')
        if not valid_quote or kind not in ('nearest', 'farthest', 'near', 'left', 'right', 'front', 'back'):
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
    parsed['Attributes'] = [a for a in attrs if quoted(query, a.get('phrase')) and isinstance(a.get('category'), str)]
    if len(parsed['Attributes']) != len(attrs):
        missing.append('unquoted or unbound appearance attribute')
    covered = ' '.join(a['phrase'] for a in parsed['Attributes'])
    for word in re.findall(r'\b(large|small|big|little|black|white|red|blue|green|gr[ae]y|brown|wooden|mounted)\b', query, re.I):
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
        missing.extend('unhandled condition: ' + p for p in parsed['Unhandled'])
    targets = parsed['Target'] if isinstance(parsed['Target'], list) else [parsed['Target']]
    if any(retrieval_category(name) in ('one','correct one','it','object','thing') for name in targets):
        missing.append('target category is an unresolved pronoun or generic noun')
    if re.search(r"\b(no|not|never|without|except)\b|n['’]t\b", query, re.I):
        missing.append('negation/exclusion is unsupported')
        parsed['Relations'] = []
    if re.search(r'\b(left|right|ahead|behind)\b', query, re.I) and not any(
            r['relation'] in ('left', 'right', 'front', 'back') for r in parsed['Relations']):
        missing.append('direction condition not represented')
    for pattern, kinds, label in ((r'\b(nearest|closest|farthest|furthest)\b', ('nearest','farthest'), 'distance extreme'),
                                  (r'\b(next to|beside|near)\b', ('near',), 'proximity')):
        if re.search(pattern, query, re.I) and not any(r['relation'] in kinds for r in parsed['Relations']):
            missing.append(label + ' condition not represented')
    if re.search(r'\b(corner|between|leftmost|rightmost|second|third|ground|floor)\b', query, re.I):
        missing.append('corner/ordinal/support/extreme conditions require additional predicates')
    if re.search(r'\b(smallest|largest|tallest|shortest|lowest|highest|lower|upper|tall|short|above|below|under|beneath)\b',query,re.I):
        missing.append('size/vertical conditions have box-proxy evidence only; no certified predicate')
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


def diagnose(parsed, objects, candidates, room_center, checks=(), trusted_categories=False):
    """Unknown category evidence stays eligible; predicates are ANDed per interpretation."""
    target_names = parsed['Target'] if isinstance(parsed['Target'], list) else [parsed['Target']]
    names = {retrieval_category(n) for n in target_names if retrieval_category(n)}
    missing = list(parsed['diagnostic_missing'])
    if len(names) != 1:
        missing.append('requires one bound target category')
    category = next(iter(names)) if len(names) == 1 else ''
    checks_by_key = {(c['id'], c['category']): c for c in checks}
    by_id = {o['bbox_id']:o for o in objects}
    def verdict(oid, name):
        if trusted_categories:
            if not match_retrieval_categories([name], [by_id[oid]]):
                return 'no'
            if not attributes_for(parsed, name):
                return 'yes'
        return checks_by_key.get((oid, retrieval_category(name)), {}).get('verdict', 'uncertain')
    eligible = [o for o in candidates if verdict(o['bbox_id'], category) != 'no']
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
        if relation['relation'] in ('nearest', 'farthest', 'near') or relation.get('scope') == 'relative_anchor':
            reference_names.add(retrieval_category(relation.get('anchor_category', '')))
    references = {}
    for name in sorted(reference_names):
        matches = [o for o in match_retrieval_categories([name], objects) if verdict(o['bbox_id'], name) != 'no']
        if not name or not matches:
            missing.append('missing reference category: ' + name)
        else:
            references[name] = matches
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
            for relation in parsed['Relations']:
                if retrieval_category(relation.get('target_category', '')) != category:
                    continue
                kind = relation['relation']
                anchor = binding.get(retrieval_category(relation.get('anchor_category', '')))
                if anchor and anchor['bbox_id'] == oid:
                    passed = False
                    break
                if kind in ('nearest', 'farthest') and anchor:
                    distances = [(o['bbox_id'], float(np.linalg.norm(np.asarray(o['bbox_3d'][:3])-anchor['bbox_3d'][:3])))
                                 for o in eligible if o['bbox_id'] != anchor['bbox_id']]
                    best = (min if kind == 'nearest' else max)(d for _, d in distances)
                    distance = next(d for i, d in distances if i == oid)
                    passed = passed and abs(distance-best) <= 1e-6
                    measured.append({'target_id': oid, 'anchor_id': anchor['bbox_id'], 'relation': kind, 'distance_m': distance})
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
            'scope': 'bounded proposed instances, center-distance geometry, parser and VLM evidence; not linguistic uniqueness',
            'room_center_definition': 'center of point-cloud AABB, approximation only',
            'hypotheses_are_votes': False, 'category_source': 'annotated' if trusted_categories else 'predicted',
            'category_policy': 'annotated categories trusted; appearance still requires evidence' if trusted_categories
                               else 'category and appearance require visual support'}
