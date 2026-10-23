"""Scoped constraint diagnosis, with shared observer/reference hypotheses, never GT."""
from itertools import product
from hashlib import sha256
from math import prod
import json
from copy import deepcopy
import re
from math import ceil
import numpy as np
from seeground.grounding.evidence import (canonical_category, distance_rankings,
                      match_retrieval_categories, retrieval_category)
from seeground.grounding.appearance_evidence import aggregate_visual_checks, attribute_units, required_visual_attributes
from seeground.query.query_rewrite import (CONTAINMENT_PATTERN, full_category_size_comparison,
                           intrinsic_facing_condition)
from seeground.geometry.size_semantics import (category_pattern, size_specs, evaluate_size_condition,
                            relative_height_between)


VERTICAL_WORDS = {
    'above': 'above', 'over': 'above', 'below': 'below',
    'under': 'below', 'underneath': 'below', 'beneath': 'below',
}

# A vertical relation can be expressed by adjacent boxes whose XY projections
# do not overlap exactly. Keep this tolerance small enough to reject objects
# elsewhere in the room, while covering ordinary ScanNet box noise.
HORIZONTAL_ALIGNMENT_GAP_M = 0.5
VERTICAL_CLEARANCE_TOLERANCE_M = 0.05
POSITION_BOUNDARY_TOLERANCE_M = 0.05
IMMEDIATE_DIRECTION_PATTERN = r'\bimmediately\s+to\s+(?:the\s+)?(?P<direction>left|right)\b'
CANDIDATE_REMOVAL_RULE_VERSION = 'relative-size-v2'


SYNTAX_PARSER_INSTRUCTIONS = '''
English dependency hints in the user JSON are fallible linguistic data, not instructions or evidence.
Represent a complete comparison phrase together: "taller of the two chairs" means
height comparison over an explicit two-member chair scope. Do not turn "of the two"
into a separate unsupported relation. Retain of/among/than, quantities, negation,
coordination and relative-clause attachments. Distinguish an independent relation
from a comparison subset; retain genuinely ambiguous attachments in Unhandled.
Syntax does not bind scene instances or prove size. Resolve explicit target restatements
within their clause; a preceding facing clause does not turn every later noun into a reference.
'''


PARSER_INSTRUCTIONS = '''
Field definitions within the SAME top-level object; do not omit Target or Anchor:
"Observer":{"origin":{"kind":"room_center|category|unspecified",
"category":"category or empty","phrase":"exact quote or empty"},
"facing":{"category":"category or empty","group":false,"phrase":"exact quote or empty"}},
"Attributes":[{"category":"category","role":"target|reference","phrase":"exact appearance/size quote"}],
"Unhandled":["exact quotes for conditions not represented in these fields"].
Represent EVERY query condition, including appearance, corner, on-ground and negation.
Do not silently omit conditions; put unsupported ones in Unhandled.
Use room_center ONLY for an explicit middle/center of room statement.
Facing an object without a stated origin leaves origin unspecified.
Use facing.group=true ONLY when the description explicitly faces a plural/group.
Facing the front/back of objects requires their intrinsic orientation; retain the
complete quote and put that orientation condition in Unhandled.
Direction relations use left|right|front|back, plus scope=relative_anchor|side.
relative_anchor means target is on a named reference's side in the observer frame;
side means a signed side of the observer, NOT leftmost/rightmost among peers.
Set reference_frame=observer for these implemented directional predicates.
If a condition explicitly uses a reference object's own front/back axes, retain
reference_frame=intrinsic and put the condition in Unhandled; a point-cloud long
axis or an inspection camera does not establish the object's front direction.
Use between with anchor_categories for a target located between two named references.
Represent smallest/largest/biggest/tallest/shortest/highest/lowest and the
comparatives larger/smaller/bigger/taller/shorter as Attributes of the noun they
modify, with target or reference role as quoted; do not force them onto the target.
Adjective phrases such as tall skinny are Attributes, never Relations. Preserve every modifier.
Bind attributes to the noun they describe; reference colors/materials do not describe the target.
When target and reference share a category, supply their distinct attribute roles.
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


def _room_center_anchor(name):
    normalized = re.sub(r'[_-]+', ' ', name.strip().casefold()) if isinstance(name, str) else ''
    return normalized in ('room', 'room center', 'center of room', 'middle of room')


def _explicit_room_center_origin(query, observer):
    origin = observer.get('origin', {}) if isinstance(observer, dict) else {}
    if not isinstance(origin, dict) or origin.get('kind') != 'room_center':
        return False
    phrase = origin.get('phrase', '')
    return (quoted(query, phrase) and re.search(r'\b(middle|cent(?:er|re))\b', phrase, re.I)
            and re.search(r'\broom\b', phrase, re.I))


def _complete_observer_coreference_clause(query, observer, relation):
    origin = observer.get('origin', {})
    facing = observer.get('facing', {})
    phrases = (origin.get('phrase', ''), facing.get('phrase', ''), relation.get('phrase', ''))
    if not all(isinstance(phrase, str) and phrase for phrase in phrases):
        return False
    pattern = (r'\s*(?:when\s+)?' + re.escape(phrases[0]) + r'[\s,]+' +
               re.escape(phrases[1]) + r'[\s,]+' + re.escape(phrases[2]) + r'[.!?]*\s*')
    return bool(re.fullmatch(pattern, query, re.I))


def recover_distance_relations(query, parsed):
    """Recover only a complete, simple distance clause with explicit bindings."""
    parsed = dict(parsed)
    raw_relations = parsed.get('Relations', [])
    if not isinstance(raw_relations, list) or any(not isinstance(r, dict) for r in raw_relations):
        return parsed
    relations = list(raw_relations)
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
    raw_relations = parsed.get('Relations', [])
    if not isinstance(raw_relations, list) or any(not isinstance(r, dict) for r in raw_relations):
        return parsed
    relations = [dict(r) for r in raw_relations]
    targets = parsed.get('Target') if isinstance(parsed.get('Target'), list) else []
    anchors = parsed.get('Anchor') if isinstance(parsed.get('Anchor'), list) else []
    anchors = list(dict.fromkeys(anchors)) if all(isinstance(a, str) for a in anchors) else anchors
    if len(targets) != 1 or not anchors:
        return parsed

    def embedded_inverse(text, anchor_name, source_phrase=None):
        # In "a shelf above it", the shelf is the reference and "it" is
        # the described target, so the target relation has the opposite sign.
        if not isinstance(anchor_name, str) or not anchor_name.strip():
            return None
        anchor_pattern = re.escape(anchor_name.strip()) + r's?'
        matches = list(re.finditer(
            r'(?P<subject>\b' + anchor_pattern +
            r')\s+(?P<operator>right\s+above|directly\s+above|above|over|'
            r'below|directly\s+under|under|underneath|beneath)\s+(?P<object>it|them|the target)\b',
            text, re.I))
        if source_phrase is not None:
            quote = source_phrase.strip().rstrip('.!?').casefold()
            matches = [match for match in matches if quote in match.group().casefold()]
            if len(matches) != 1:
                return None
        if not matches:
            return None
        match = matches[0]
        # Once a vertical operator has already occurred, a later matching
        # noun phrase belongs to a subsequent clause rather than the target's
        # embedded reference (for example, "shelf above table, with shelf above it").
        if re.search(r'\b(?:above|over|below|under|underneath|beneath)\b',
                     text[:match.start('subject')], re.I):
            return None
        operator = re.search(r'\b(above|over|below|under|underneath|beneath)\b',
                             match['operator'], re.I).group().lower()
        relation = 'below' if operator in ('above', 'over') else 'above'
        start = match.start('subject')
        return {'relation': relation, 'start': start, 'end': match.end()}

    # Correct the common Qwen failure: "right above" returned as near/other.
    unresolved_short_quote = False
    for relation in relations:
        # An explicit malformed anchor must remain malformed so structural
        # validation can report it; recovery may only fill an omitted/blank
        # string from the single parsed anchor.
        anchor_value = relation.get('anchor_category')
        if 'anchor_category' in relation and not isinstance(anchor_value, str):
            continue
        phrase = relation.get('phrase', '')
        if not isinstance(phrase, str):
            continue
        anchor_name = (anchor_value.strip() if isinstance(anchor_value, str) and anchor_value.strip()
                       else anchors[0] if len(anchors) == 1 else None)
        if not isinstance(anchor_name, str) or not anchor_name.strip():
            continue
        inverse = embedded_inverse(phrase, anchor_name)
        if not inverse and re.fullmatch(
                r'(?:directly\s+)?(?:above|over|below|under|underneath|beneath)\s+(?:it|them|the target)[.!?]?',
                phrase.strip(), re.I):
            inverse = embedded_inverse(query, anchor_name, source_phrase=phrase)
            if not inverse:
                # An operator plus "it" does not establish its subject or
                # direction. Preserve this clause for validation to mark missing.
                relation['relation'] = 'other'
                unresolved_short_quote = True
                continue
        if inverse:
            relation['relation'] = inverse['relation']
            relation['scope'] = 'relative_anchor'
            relation['target_category'] = relation.get('target_category') or targets[0]
            if not anchor_value:
                relation['anchor_category'] = anchor_name
            relation['source_subject'] = anchor_name
            relation['source_object'] = targets[0]
            continue
        match = re.search(r'\b(?:right\s+above|directly\s+above|above|over|below|directly\s+under|under|underneath|beneath)\b',
                          phrase, re.I)
        if match:
            relation['relation'] = VERTICAL_WORDS[re.search(
                r'\b(above|over|below|under|underneath|beneath)\b', match.group(), re.I).group().lower()]
            relation['scope'] = 'relative_anchor'
            relation['target_category'] = relation.get('target_category') or targets[0]
            if not anchor_value:
                relation['anchor_category'] = anchor_name

    if unresolved_short_quote or any(r.get('relation') in ('above', 'below') for r in relations):
        parsed['Relations'] = relations
        return parsed
    if len(anchors) != 1:
        parsed['Relations'] = relations
        return parsed

    inverse = embedded_inverse(query, anchors[0])
    if inverse:
        phrase = query[inverse['start']:inverse['end']].strip().rstrip('.,!?')
        relations.append({'target_category': targets[0], 'anchor_category': anchors[0],
                          'relation': inverse['relation'], 'scope': 'relative_anchor',
                          'phrase': phrase})
        parsed['Relations'] = relations
        return parsed

    vertical = re.search(
        r'\b(?:right\s+above|directly\s+above|above|over|below|directly\s+under|under|underneath|beneath)\b',
        query, re.I)
    if not vertical:
        parsed['Relations'] = relations
        return parsed
    operator = re.search(r'\b(above|over|below|under|underneath|beneath)\b', vertical.group(), re.I).group().lower()
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
    raw_relations = parsed.get('Relations', [])
    if not isinstance(raw_relations, list) or any(not isinstance(r, dict) for r in raw_relations):
        return parsed
    relations = [dict(r) for r in raw_relations]
    targets = parsed.get('Target') if isinstance(parsed.get('Target'), list) else []
    anchors = parsed.get('Anchor') if isinstance(parsed.get('Anchor'), list) else []
    if len(targets) != 1 or len(anchors) < 2:
        parsed['Relations'] = relations
        return parsed
    for relation in relations:
        if relation.get('relation') == 'between':
            names = relation.get('anchor_categories')
            # None/missing and a wrong-length list are recoverable omissions;
            # scalar values are malformed and must reach validation intact.
            if (names is None or (isinstance(names, list) and len(names) != 2)) and len(anchors) >= 2:
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


def attribute_role(parsed, attribute):
    """Legacy category-only bindings are usable only for a single language role."""
    if attribute.get('role') in ('target', 'reference'):
        return attribute['role']
    category = retrieval_category(attribute.get('category', ''))
    targets = parsed.get('Target', [])
    targets = [targets] if isinstance(targets, str) else targets
    references = parsed.get('Anchor', [])
    references = [references] if isinstance(references, str) else list(references)
    for relation in parsed.get('Relations', []):
        if isinstance(relation, dict):
            names = relation.get('anchor_categories') if relation.get('relation') == 'between' else [relation.get('anchor_category', '')]
            references.extend(names if isinstance(names, list) else [])
    observer = parsed.get('Observer', {})
    if isinstance(observer, dict):
        references.extend(entry.get('category', '') for entry in observer.values()
                          if isinstance(entry, dict))
    target_match = category in {retrieval_category(name) for name in targets}
    reference_match = category in {retrieval_category(name) for name in references}
    if target_match != reference_match:
        return 'target' if target_match else 'reference'
    return None


def _source_attribute_bindings(query, parsed, phrase):
    """Find direct noun attachments without collapsing repeated language roles."""
    if not quoted(query, phrase):
        return set()
    from seeground.grounding.v4_policy import v7_enabled
    recovery = parsed.get('category_recovery') if v7_enabled() else None
    if recovery:
        part = re.search(r'\b' + re.escape(recovery['source_quote']) + r'\b', query, re.I)
        occurrences = list(re.finditer(re.escape(phrase), query, re.I))
        if part and len(occurrences) == 1 and occurrences[0].end() <= part.start() and re.fullmatch(
                r'(?:\s+(?:big|small|large|largest|smallest|brown|white|round))*\s*',
                query[occurrences[0].end():part.start()], re.I):
            return {(recovery['canonical'], 'target')}
    target_names = parsed.get('Target', [])
    target_names = [target_names] if isinstance(target_names, str) else target_names
    anchor_names = parsed.get('Anchor', [])
    anchor_names = [anchor_names] if isinstance(anchor_names, str) else anchor_names
    names = target_names + anchor_names
    targets = {retrieval_category(name) for name in target_names}
    observer_spans = []
    for entry in parsed.get('Observer', {}).values():
        if isinstance(entry, dict) and quoted(query, entry.get('phrase', '')):
            observer_spans.extend((match.start(), match.end()) for match in
                                  re.finditer(re.escape(entry['phrase']), query, re.I))
    mentions = []
    for name in names:
        category = retrieval_category(name)
        if not category:
            continue
        for match in re.finditer(r'\b' + category_pattern(category) + r'\b', query, re.I):
            key = (match.start(), match.end(), category)
            if key not in mentions:
                mentions.append(key)
    syntax_enabled = parsed.get('query_syntax', {}).get('backend') == 'spacy'
    if syntax_enabled:
        mentions = [(a, b, c) for a, b, c in mentions
                    if not any(left <= a and b <= right and (left, right) != (a, b)
                               for left, right, _ in mentions)]
    attachments = []
    direct_connector = r'(?:with|is|has|that\s+(?:is|has))\s+(?:(?:the|a|an)\s+)?'
    spatial_words = (r'\b(?:above|over|below|under|underneath|beneath|between|'
                     r'near|beside|next to|left of|right of|behind|in front of|facing)\b')
    head_modifiers = _SOURCE_VISUAL_MODIFIERS | {
        'small', 'smallest', 'smaller', 'large', 'largest', 'larger', 'big',
        'biggest', 'bigger', 'little', 'skinny', 'narrow', 'thin', 'wide',
        'thick', 'tall', 'tallest', 'taller', 'high', 'highest', 'short',
        'shortest', 'shorter', 'low', 'lowest',
    }
    for attr_match in re.finditer(re.escape(phrase), query, re.I):
        # Parser attributes sometimes include their noun ("smaller trash can").
        # Bind that quoted head exactly as a bare modifier would be bound. A
        # phrase spanning multiple modeled nouns cannot identify one role.
        embedded_nouns = [(a, b, c) for a, b, c in mentions
                          if attr_match.start() <= a and b <= attr_match.end()]
        for start, end, category in mentions:
            between_tokens = re.findall(r'\b\w+\b', query[attr_match.end():start].casefold())
            before_noun = (attr_match.end() <= start and
                           all(token in head_modifiers for token in between_tokens) and
                           not re.search(r'[^a-z\s-]', query[attr_match.end():start], re.I))
            gap = query[end:attr_match.start()]
            # Keep a quoted connector/article on either side of the property
            # boundary, but never step through a different modeled noun.
            after_noun = (end <= attr_match.start() and not embedded_nouns and
                          not re.search(spatial_words, phrase, re.I) and bool(
                              re.fullmatch(r'\s+' + direct_connector, gap, re.I) or
                              (re.fullmatch(r'\s+', gap) and
                               re.match(direct_connector + r'\S', phrase, re.I))))
            if any(attr_match.end() <= noun_start and
                   all(token in head_modifiers for token in re.findall(
                       r'\b\w+\b', query[attr_match.end():noun_start].casefold())) and
                   not re.search(r'[^a-z\s-]', query[attr_match.end():noun_start], re.I)
                   for noun_start, _, _ in mentions):
                after_noun = False
            prefix_tokens = re.findall(r'\b\w+\b', query[attr_match.start():start].casefold())
            if prefix_tokens and prefix_tokens[0] in {'the', 'a', 'an', 'another', 'other'}:
                prefix_tokens = prefix_tokens[1:]
            embedded_head = (embedded_nouns == [(start, end, category)] and
                             all(token in head_modifiers for token in prefix_tokens) and
                             not re.search(spatial_words, query[end:attr_match.end()], re.I))
            if not (before_noun or after_noun or embedded_head):
                continue
            # An explicit selection starts the target clause anew, even when
            # an observer mentioned the same category before it.
            commands = list(re.finditer(r'\b(?:choose|select|find|pick|click)\b',
                                        query[:start], re.I))
            clause_start = commands[-1].end() if commands else 0
            if syntax_enabled:
                restatements = list(re.finditer(
                    r'\bthe\s+(?:correct|target)\s+(?:object|' + category_pattern(category) +
                    r')\s+is\s+', query[clause_start:start], re.I))
                if category in targets and restatements:
                    clause_start += restatements[-1].end()
            prefix = query[clause_start:start]
            earlier_target = any(clause_start <= a < start and c in targets
                                 for a, _, c in mentions)
            in_reference_clause = bool(re.search(
                r'\b(?:facing|standing|looking|above|over|below|under|underneath|beneath|'
                r'between|near|beside|next to|left of|right of|behind|in front of|with)\b',
                prefix, re.I))
            role = ('target' if category in targets and not earlier_target and
                    not in_reference_clause else 'reference')
            if role == 'reference' and category in targets and not any(
                    a <= start and end <= b for a, b in observer_spans):
                earlier_nonobserver_target = any(
                    clause_start <= a < start and c in targets and not any(
                        left <= a and b <= right for left, right in observer_spans)
                    for a, b, c in mentions)
                spatial_reference = re.search(
                    r'\b(?:above|over|below|under|underneath|beneath|between|near|'
                    r'beside|next to|left of|right of|behind|in front of|with)\b', prefix, re.I)
                if not earlier_nonobserver_target and not spatial_reference:
                    # A preceding observer clause does not uniquely assign the
                    # role of a later noun. Keep both interpretations pending.
                    attachments.append((category, 'target'))
            attachments.append((category, role))
    bindings = set(attachments)
    if not bindings and len(targets) == 1 and re.search(
            r'\bthe (?:correct|target) one\s+(?:is|has)\s+' + re.escape(phrase), query, re.I):
        bindings.add((next(iter(targets)), 'target'))
    return bindings


_SOURCE_VISUAL_MODIFIERS = {
    'black', 'white', 'red', 'blue', 'green', 'grey', 'gray', 'brown', 'yellow',
    'orange', 'purple', 'pink', 'rainbow', 'wooden', 'metal', 'metallic',
    'plastic', 'glass', 'striped', 'dotted', 'checkered', 'speckled', 'smooth',
    'rough', 'textured', 'folded', 'vertical', 'horizontal', 'round', 'square',
    'rectangular', 'dark', 'light',
}


def _recover_surface_attributes(query, parsed, attributes):
    """Recover an omitted visual property only when it names a clear object part."""
    parts = r'(?:top|bottom|front|back|side|surface|panel|upholstery|fabric)'
    modifier = '|'.join(sorted(_SOURCE_VISUAL_MODIFIERS, key=len, reverse=True))
    pattern = re.compile(r'\bwith\s+(?:the\s+|a\s+|an\s+)?'
                         r'(?P<phrase>(?:' + modifier + r')\s+' + parts + r')\b', re.I)
    existing = {(retrieval_category(item.get('category', '')),
                 item.get('phrase', '').casefold())
                for item in attributes if isinstance(item, dict)}
    missing = []
    for match in pattern.finditer(query):
        phrase = match.group('phrase')
        bindings = _source_attribute_bindings(query, parsed, phrase)
        if len(bindings) != 1:
            missing.append('appearance condition role not represented: ' + phrase)
            continue
        category, role = next(iter(bindings))
        key = (retrieval_category(category), phrase.casefold())
        if key not in existing:
            attributes.append({'category': category, 'role': role, 'phrase': phrase,
                               'binding_source': 'quoted surface attribute recovery'})
            existing.add(key)
    return attributes, missing


def _normalize_relation_scope(query, relation, targets, observer=None):
    relation = dict(relation)
    kind, phrase = relation.get('relation'), relation.get('phrase', '')
    if not quoted(query, phrase):
        return relation
    anchor = relation.get('anchor_category')
    facing = observer.get('facing', {}) if isinstance(observer, dict) else {}
    target = targets[0] if len(targets) == 1 else ''
    if (kind in ('left', 'right') and target and isinstance(facing, dict) and
            facing.get('group') is True and _explicit_room_center_origin(query, observer) and
            retrieval_category(target) == retrieval_category(facing.get('category', '')) ==
            retrieval_category(anchor or '') and
            quoted(query, facing.get('phrase')) and re.search(
                r'\b(?:fac(?:e|ing)|look(?:ing)?)\b', facing['phrase'], re.I) and
            re.fullmatch(r'(?:the\s+)?' + re.escape(target) +
                         r'\s+on\s+(?:the\s+)?' + kind, phrase.strip(), re.I)):
        origin_phrase = observer['origin']['phrase']
        complete = (r'\s*(?:when\s+)?' + re.escape(origin_phrase) + r'[\s,]+' +
                    re.escape(facing['phrase']) + r'[\s,]+' +
                    r"(?:(?:it(?:'s|\s+is)|the\s+correct\s+one\s+is)\s+)?" +
                    re.escape(phrase) + r'[.!?]*\s*')
        if re.fullmatch(complete, query, re.I):
            relation.update(anchor_category='', anchor_categories=[], scope='side')
            anchor = ''
    if kind in ('above', 'below') and isinstance(anchor, str) and anchor.strip():
        relation['scope'] = 'relative_anchor'
    elif kind in ('left', 'right', 'front', 'back') and not anchor:
        extreme = re.search(r'\b(?:far|most|furthest|farthest|leftmost|rightmost)\b', phrase, re.I)
        relative = re.search(r'\b(?:of|than)\b', phrase, re.I)
        side = re.search(r'\b(?:on|to|at)\s+(?:the\s+)?' + kind + r'\b', phrase, re.I)
        adjective = (len(targets) == 1 and re.fullmatch(
            r'(?:the\s+)?' + kind + r'\s+' + re.escape(targets[0]) + r'[.!?]?', phrase.strip(), re.I))
        if not extreme and not relative and (side or adjective):
            relation['scope'] = 'side'
    return relation


def _condition_records(query, parsed, raw_relations, raw_attributes):
    """Keep quoted conditions and role/evidence state together for audit."""
    targets = parsed.get('Target', [])
    targets = [targets] if isinstance(targets, str) else (targets if isinstance(targets, list) else [])
    anchors = parsed.get('Anchor', [])
    anchors = [anchors] if isinstance(anchors, str) else (anchors if isinstance(anchors, list) else [])
    target_categories = {retrieval_category(name) for name in targets
                         if isinstance(name, str) and retrieval_category(name)}
    reference_categories = {retrieval_category(name) for name in anchors
                            if isinstance(name, str) and retrieval_category(name)}
    for relation in parsed.get('Relations', []):
        if not isinstance(relation, dict):
            continue
        for key in ('anchor_category', 'anchor_categories'):
            names = relation.get(key, []) or []
            names = [names] if isinstance(names, str) else names
            names = names if isinstance(names, list) else []
            reference_categories.update(retrieval_category(name) for name in names
                                        if isinstance(name, str) and retrieval_category(name))
    observer = parsed.get('Observer', {})
    if isinstance(observer, dict):
        for role in ('origin', 'facing'):
            entry = observer.get(role, {})
            name = entry.get('category', '') if isinstance(entry, dict) else ''
            category = retrieval_category(name) if isinstance(name, str) else ''
            if category:
                reference_categories.add(category)

    records = []

    def add(condition_type, phrase, target_role=None, reference_role=None,
            other_role=None, binding_status='bound', evidence_status='pending', reason=None):
        quote = phrase if quoted(query, phrase) else None
        records.append({
            'condition_type': condition_type,
            'source_quote': quote,
            'reported_phrase': phrase if isinstance(phrase, str) else '',
            'target_role': target_role,
            'reference_role': reference_role,
            'other_role': other_role,
            'binding_status': binding_status,
            'evidence_status': evidence_status,
            'evidence': [],
            **({'reason': reason} if reason else {}),
        })

    for attribute in raw_attributes:
        if not isinstance(attribute, dict):
            continue
        phrase = attribute.get('phrase', '')
        category_name = attribute.get('category', '')
        category = retrieval_category(category_name) if isinstance(category_name, str) else ''
        target_match = category in target_categories
        reference_match = category in reference_categories
        explicit_role = attribute.get('role')
        target_role = reference_role = None
        binding_status = 'bound'
        reason = None
        if explicit_role == 'target' and target_match:
            target_role = {'category': category}
        elif explicit_role == 'reference' and reference_match:
            reference_role = {'category': category}
        elif explicit_role in ('target', 'reference'):
            binding_status = 'unbound'
            reason = 'explicit attribute role does not match a supplied category'
        elif target_match and reference_match:
            binding_status = 'ambiguous'
            reason = 'attribute category occurs as both target and reference; role was not supplied'
        elif target_match:
            target_role = {'category': category}
        elif reference_match:
            reference_role = {'category': category}
        else:
            binding_status = 'unbound'
            reason = 'attribute category is not a supplied target or reference'
        phrase_text = phrase.casefold() if isinstance(phrase, str) else ''
        condition_type = ('size' if re.search(
            r'\b(smallest|largest|biggest|tallest|shortest|highest|lowest|larger|smaller|'
            r'bigger|taller|shorter|longer|large|small|big|little|skinny|narrow|wide|thin|thick|'
            r'tall|high|short|low)\b', phrase_text) else 'appearance')
        add(condition_type, phrase, target_role, reference_role,
            binding_status=binding_status,
            evidence_status='missing' if binding_status != 'bound' else 'pending', reason=reason)
        if condition_type == 'size' and parsed.get('query_syntax', {}).get('backend') == 'spacy':
            from seeground.query.query_syntax import comparison_hint
            hints = [comparison_hint(parsed, word, category) for word in
                     re.findall(r'\b\w+\b', phrase_text)]
            records[-1]['language_comparisons'] = [hint for hint in hints if hint]
            records[-1]['scene_binding_status'] = 'pending'

    scope_match = re.search(r'\bof the two(?:\s+[a-z-]+)?\b', query, re.I)
    if scope_match and re.search(
            r'\b(smallest|largest|biggest|tallest|shortest|highest|lowest|'
            r'larger|smaller|bigger|taller|shorter|longer|small|large|big|little|tall|short)\b', query, re.I):
        add('size_scope', scope_match.group(),
            {'category': next(iter(target_categories))} if len(target_categories) == 1 else None,
            binding_status='pending', evidence_status='pending',
            reason='exactly two instances must be bound by measured size evidence')

    for relation in raw_relations:
        if not isinstance(relation, dict):
            add('relation', '', binding_status='unbound', evidence_status='missing',
                reason='relation entry is not an object')
            continue
        kind = relation.get('relation', 'other')
        target_name = relation.get('target_category', '')
        target = retrieval_category(target_name) if isinstance(target_name, str) else ''
        reference_names = (relation.get('anchor_categories', []) or [] if kind == 'between'
                           else [relation.get('anchor_category', '')])
        reference_names = ([reference_names] if isinstance(reference_names, str)
                           else reference_names if isinstance(reference_names, list) else [])
        reference_names = [retrieval_category(name) for name in reference_names
                           if isinstance(name, str) and retrieval_category(name)]
        phrase = relation.get('phrase', '')
        valid = relation in parsed.get('Relations', [])
        add('relation:' + str(kind), phrase,
            {'category': target} if valid and target else None,
            {'categories': reference_names} if valid and reference_names else None,
            binding_status='bound' if valid else 'unbound',
            evidence_status='pending' if valid else 'unsupported',
            reason=None if valid else 'relation failed structural validation')

    for match in re.finditer(IMMEDIATE_DIRECTION_PATTERN, query, re.I):
        bound = []
        for relation in parsed.get('Relations', []):
            phrase = relation.get('phrase', '')
            if relation.get('relation') != match['direction'].lower() or not quoted(query, phrase):
                continue
            occurrences = list(re.finditer(re.escape(phrase), query, re.I))
            if len(occurrences) == 1 and (occurrences[0].start() < match.end()
                                          and occurrences[0].end() > match.start()):
                bound.append(relation)
        relation = bound[0] if len(bound) == 1 else {}
        target = retrieval_category(relation.get('target_category', ''))
        reference = retrieval_category(relation.get('anchor_category', ''))
        add('relation:immediate_proximity', match.group(),
            target_role={'category': target} if target else None,
            reference_role={'categories': [reference]} if reference else None,
            binding_status='bound' if target and reference else 'unbound',
            evidence_status='missing',
            reason='immediate proximity has no calibrated predicate: ' + match.group())

    for coreference in parsed.get('target_coreferences', []):
        if not isinstance(coreference, dict):
            continue
        category = retrieval_category(coreference.get('category', ''))
        phrase = coreference.get('phrase', '')
        if category and quoted(query, phrase):
            add('target_coreference', phrase, target_role={'category': category},
                evidence_status='pending',
                reason='bound to the unique target; its represented relation supplies the evidence')

    if isinstance(observer, dict):
        for role in ('origin', 'facing'):
            entry = observer.get(role, {})
            if isinstance(entry, dict) and entry.get('phrase'):
                category = retrieval_category(entry.get('category', ''))
                add('viewpoint:' + role, entry.get('phrase'),
                    other_role={'role': role, 'category': category or None},
                    binding_status='bound' if category or entry.get('kind') == 'room_center' else 'unbound',
                    evidence_status='pending' if category or entry.get('kind') == 'room_center' else 'missing')
                if role == 'facing' and entry.get('reference_frame') == 'intrinsic':
                    add('viewpoint:intrinsic_orientation', entry['phrase'],
                        other_role={'role': 'facing_orientation', 'category': category,
                                    'part': entry.get('orientation_part')},
                        binding_status='bound', evidence_status='missing',
                        reason='object-intrinsic facing orientation is unavailable: ' + entry['phrase'])

    for phrase in parsed.get('Unhandled', []):
        add('unhandled', phrase, binding_status='unbound', evidence_status='unsupported',
            reason='parser did not map this condition to a supported evidence field')

    for match in re.finditer(r"\b(no|not|nothing|never|without|except)\b|n['’]t\b", query, re.I):
        add('negation', match.group(), binding_status='unbound', evidence_status='unsupported',
            reason='negation/exclusion predicate is not implemented')

    for diagnostic in parsed.get('diagnostic_missing', []):
        if not isinstance(diagnostic, str):
            continue
        phrase = None
        condition_type = 'coverage_gap'
        binding_status = 'unbound'
        if diagnostic.startswith('appearance condition role not represented: '):
            gap = re.fullmatch(r'appearance condition role not represented: (.+) \((target|reference) (.+)\)',
                               diagnostic)
            if gap:
                phrase, role, category = gap.groups()
                add('appearance', phrase,
                    target_role={'category': category} if role == 'target' else None,
                    reference_role={'category': category} if role == 'reference' else None,
                    binding_status='unbound', evidence_status='missing', reason=diagnostic)
                continue
        elif diagnostic.startswith('appearance condition not represented: '):
            word = diagnostic.split(': ', 1)[1]
            found = re.search(r'\b' + re.escape(word) + r'\b', query, re.I)
            phrase = found.group() if found else None
            condition_type = 'size' if re.search(
                r'\b(smallest|largest|biggest|tallest|shortest|highest|lowest|larger|smaller|'
                r'bigger|taller|shorter|longer|large|small|big|little|skinny|narrow|wide|thin|thick|'
                r'tall|high|short|low)\b', word, re.I) else 'appearance'
        elif diagnostic.startswith('unsupported relation: '):
            phrase = diagnostic[len('unsupported relation: '):]
            condition_type = 'relation'
        elif diagnostic.startswith('unhandled condition: '):
            phrase = diagnostic[len('unhandled condition: '):]
            condition_type = 'unhandled'
        elif diagnostic == 'vertical condition not represented':
            found = re.search(r'\b(above|over|below|under|underneath|beneath)\b', query, re.I)
            phrase = found.group() if found else None
            condition_type = 'relation:vertical'
        elif diagnostic.startswith('size comparison scope not bound: '):
            phrase = diagnostic[len('size comparison scope not bound: '):]
            condition_type = 'size_scope'
        elif diagnostic.startswith('object-intrinsic facing orientation is unavailable: '):
            phrase = diagnostic[len('object-intrinsic facing orientation is unavailable: '):]
            condition_type = 'viewpoint:intrinsic_orientation'
        elif diagnostic.startswith('attribute target/reference role is ambiguous: '):
            phrase = diagnostic[len('attribute target/reference role is ambiguous: '):]
            binding_status = 'ambiguous'
            condition_type = 'size' if re.search(
                r'\b(smallest|largest|biggest|tallest|shortest|highest|lowest|larger|smaller|'
                r'bigger|taller|shorter|longer|large|small|big|little|skinny|narrow|wide|thin|thick|'
                r'tall|high|short|low)\b', phrase, re.I) else 'appearance'
            record = next((item for item in records
                           if item['condition_type'] == condition_type and
                           item['source_quote'] and
                           item['source_quote'].casefold() == phrase.casefold()), None)
            if record:
                record['binding_status'] = 'ambiguous'
                record['evidence_status'] = 'missing'
                record['reason'] = diagnostic
                continue
        elif diagnostic == 'observer origin or facing direction is unspecified':
            found = re.search(r'\b(left|right|ahead|behind|front|back)\b', query, re.I)
            phrase = found.group() if found else None
            condition_type = 'viewpoint'
        elif diagnostic == 'direction condition not represented':
            found = re.search(r'\b(left|right|ahead|behind)\b', query, re.I)
            phrase = found.group() if found else None
            condition_type = 'relation:direction'
        elif diagnostic.startswith('immediate proximity has no calibrated predicate: '):
            continue
        if phrase and any(record['source_quote'] and
                          record['source_quote'].casefold() == phrase.casefold() and
                          record['evidence_status'] in ('missing', 'unsupported')
                          for record in records):
            continue
        add(condition_type, phrase or '', binding_status=binding_status, evidence_status='missing',
            reason=diagnostic)

    return records


def validate_structure(query, parsed):
    parsed = dict(parsed)
    from seeground.query.query_syntax import analyze_query
    parsed['query_syntax'] = analyze_query(query)
    parsed.pop('_size_scope_bindings', None)
    parsed.pop('_verified_reference_bindings', None)
    parsed['source_query'] = query
    from seeground.grounding.v4_policy import v7_enabled
    if v7_enabled():
        from seeground.grounding.v7_evidence import recover_part_category
        recover_part_category(query, parsed)
    input_relations = parsed.get('Relations', [])
    preflight_missing = []
    malformed_relations = []
    if not isinstance(input_relations, list):
        preflight_missing.append('relation coverage not supplied by parser')
        # Keep the malformed root value in the audit input.  Recovery needs a
        # list to operate on, but dropping the original value would make the
        # condition ledger claim that no relation was supplied.
        malformed_relations = [input_relations]
        parsed['Relations'] = []
    else:
        malformed_relations = [relation for relation in input_relations
                               if not isinstance(relation, dict)]
        if malformed_relations:
            preflight_missing.append('parser produced a non-object relation entry')
        parsed['Relations'] = [relation for relation in input_relations
                               if isinstance(relation, dict)]
    parsed = recover_between_relations(query, recover_vertical_relations(query, parsed))
    observer_input = parsed.get('Observer', {})
    room_center_origin = _explicit_room_center_origin(query, observer_input)
    if room_center_origin:
        anchors = parsed.get('Anchor', [])
        if isinstance(anchors, list):
            parsed['Anchor'] = [name for name in anchors if not _room_center_anchor(name)]
        elif _room_center_anchor(anchors):
            parsed['Anchor'] = []
        relations = []
        for relation in parsed['Relations']:
            if isinstance(relation, dict):
                relation = dict(relation)
                if (relation.get('relation') in ('left', 'right', 'front', 'back') and
                        _room_center_anchor(relation.get('anchor_category'))):
                    relation['anchor_category'] = ''
                    relation['scope'] = 'side'
            relations.append(relation)
        parsed['Relations'] = relations
    # Recover only a complete, simple size request; compound conditions stay untouched.
    targets = parsed.get('Target', [])
    targets = [targets] if isinstance(targets, str) else targets
    if isinstance(parsed.get('Attributes'), list):
        parsed['Attributes'] = [
            {key: value for key, value in attribute.items()
             if key not in ('geometry_attribute', 'comparison_scope', 'scope_phrase', 'required_member_count', 'size_semantics')}
            if isinstance(attribute, dict) else attribute for attribute in parsed['Attributes']]
    comparison = full_category_size_comparison(query, parsed)
    if comparison and isinstance(parsed.get('Unhandled'), list):
        parsed['Anchor'] = []
        parsed['Attributes'] = [comparison]
        covered = []
        kept = []
        for phrase in parsed['Unhandled']:
            covered_phrase = (quoted(query, phrase) if parsed['query_syntax']['backend'] == 'spacy'
                              else isinstance(phrase, str) and
                              phrase.strip().rstrip('.!?').casefold() == query.strip().rstrip('.!?').casefold())
            if covered_phrase:
                covered.append({'source_quote': phrase, 'covered_by': 'complete_size_comparison',
                                'source_phrase': query, 'scene_binding_status': 'pending'})
            else:
                kept.append(phrase)
        parsed['Unhandled'] = kept
        parsed['condition_coverage'] = covered
    if parsed['query_syntax']['backend'] == 'spacy' and isinstance(parsed.get('Unhandled'), list):
        # Cover only the quoted cardinal scope. Related appearance or spatial
        # clauses retain their own obligations, and scene binding stays pending.
        kept = []
        for phrase in parsed['Unhandled']:
            covered = next((hint for hint in parsed['query_syntax']['comparisons']
                if hint['scope_type'] == 'explicit_set' and hint['required_member_count'] == 2
                and isinstance(phrase, str) and quoted(query, phrase)
                and re.fullmatch(r'of\s+the\s+two(?:\s+' + category_pattern(hint['head_category']) + r')?',
                                 phrase.strip().rstrip('.!?'), re.I)
                and any(isinstance(attr, dict) and attr.get('role') == 'target'
                        and retrieval_category(attr.get('category', '')) == retrieval_category(hint['head_category'])
                        and hint['word'] in re.findall(r'\b\w+\b', attr.get('phrase', '').casefold())
                        for attr in parsed.get('Attributes', []))), None)
            if covered:
                parsed.setdefault('condition_coverage', []).append({
                    'source_quote': phrase, 'covered_by': 'explicit_comparison_scope',
                    'source_phrase': covered['source_phrase'], 'scene_binding_status': 'pending'})
            else:
                kept.append(phrase)
        parsed['Unhandled'] = kept
    if len(targets) == 1:
        match = re.fullmatch(
            r'\s*(?:(?:choose|select|find|pick|click)\s+|it\s+is\s+)?(?:the\s+)?'
            r'(smallest|largest|biggest|tallest|shortest|highest|lowest|'
            r'larger|smaller|bigger|taller|shorter|longer|small|large|big|little|tall|short)\s+'
            + re.escape(targets[0]) + r'\s*[.!?]?\s*', query, re.I)
        if (match and not parsed.get('Anchor') and not parsed.get('Relations')
                and isinstance(parsed.get('Unhandled'), list)
                and all(isinstance(p, str) for p in parsed['Unhandled'])):
            parsed['Attributes'] = [{'category': targets[0], 'phrase': match.group(1)}]
            parsed['Unhandled'] = [phrase for phrase in parsed.get('Unhandled', [])
                                   if phrase.strip().rstrip('.!?').casefold() not in
                                   (query.strip().rstrip('.!?').casefold(), match.group(1).casefold())]
    missing = list(preflight_missing)
    relations = parsed.get('Relations', [])
    if not isinstance(relations, list):
        missing.append('relation coverage not supplied by parser')
        relations = []
    relations = [_normalize_relation_scope(query, relation, targets, observer_input)
                 if isinstance(relation, dict) else relation for relation in relations]
    raw_relations = malformed_relations + list(relations)
    valid = []
    for relation in relations:
        if not isinstance(relation, dict):
            missing.append('parser produced a non-object relation entry')
            continue
        kind = relation.get('relation')
        phrase = relation.get('phrase')
        valid_quote = quoted(query, phrase)
        target_name = relation.get('target_category')
        anchor_name = relation.get('anchor_category', '')
        anchor_names = relation.get('anchor_categories', []) if kind == 'between' else [anchor_name]
        valid_quote = (valid_quote and isinstance(kind, str) and isinstance(target_name, str)
                       and bool(target_name.strip()))
        if anchor_name is not None and not isinstance(anchor_name, str):
            valid_quote = False
        if relation.get('anchor_categories') is not None and not isinstance(
                relation.get('anchor_categories'), list):
            valid_quote = False
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
            valid_quote = valid_quote and not re.search(
                r'\b(?:far|most|leftmost|rightmost|furthest|farthest)\b', phrase or '', re.I)
            if relation.get('reference_frame', 'observer') != 'observer':
                missing.append('object-intrinsic directional frame is unavailable: ' + phrase
                               if quoted(query, phrase)
                               else 'unsupported directional reference frame')
                valid_quote = False
        if kind in ('above', 'below'):
            valid_quote = valid_quote and bool(re.search(r'\b(above|over|below|under|underneath|beneath)\b', phrase, re.I))
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
    intrinsic_facing = intrinsic_facing_condition(query, facing.get('category', ''))
    if intrinsic_facing:
        facing = {**facing, 'phrase': intrinsic_facing['phrase'],
                  'group': intrinsic_facing['group'], 'reference_frame': 'intrinsic',
                  'orientation_part': intrinsic_facing['part']}
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
        elif re.fullmatch(r'(?:facing|looking\s+(?:at|toward))\s+(?:the\s+)?' +
                          re.escape(retrieval_category(facing['category'])) + r's', phrase.strip(), re.I):
            facing = {**facing, 'group': True}
        if intrinsic_facing:
            missing.append('object-intrinsic facing orientation is unavailable: ' + facing['phrase'])
    parsed['Observer'] = {'origin': origin, 'facing': facing}
    targets = parsed.get('Target', [])
    targets = [targets] if isinstance(targets, str) else targets
    recovered_observer_coreference = False
    if (not targets and _explicit_room_center_origin(query, parsed['Observer']) and
            facing.get('group') and re.search(r'\bthe correct one\b', query, re.I)):
        matching_relations = [relation for relation in parsed['Relations']
                              if relation.get('relation') in ('left', 'right') and
                              relation.get('scope') == 'side' and
                              retrieval_category(relation.get('target_category', '')) ==
                              retrieval_category(facing.get('category', '')) and
                              _complete_observer_coreference_clause(query, parsed['Observer'], relation)]
        if len(matching_relations) == 1:
            target_category = matching_relations[0]['target_category']
            targets = [target_category]
            parsed['Target'] = targets
            parsed['target_binding_recovery'] = {
                'phrase': 'the correct one', 'category': target_category,
                'basis': 'explicit room-center frame and matching facing-group directional relation',
            }
            recovered_observer_coreference = True
    target_coreferences = []
    if (len(targets) == 1 and retrieval_category(targets[0]) not in
            ('one', 'correct one', 'it', 'object', 'thing') and origin and facing):
        coreference_phrases = list(parsed.get('Unhandled', []))
        if recovered_observer_coreference:
            coreference_phrases.append('the correct one')
        for phrase in coreference_phrases:
            if not isinstance(phrase, str) or phrase.strip().rstrip('.!?').casefold() != 'the correct one':
                continue
            coref_quote = phrase.strip().rstrip('.!?')
            coref_start = query.casefold().find(coref_quote.casefold())

            def relation_follows_coreference(relation):
                relation_phrase = relation.get('phrase', '')
                relation_start = query.casefold().find(relation_phrase.casefold())
                if relation_start > coref_start:
                    return True
                relation_end = relation_start + len(relation_phrase)
                return (relation_start <= coref_start < relation_end and
                        bool(re.search(r'\b' + re.escape(relation['relation']) + r'\b',
                                       query[coref_start + len(coref_quote):relation_end], re.I)))

            represented_relation = next((relation for relation in parsed['Relations']
                if relation.get('relation') in ('left', 'right', 'front', 'back') and
                retrieval_category(relation.get('target_category', '')) == retrieval_category(targets[0]) and
                quoted(query, relation.get('phrase')) and
                relation_follows_coreference(relation)), None)
            if represented_relation:
                target_coreferences.append({'phrase': phrase, 'category': targets[0]})
    parsed['target_coreferences'] = target_coreferences
    attrs = parsed.get('Attributes')
    if not isinstance(attrs, list) or any(not isinstance(a, dict) for a in attrs):
        missing.append('attribute coverage not supplied by parser')
        attrs = []
    attrs = [dict(attribute) for attribute in attrs]
    attrs, recovered_attribute_missing = _recover_surface_attributes(query, parsed, attrs)
    contrast = relative_height_between(query, targets[0]) if len(targets) == 1 else None
    if contrast:
        for role, word in (('target', contrast['target_word']), ('reference', contrast['reference_word'])):
            if not any(a.get('phrase') == word and a.get('role') == role for a in attrs):
                attrs.append({'category': targets[0], 'phrase': word, 'role': role,
                              'binding_source': 'quoted contrast around the two between references'})
        parsed['relative_height_binding'] = contrast
    missing.extend(recovered_attribute_missing)
    raw_attrs = list(attrs)
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
        relation_anchor_categories = relation.get('anchor_categories', [])
        if not isinstance(relation_anchor_categories, list):
            relation_anchor_categories = []
        for name in relation_anchor_categories:
            if isinstance(name, str) and retrieval_category(name):
                bound_categories.add(retrieval_category(name))
    observer = parsed['Observer']
    for role in ('origin', 'facing'):
        entry = observer.get(role, {})
        name = entry.get('category', '') if isinstance(entry, dict) else ''
        if isinstance(name, str) and retrieval_category(name):
            bound_categories.add(retrieval_category(name))
    valid_attrs = []
    for attribute in attrs:
        source_bindings = _source_attribute_bindings(query, parsed, attribute.get('phrase', ''))
        source_binding = next(iter(source_bindings)) if len(source_bindings) == 1 else None
        if source_binding and retrieval_category(attribute.get('category', '')) in bound_categories:
            source_category, source_role = source_binding
            if (retrieval_category(attribute.get('category', '')) != source_category or
                    (attribute.get('role') and attribute.get('role') != source_role) or
                    attribute_role(parsed, attribute) is None):
                attribute['reported_category'] = attribute.get('category')
                attribute['reported_role'] = attribute.get('role')
                attribute.update(category=source_category, role=source_role,
                                 binding_source='direct quoted noun attachment')
        category_name = attribute.get('category')
        category = retrieval_category(category_name) if isinstance(category_name, str) else ''
        target_categories = parsed.get('Target', [])
        target_categories = [target_categories] if isinstance(target_categories, str) else target_categories
        reference_categories = parsed.get('Anchor', [])
        reference_categories = [reference_categories] if isinstance(reference_categories, str) else reference_categories
        target_categories = {retrieval_category(name) for name in target_categories if retrieval_category(name)}
        reference_categories = {retrieval_category(name) for name in reference_categories if retrieval_category(name)}
        for relation in parsed['Relations']:
            for key in ('anchor_category', 'anchor_categories'):
                names = relation.get(key, []) or []
                names = [names] if isinstance(names, str) else names
                names = names if isinstance(names, list) else []
                reference_categories.update(retrieval_category(name) for name in names
                                            if isinstance(name, str) and retrieval_category(name))
        for role in ('origin', 'facing'):
            entry = parsed['Observer'].get(role, {})
            name = entry.get('category', '') if isinstance(entry, dict) else ''
            if isinstance(name, str) and retrieval_category(name):
                reference_categories.add(retrieval_category(name))
        explicit_role = attribute.get('role')
        role_matches = ((explicit_role == 'target' and category in target_categories) or
                        (explicit_role == 'reference' and category in reference_categories))
        inferred_role_matches = ((not explicit_role and category in target_categories) or
                                 (not explicit_role and category in reference_categories))
        role_ambiguous = (not explicit_role and category in target_categories and
                          category in reference_categories)
        if (size_specs(attribute.get('phrase', ''), category) and
                category in target_categories and category in reference_categories and
                len(source_bindings) != 1 and len(list(re.finditer(
                    re.escape(attribute.get('phrase', '')), query, re.I))) == 1):
            role_ambiguous = True
        if (quoted(query, attribute.get('phrase')) and category in bound_categories and
                ((role_matches and not role_ambiguous) or
                 (inferred_role_matches and not role_ambiguous))):
            if not any(retrieval_category(item.get('category', '')) == category and
                       attribute_role(parsed, item) == attribute_role(parsed, attribute) and
                       item.get('phrase', '').casefold() == attribute.get('phrase', '').casefold()
                       for item in valid_attrs):
                valid_attrs.append(attribute)
        elif quoted(query, attribute.get('phrase')) and role_ambiguous:
            missing.append('attribute target/reference role is ambiguous: ' + attribute.get('phrase', ''))
        elif quoted(query, attribute.get('phrase')):
            missing.append('attribute category is not a supplied target/reference: ' + str(category_name))
            missing.append('appearance condition not represented: ' + attribute.get('phrase', ''))
    parsed['Attributes'] = valid_attrs
    if len(parsed['Attributes']) != len(attrs):
        missing.append('unquoted or unbound appearance attribute')
    for phrase in dict.fromkeys(attribute.get('phrase', '') for attribute in attrs):
        phrase_bindings = _source_attribute_bindings(query, parsed, phrase)
        for unit in attribute_units(phrase):
            bindings = phrase_bindings | _source_attribute_bindings(query, parsed, unit)
            for category, role in sorted(bindings):
                if not any(unit.casefold() in {part.casefold() for part in
                           attribute_units(attribute.get('phrase', ''))}
                           and retrieval_category(attribute.get('category', '')) == category
                           and attribute_role(parsed, attribute) == role
                           for attribute in valid_attrs):
                    missing.append('appearance condition role not represented: '
                                   + unit + ' (' + role + ' ' + category + ')')
    covered = ' '.join(a['phrase'] for a in parsed['Attributes'])
    for word in re.findall(r'\b(smallest|largest|biggest|tallest|shortest|highest|lowest|'
                           r'larger|smaller|bigger|taller|shorter|longer|large|small|big|little|'
                           r'skinny|narrow|wide|thin|thick|dark|light|black|white|red|blue|'
                           r'green|gr[ae]y|brown|yellow|orange|purple|pink|rainbow|wooden|'
                           r'mounted|folded|vertical)\b', query, re.I):
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
        bound_coreference_quotes = {item['phrase'].strip().rstrip('.!?').casefold()
                                    for item in target_coreferences}
        parsed['Unhandled'] = [p for p in parsed['Unhandled']
                               if p.strip().rstrip('.!?').casefold() not in bound_coreference_quotes]
        if recovered_observer_coreference:
            parsed['Unhandled'] = [p for p in parsed['Unhandled']
                                   if p.strip().casefold() != query.strip().casefold()]
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
        from seeground.grounding.v4_policy import v6_enabled
        if v6_enabled():
            from seeground.query.query_graph import retain_positive_predicates
            retain_positive_predicates(query, parsed)
        else:
            parsed['Relations'] = []
    if any(r['relation'] in ('left', 'right', 'front', 'back') for r in parsed['Relations']):
        observer = parsed['Observer']
        if not observer['origin'] or not observer['facing']:
            missing.append('observer origin or facing direction is unspecified')
    if re.search(r'\b(smallest|largest|biggest|tallest|shortest|highest|lowest|'
                 r'larger|smaller|bigger|taller|shorter)\b', query, re.I):
        scope_match = re.search(r'\bof the two(?:\s+[a-z-]+)?\b', query, re.I)
        scope_specs = (size_conditions(parsed, targets[0])
                       if len(targets) == 1 and parsed['query_syntax']['backend'] == 'spacy' else
                       [spec for attr in parsed.get('Attributes', [])
                        for spec in size_specs(attr.get('phrase', ''), targets[0] if len(targets) == 1 else '',
                                               query=query)])
        if scope_match and not any(spec.get('required_member_count') == 2 for spec in scope_specs):
            missing.append('size comparison scope not bound: ' + scope_match.group())
    direction_query = re.sub(r'\bright\s+(?:above|over|below|under|underneath|beneath)\b', '', query, flags=re.I)
    if re.search(r'\b(left|right|ahead|behind)\b', direction_query, re.I) and not any(
            r['relation'] in ('left', 'right', 'front', 'back') for r in parsed['Relations']):
        missing.append('direction condition not represented')
    for pattern, kinds, label in ((r'\b(nearest|closest|farthest|furthest)\b', ('nearest','farthest'), 'distance extreme'),
                                  (r'\b(next to|beside|near)\b', ('near',), 'proximity')):
        if re.search(pattern, query, re.I) and not any(r['relation'] in kinds for r in parsed['Relations']):
            missing.append(label + ' condition not represented')
    for match in re.finditer(IMMEDIATE_DIRECTION_PATTERN, query, re.I):
        missing.append('immediate proximity has no calibrated predicate: ' + match.group())
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
    from seeground.grounding.v4_policy import v6_enabled
    if v6_enabled():
        from seeground.query.query_graph import retain_positive_predicates
        retain_positive_predicates(query, parsed)
    parsed['condition_records'] = _condition_records(query, parsed, raw_relations, raw_attrs)
    if v6_enabled():
        from seeground.query.query_graph import structure
        previous = parsed.get('query_structure', {}).get('conditions', [])
        parsed['query_structure'] = structure(query, parsed)
        present = {(c.get('condition_type'), c.get('source_quote')) for c in parsed['query_structure']['conditions']}
        parsed['query_structure']['conditions'] += [c for c in previous
            if c.get('polarity') != 'positive' and (c.get('condition_type'), c.get('source_quote')) not in present]
        for index, condition in enumerate(parsed['query_structure']['conditions']):
            condition['condition_id'] = f'c{index}'
        parsed['diagnostic_missing'].extend('query modifier not represented: ' + quote
            for quote in parsed['query_structure']['uncovered_modifiers'])
    return parsed


def refresh_condition_records(query, parsed):
    """Rebuild the audit ledger after another parser operation updates diagnostics."""
    parsed['condition_records'] = _condition_records(
        query, parsed, parsed.get('Relations') or [], parsed.get('Attributes') or [])
    from seeground.grounding.v4_policy import v6_enabled
    if v6_enabled():
        from seeground.query.query_graph import structure
        previous = parsed.get('query_structure', {}).get('conditions', [])
        parsed['query_structure'] = structure(query, parsed)
        present = {(c.get('condition_type'), c.get('source_quote')) for c in parsed['query_structure']['conditions']}
        parsed['query_structure']['conditions'] += [c for c in previous
            if (c.get('condition_type'), c.get('source_quote')) not in present]
        for index, condition in enumerate(parsed['query_structure']['conditions']):
            condition['condition_id'] = f'c{index}'
    return parsed


def observer_frames(parsed, objects, room_center):
    observer = parsed.get('Observer', {})
    if not isinstance(observer, dict):
        observer = {}
    origin = observer.get('origin', {})
    facing = observer.get('facing', {})
    if facing.get('reference_frame') == 'intrinsic':
        return [], ['object-intrinsic facing orientation is unavailable: ' + facing.get('phrase', '')]
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
        frames.append(_language_frame_record({
            'origin_id': start['bbox_id'], 'facing_ids': [o['bbox_id'] for o in group],
            'origin': list(start['bbox_3d'][:3]), 'forward': forward.tolist(),
            'right': right.tolist()}, parsed))
    return frames, missing


def _language_frame_record(frame, parsed):
    """Annotate a geometric frame as a language reference interpretation.

    The active renderer may use any inspection camera later.  It must not be
    mistaken for the speaker frame used by left/right/front/back predicates,
    so the distinction is explicit in every frame record.
    """
    observer = parsed.get('Observer', {}) if isinstance(parsed, dict) else {}
    observer = observer if isinstance(observer, dict) else {}
    origin = observer.get('origin', {})
    facing = observer.get('facing', {})
    origin = origin if isinstance(origin, dict) else {}
    facing = facing if isinstance(facing, dict) else {}
    assumptions = [
        'forward is the horizontal vector from the language origin to the facing object/group center',
        'right is the horizontal cross product of forward and world-up',
        'this frame describes language relations; an inspection camera does not change it',
    ]
    if origin.get('kind') == 'room_center':
        assumptions.append('room-center origin is approximated by the scene point-cloud AABB center')
    elif origin.get('kind') == 'category':
        assumptions.append('named origin is approximated by the bound reference object center')
    if facing.get('group'):
        assumptions.append('group-facing direction is approximated by the mean center of the bound group')
    return {
        **frame,
        'frame_type': 'language_reference',
        'source_quotes': {
            'origin': origin.get('phrase', ''),
            'facing': facing.get('phrase', ''),
            'viewpoint': parsed.get('Viewpoint', '') if isinstance(parsed.get('Viewpoint', ''), str) else '',
        },
        'assumptions': assumptions,
        'capture_camera': {
            'role': 'inspection_only',
            'affects_language_frame': False,
            'source': 'active appearance renderer when requested',
        },
    }


def viewpoint_evidence(parsed, diagnosis=None):
    """Return structured viewpoint evidence without inventing a speaker pose.

    Orientation-only descriptions and object-intrinsic front/back requests are
    deliberately unknown.  A camera used to inspect RGB/point-cloud evidence
    is represented separately from the language frame hypotheses.
    """
    parsed = parsed if isinstance(parsed, dict) else {}
    observer = parsed.get('Observer', {})
    observer = observer if isinstance(observer, dict) else {}
    origin = observer.get('origin', {})
    facing = observer.get('facing', {})
    origin = origin if isinstance(origin, dict) else {}
    facing = facing if isinstance(facing, dict) else {}
    if not (origin.get('category') or origin.get('phrase') or
            origin.get('kind') not in (None, '', 'unspecified')):
        origin = {}
    if not (facing.get('category') or facing.get('phrase') or facing.get('group')):
        facing = {}
    relations = parsed.get('Relations', [])
    relations = relations if isinstance(relations, list) else []
    directional = [item for item in relations if isinstance(item, dict)
                   and item.get('relation') in ('left', 'right', 'front', 'back')]
    source_quotes = {
        'origin': origin.get('phrase', ''),
        'facing': facing.get('phrase', ''),
        'viewpoint': parsed.get('Viewpoint', '') if isinstance(parsed.get('Viewpoint', ''), str) else '',
        'direction_relations': [item.get('phrase', '') for item in directional],
    }
    raw_hypotheses = diagnosis.get('hypotheses', []) if isinstance(diagnosis, dict) else []
    hypotheses = []
    for index, hypothesis in enumerate(raw_hypotheses):
        if not isinstance(hypothesis, dict):
            continue
        frame = hypothesis.get('observer')
        if frame is None:
            continue
        hypotheses.append({
            'hypothesis_id': index,
            'frame': _language_frame_record(frame, parsed),
            'reference_bindings': dict(hypothesis.get('references', {})),
            'candidate_ids': list(hypothesis.get('candidate_ids', [])),
            'conditions_share_frame': True,
        })
    missing = []
    assumptions = []
    orientation_only = bool(facing) and not bool(origin)
    if orientation_only:
        missing.append('description constrains orientation only; observer origin is unspecified')
        assumptions.append('facing direction alone does not establish a language origin')
    if directional and not hypotheses:
        missing.append('language reference frame is unavailable')
    diagnostics = (diagnosis.get('missing', []) if isinstance(diagnosis, dict)
                   else parsed.get('diagnostic_missing', []))
    intrinsic_missing = [item for item in diagnostics if isinstance(item, str)
                         and item.startswith(('object-intrinsic directional frame is unavailable',
                                              'object-intrinsic facing orientation is unavailable'))]
    missing.extend(intrinsic_missing)
    if directional and hypotheses:
        assumptions.append('all directional conditions are evaluated within each single frame hypothesis')
    if not directional and (origin or facing or source_quotes['viewpoint']):
        assumptions.append('observer clues are retained as context; no directional predicate is certified')
    status = ('known' if hypotheses and not missing else
              'unknown' if directional or origin or facing or intrinsic_missing else 'not_applicable')
    return {
        'dimension': 'viewpoint',
        'status': status,
        'language_reference_frames': hypotheses,
        'frame_hypotheses_are_alternatives': True,
        'conditions_share_hypothesis': True,
        'observer': {'origin': origin, 'facing': facing},
        'source_quotes': source_quotes,
        'assumptions': assumptions,
        'missing': list(dict.fromkeys(missing)),
        'capture_camera': {
            'role': 'inspection_only',
            'affects_language_frame': False,
            'policy': 'appearance rendering camera is independent of the speaker/reference frame',
        },
    }


def attributes_for(parsed, category, role=None, geometry=False):
    return [a.get('geometry_attribute', a['phrase']) if geometry else a['phrase']
            for a in parsed['Attributes']
            if retrieval_category(a['category']) == retrieval_category(category)
            and (role is None or attribute_role(parsed, a) == role)]


def size_conditions(parsed, category, role='target'):
    conditions = []
    query = parsed.get('source_query', parsed.get('query_rewrite', {}).get('query', ''))
    for attribute in parsed.get('Attributes', []):
        if (retrieval_category(attribute.get('category', '')) != retrieval_category(category)
                or attribute_role(parsed, attribute) != role):
            continue
        if attribute.get('polarity') == 'exclude':
            continue
        for spec in size_specs(attribute.get('phrase', ''), category, query=query,
                               role=role, source=attribute):
            from seeground.query.query_syntax import comparison_hint
            hint = comparison_hint(parsed, spec['word'], category)
            from seeground.grounding.v4_policy import enabled
            if enabled() and hint:
                spec['syntax_attachment'] = hint['attachment']
            if hint:
                spec.update(source_phrase=hint['source_phrase'], source_span=hint['source_span'],
                            syntax_scope_type=hint['scope_type'])
                if hint.get('required_member_count'):
                    spec['required_member_count'] = hint['required_member_count']
            if attribute.get('source_phrase') == query.strip() and attribute.get('comparison_scope'):
                spec.update(scope_kind=attribute['comparison_scope'],
                            scope_phrase=attribute['scope_phrase'])
                if attribute.get('required_member_count'):
                    spec['required_member_count'] = attribute['required_member_count']
            # A post-noun relation can identify a local comparison subset.  A
            # category match alone does not bind that subset, even if the
            # parser omitted the relation or supplied a scene-wide pool.
            if (spec.get('scope_kind') == 'full_category' and spec['form'] != 'superlative'
                    and not (enabled() and hint and hint['attachment'] == 'independent')):
                noun = re.escape(retrieval_category(category))
                dependency = re.search(
                    r'\b' + re.escape(spec['word']) + r'\s+(?:[a-z-]+\s+)*?' + noun +
                    r's?\s+(?P<scope>(?:that|which|on|in|at|near|beside|between|behind|'
                    r'above|below|across|next\s+to|with)\b[^.!?;,]*)', query, re.I)
                if dependency:
                    spec.update(scope_kind='unbound_explicit_scope',
                                scope_phrase=dependency['scope'].strip(),
                                scope_binding_status='unknown',
                                scope_binding_reason=('comparison modifier attachment is ambiguous'
                                    if hint else 'comparison scope depends on an unbound quoted relation'))
            if hint and hint['attachment'] == 'restrictive_subset':
                spec.update(scope_kind='relation_subset',
                            scope_relation_quotes=[m['text'] for m in hint['modifiers']],
                            scope_phrase=hint['scope_phrase'])
                if hint.get('required_member_count'):
                    spec['required_member_count'] = hint['required_member_count']
            if (enabled() and hint and hint['attachment'] == 'ambiguous' and hint.get('modifiers')
                    and not attribute.get('scene_scope_assumption')
                    and not re.search(r'\band\s+(?:above|below|near|behind|in front)\b', query, re.I)):
                spec.update(scope_kind='unbound_explicit_scope',
                            scope_relation_quotes=[m['text'] for m in hint['modifiers']],
                            scope_binding_reason='independent condition and comparison-subset attachments remain ambiguous')
                from seeground.grounding.v4_policy import v5_enabled
                if v5_enabled('scope'):
                    spec['scope_hypotheses'] = [
                        {'kind': 'full_category', 'interpretation': 'size with independent quoted relation'},
                        {'kind': 'relation_subset', 'interpretation': 'size within quoted relation members'}]
            if enabled('size') and spec['metric'] == 'volume_m3':
                if retrieval_category(category) in ('table', 'desk'):
                    spec.update(metric='oriented_footprint_m2',
                                metric_reason='table/desk horizontal enclosing area proxy; not exact tabletop area',
                                tentative_metric=True,
                                metric_choice_source='category surface structure; declared v4 policy')
                else:
                    spec.update(metric='oriented_volume_m3',
                                metric_reason='observed oriented volume proxy; completeness and alternative rankings retained')
            if enabled('size') and spec['metric'] == 'height_m' and retrieval_category(category) in ('desk', 'table'):
                spec.update(metric='tabletop_height_m',
                            metric_reason='fitted horizontal tabletop above a supplied local floor; total span is not tabletop height')
            from seeground.grounding.v4_policy import v5_enabled
            if v5_enabled('measurements') and retrieval_category(category) in ('mirror', 'whiteboard', 'book') and spec['metric'] in ('volume_m3', 'oriented_volume_m3'):
                spec.update(metric='body_plane_area_m2', tentative_metric=True,
                    metric_reason='observed dominant body-plane area; overall-size semantics remain tentative',
                    metric_choice_source='v5 measured planar structure')
            choice = parsed.get('_v4_metric_choices', {}).get('|'.join((role, retrieval_category(category), spec['word'])))
            if enabled('size') and choice and spec['metric'] in ('oriented_volume_m3', 'oriented_footprint_m2'):
                spec.update(metric=choice['metric'], metric_reason=choice['reason'], tentative_metric=True,
                            metric_choice_source='structured internal visual observation')
            from seeground.grounding.v4_policy import v7_enabled
            if v7_enabled():
                from seeground.grounding.v7_evidence import box_size_spec
                spec = box_size_spec(spec, category, query)
                if retrieval_category(category) == 'book' and choice and choice['metric'] == 'body_plane_area_m2':
                    spec.update(metric='box_plane_area_m2', metric_choice_source='visual single-book plane interpretation',
                                metric_reason=choice['reason'])
                if parsed.get('category_recovery'):
                    corner = re.search(r'(?:located\s+)?in\s+(?:a|the)\s+corner\b', query, re.I)
                    if corner:
                        spec.update(scope_kind='relation_subset', scope_relation_quotes=[corner[0]],
                                    scope_phrase=corner[0])
            if not any(c['word'] == spec['word'] and c['role'] == role for c in conditions):
                conditions.append(spec)
    return conditions


def appearance_attributes_for(parsed, category, role='target'):
    """Size has its own measured obligation; keep the other quoted modifiers."""
    result = []
    from seeground.grounding.v4_policy import v7_enabled
    for phrase in attributes_for(parsed, category, role=role):
        negative = next((a for a in parsed['Attributes'] if a['phrase'] == phrase
            and retrieval_category(a['category']) == retrieval_category(category)
            and attribute_role(parsed, a) == role and a.get('polarity') == 'exclude'), None)
        if v7_enabled() and negative:
            result.append('not ' + phrase)
            continue
        specs = size_specs(phrase, category, role=role)
        if not specs:
            result.append(phrase)
            continue
        remainder = phrase
        for spec in specs:
            remainder = re.sub(r'\b' + re.escape(spec['word']) + r'\b', '', remainder, flags=re.I)
        remainder = re.sub(r'\b(?:the|a|an)\b', '', remainder, flags=re.I)
        remainder = re.sub(r'\b' + re.escape(category) + r's?\b', '', remainder, flags=re.I)
        remainder = ' '.join(remainder.split()).strip(' ,')
        if remainder:
            result.append(remainder)
    return list(dict.fromkeys(result))


def appearance_checks_for(checks, parsed, category, role='target'):
    """Retain old visual traces while size moves to its measured obligation."""
    expected = required_visual_attributes(appearance_attributes_for(parsed, category, role))
    original = required_visual_attributes(attributes_for(parsed, category, role))
    moved = set(original) - set(expected)
    return [{**check, 'attribute_checks': [item for item in check.get('attribute_checks', [])
             if not isinstance(item, dict) or item.get('attribute') not in moved]}
            for check in checks]



def evaluate_scope_hypotheses(parsed, objects, category, spec, checks, measurements, candidate_ids, scope_bound=True):
    """Every possible attachment keeps its own complete member proof."""
    hypotheses = []
    for hypothesis in spec['scope_hypotheses']:
        alternative = {**spec, 'scope_kind': hypothesis['kind']}
        alternative.pop('scope_hypotheses', None)
        if hypothesis['kind'] == 'full_category':
            alternative['scope_relation_quotes'] = []
        pool, binding = size_condition_scope(parsed, objects, category, alternative, checks, spec.get('role', 'target'))
        proof = evaluate_size_condition(alternative, measurements, [o['bbox_id'] for o in pool],
            scope_bound=scope_bound and binding['status'] == 'bound',
            candidate_ids=candidate_ids, scope_source=binding['source'],
            scope_unknown_ids=binding.get('unknown_ids', []))
        proof.update(interpretation=hypothesis['interpretation'], member_binding=binding)
        hypotheses.append(proof)
    combined = deepcopy(hypotheses[0])
    for item in combined['candidate_evidence']:
        states = [next((p['status'] for p in h['candidate_evidence'] if p['candidate_id'] == item['candidate_id']),
                       'unknown') for h in hypotheses]
        item['status'] = states[0] if len(set(states)) == 1 else 'unknown'
        item['reason'] = 'all possible scope interpretations agree' if item['status'] != 'unknown' else 'scope interpretations remain unresolved'
    bound = all(h['scope_status'] == 'bound' for h in hypotheses)
    known = (bound and all(h['status'] == 'known' for h in hypotheses)
             and all(e['status'] != 'unknown' for e in combined['candidate_evidence']))
    combined.update(scope_hypotheses=hypotheses, scope_status='bound' if bound else 'unknown',
        status='known' if known else 'unknown', selected_ids=[], tentative_selected_ids=[],
        reason=None if known else 'comparison attachment hypotheses require binding',
        next_evidence=[{'dimension': 'distance' if re.search(r'\b(near|by|next to|beside)\b', q) else 'position',
            'source_quote': q, 'reason': 'bind reference and verify every member of quoted comparison subset'}
            for q in spec.get('scope_relation_quotes', [])])
    selected = {e['candidate_id'] for e in combined['candidate_evidence'] if e['status'] == 'pass'}
    if known:
        combined['selected_ids'] = sorted(selected)
    else:
        choices = [set(h.get('tentative_selected_ids') or h.get('selected_ids', [])) for h in hypotheses
                   if h.get('tentative_selected_ids') or h.get('selected_ids')]
        if choices and len(set.intersection(*choices)) == 1:
            combined['tentative_selected_ids'] = sorted(set.intersection(*choices))
    combined['unknown_measurement_ids'] = sorted({oid for h in hypotheses for oid in h['unknown_measurement_ids']})
    combined['unresolved_hypotheses'] = [index for index, h in enumerate(hypotheses)
                                        if h['scope_status'] != 'bound' or h['status'] != 'known']
    if spec['original_quote'] in parsed.get('_v4_invalidated_conditions', []):
        combined.update(status='unknown', selected_ids=[], tentative_selected_ids=[],
                        reason='supported backtrack requires fresh scope or measurement evidence')
        for proof in combined['candidate_evidence']:
            proof['status'] = 'unknown'
    return combined


def _v5_relation_subset(parsed, pool, objects, category, quotes, checks):
    """Keep every supplied member and every quoted predicate until proved false."""
    from seeground.geometry.v4_relations import verified_members
    evidence, by_quote = [], {}
    for quote in quotes:
        relation_quote = re.sub(r'^(?:that|which)\s+(?:is|are)\s+', '', quote, flags=re.I)
        from seeground.grounding.v4_policy import enabled
        members = verified_members(parsed, relation_quote, category) if enabled('relations') else {}
        if members:
            states = {o['bbox_id']: members.get(o['bbox_id'], {}).get('status', 'unknown') for o in pool}
            evidence.extend(members.values())
        else:
            relations = [r for r in parsed.get('Relations', [])
                         if r.get('relation') in ('above', 'below')
                         and retrieval_category(r.get('target_category', '')) == retrieval_category(category)
                         and relation_quote.casefold() in r.get('phrase', '').casefold()]
            states = {o['bbox_id']: 'unknown' for o in pool}
            if len(relations) == 1:
                relation = relations[0]
                anchors = match_retrieval_categories([relation['anchor_category']], objects)
                anchors, anchor_scope = _attribute_comparison_pool(
                    parsed, anchors, relation['anchor_category'], checks, 'reference')
                for obj in pool:
                    alternatives = [{
                        'anchor_id': anchor['bbox_id'],
                        'status': _vertical_status(_pair_geometry(obj, anchor), relation['relation'])}
                        for anchor in anchors]
                    values = [p['status'] for p in alternatives]
                    # A definite reference may have several possible bindings.
                    # All must agree; one failing bed cannot reject its pillow.
                    states[obj['bbox_id']] = ('fail' if values and all(s == 'fail' for s in values) else
                        'pass' if values and all(s == 'pass' for s in values)
                        and (not anchor_scope or anchor_scope['status'] == 'bound') else 'unknown')
                    evidence.append({'candidate_id': obj['bbox_id'], 'status': states[obj['bbox_id']],
                        'source_quote': quote, 'anchor_hypotheses': alternatives,
                        'relation': relation['relation']})
            else:
                evidence.extend({'candidate_id': oid, 'status': 'unknown', 'source_quote': quote,
                                 'reason': 'quoted subset predicate needs reference/member evidence'}
                                for oid in states)
        by_quote[quote] = states
    member_states = {o['bbox_id']: ('fail' if any(states[o['bbox_id']] == 'fail' for states in by_quote.values())
        else 'pass' if by_quote and all(states[o['bbox_id']] == 'pass' for states in by_quote.values())
        else 'unknown') for o in pool}
    members = [o for o in pool if member_states[o['bbox_id']] != 'fail']
    unknown = sorted(oid for oid, state in member_states.items() if state == 'unknown')
    return members, {'status': 'unknown' if unknown or not quotes else 'bound',
        'selected_ids': sorted(o['bbox_id'] for o in members), 'unknown_ids': unknown,
        'source': 'quoted relation comparison subset', 'member_evidence': evidence,
        'member_states': {str(oid): state for oid, state in member_states.items()},
        'predicate_member_states': {quote: {str(oid): state for oid, state in states.items()}
                                    for quote, states in by_quote.items()},
        'next_evidence': [{'source_quote': quote, 'candidate_ids': sorted(oid for oid, state in states.items()
                           if state == 'unknown'), 'reason': 'bind quoted reference and verify unknown subset members'}
                          for quote, states in by_quote.items() if 'unknown' in states.values()],
        **({'reason': 'comparison subset contains unresolved relation members'} if unknown else {})}


def size_condition_scope(parsed, objects, category, spec, checks=(), role='target'):
    """Every predicate binds its pool from the source, before target filtering."""
    pool = match_retrieval_categories([category], objects)
    category_pool_missing = not pool
    record = {'status': 'bound', 'selected_ids': sorted(o['bbox_id'] for o in pool),
              'source': 'complete supplied category pool', 'unknown_ids': []}
    if bool(spec.get('scope_modifiers')):
        subset_parsed = {**parsed, 'Attributes': [{'category': category, 'role': role,
                          'phrase': spec.get('scope_modifiers', '')}]}
        pool, subset = _attribute_comparison_pool(subset_parsed, pool, category, checks, role)
        if subset:
            record.update(subset)
        record['source'] = 'explicit quoted appearance subset'
    from seeground.grounding.v4_policy import v5_enabled
    if spec.get('scope_kind') == 'relation_subset' and v5_enabled('scope'):
        appearance_unknown = set(record['unknown_ids'])
        pool, subset = _v5_relation_subset(parsed, pool, objects, category,
                                          spec.get('scope_relation_quotes', []), checks)
        record.update(subset, required_member_count=spec.get('required_member_count'))
        record['unknown_ids'] = sorted(set(record['unknown_ids']) |
                                      (appearance_unknown & set(record['selected_ids'])))
        if record['unknown_ids']:
            record['status'] = 'unknown'
    elif spec.get('scope_kind') == 'relation_subset':
        original_pool = list(pool)
        proofs = []
        for quote in spec.get('scope_relation_quotes', []):
            relation_quote = re.sub(r'^(?:that|which)\s+(?:is|are)\s+', '', quote, flags=re.I)
            from seeground.grounding.v4_policy import enabled
            if enabled('relations'):
                from seeground.geometry.v4_relations import verified_members
                members = verified_members(parsed, relation_quote, category)
                if members:
                    states = {o['bbox_id']: members.get(o['bbox_id'], {}).get('status', 'unknown') for o in pool}
                    proofs.extend(members.values())
                    pool = [o for o in pool if states[o['bbox_id']] != 'fail']
                    unknown = [oid for oid, state in states.items() if state == 'unknown']
                    if unknown:
                        record.update(status='unknown', unknown_ids=unknown,
                                      reason='comparison subset contains unresolved relation members')
                        break
                    continue
            relations = [r for r in parsed.get('Relations', [])
                         if r.get('relation') in ('above', 'below')
                         and retrieval_category(r.get('target_category', '')) == retrieval_category(category)
                         and relation_quote.casefold() in r.get('phrase', '').casefold()]
            if len(relations) != 1:
                record.update(status='unknown', unknown_ids=[o['bbox_id'] for o in pool],
                              reason='quoted comparison subset relation is unsupported or ambiguous')
                break
            relation = relations[0]
            anchor_category = relation['anchor_category']
            anchors = match_retrieval_categories([anchor_category], objects)
            anchors, anchor_scope = _attribute_comparison_pool(
                parsed, anchors, anchor_category, checks, 'reference')
            if len(anchors) != 1 or (anchor_scope and anchor_scope['status'] != 'bound'):
                record.update(status='unknown', unknown_ids=[o['bbox_id'] for o in pool],
                              reason='comparison subset reference is not uniquely bound')
                break
            states = {o['bbox_id']: _vertical_status(_pair_geometry(o, anchors[0]), relation['relation'])
                      for o in pool}
            proofs.extend({'candidate_id': oid, 'status': state, 'source_quote': quote,
                           'anchor_id': anchors[0]['bbox_id'], 'relation': relation['relation']}
                          for oid, state in states.items())
            pool = [o for o in pool if states[o['bbox_id']] != 'fail']
            unknown = [oid for oid, state in states.items() if state == 'unknown']
            if unknown:
                record.update(status='unknown', unknown_ids=unknown,
                              reason='comparison subset contains unresolved relation members')
                break
        if not spec.get('scope_relation_quotes'):
            record.update(status='unknown', unknown_ids=[o['bbox_id'] for o in original_pool],
                          reason='comparison subset has no quoted relation')
        record.update(selected_ids=sorted(o['bbox_id'] for o in pool),
                      source='quoted relation comparison subset', member_evidence=proofs)
        from seeground.grounding.v4_policy import v5_enabled
        if v5_enabled('scope'):
            record['member_states'] = {str(o['bbox_id']): next((p['status'] for p in proofs
                if p['candidate_id'] == o['bbox_id']), 'unknown') for o in original_pool}
            record['required_member_count'] = spec.get('required_member_count')
    category_scope = size_category_scope(pool)
    if category_scope:
        record.update(category_scope)
    if spec.get('scope_kind') == 'unbound_explicit_scope':
        record.update(status='unknown', unknown_ids=record['selected_ids'],
                      source='unbound quoted comparison scope',
                      reason=spec.get('scope_binding_reason') or
                          'quoted comparison scope is not supported or fully bound')
    generation = parsed.get('candidate_generation', {})
    if (role == 'target' and isinstance(generation, dict) and
            generation.get('source') == 'broad_fallback' and generation.get('status') != 'bound'):
        record.update(status='unknown', unknown_ids=record['selected_ids'],
                      source='unverified broad fallback',
                      reason='broad fallback does not bind a size comparison category')
    if spec.get('required_member_count') and len(pool) != spec['required_member_count']:
        record.update(status='unknown', unknown_ids=record['selected_ids'],
                      reason=('explicit pair comparison requires exactly two bound instances'
                              if spec['required_member_count'] == 2 else
                              'explicit comparison requires the quoted number of bound instances'))
    if not pool:
        record['status'] = 'unknown'
        if category_pool_missing:
            record.update(source='target/reference category recall is missing',
                          reason='category recall/binding missing; comparison pool is empty')
        else:
            record.setdefault('reason', 'comparison pool is empty')
    key = json.dumps([role, category, spec['word'], spec.get('scope_kind'),
                      spec.get('scope_phrase'), spec.get('scope_modifiers')], sort_keys=True)
    bindings = parsed.setdefault('_size_scope_bindings', {})
    frozen = bindings.get(key)
    if frozen is None and record['status'] == 'bound':
        bindings[key] = dict(record)
    elif frozen is not None:
        by_id = {o['bbox_id']: o for o in objects}
        changed = (record['status'] != 'bound' or
                   record['selected_ids'] != frozen['selected_ids'])
        proposed = dict(record)
        pool = [by_id[oid] for oid in frozen['selected_ids'] if oid in by_id]
        record = {**frozen, 'frozen': True}
        if changed or len(pool) != len(frozen['selected_ids']):
            record.update(status='unknown', unknown_ids=frozen['selected_ids'],
                          reason='bound size comparison membership changed or became unknown',
                          proposed_binding=proposed)
            from seeground.grounding.v4_policy import enabled
            if enabled():
                parsed.setdefault('_v4_binding_changes', {})[key] = {
                    'source_quote': spec['original_quote'], 'old_ids': frozen['selected_ids'],
                    'new_ids': proposed['selected_ids'], 'status': proposed['status'],
                    'evidence': proposed.get('member_evidence', [])}
    return pool, record


def _checks_for_role(parsed, checks, category, role):
    language_role = 'target' if role == 'target' else 'reference'
    legacy_role = attribute_role(parsed, {'category': category})
    return [check for check in checks
            if retrieval_category(check.get('category', '')) == retrieval_category(category)
            and (check.get('role') == role or
                 (language_role == 'reference' and check.get('role') == 'reference') or
                 ('role' not in check and legacy_role == language_role))]


def _box_metrics(obj):
    x, y, z, width, length, height = map(float, obj['bbox_3d'])
    short_extent = min(width, length)
    long_extent = max(width, length)
    reliability = 'usable' if min(width, length, height) > 1e-9 else 'unreliable'
    return {'volume_m3': width * length * height, 'footprint_m2': width * length,
            'height_m': height, 'center_z_m': z,
            'bottom_z_m': z - height / 2., 'top_z_m': z + height / 2.,
            'extent_x_m': width, 'extent_y_m': length, 'extent_z_m': height,
            'horizontal_short_extent_m': short_extent,
            'horizontal_long_extent_m': long_extent,
            'horizontal_aspect_ratio': (long_extent / short_extent
                                        if short_extent > 1e-9 else None),
            'height_to_horizontal_long_ratio': (height / long_extent
                                                if long_extent > 1e-9 else None),
            'source': 'annotated_box',
            'measurement_reliability': reliability}


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
                    'relation': 'between', 'passes': None, 'status': 'unknown',
                    'reference_policy': 'finite wall projection',
                    'reason': 'between needs exactly one wall with a valid fitted vertical plane',
                    'wall_fits': {str(w['bbox_id']): w.get('wall_geometry') for w in walls},
                    'center_point_comparison': center_comparison}
        from seeground.geometry.wall_geometry import wall_reference_point
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
                'status': 'fail',
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
    endpoint_tolerance = POSITION_BOUNDARY_TOLERANCE_M / length
    outside_segment = projection_t < -endpoint_tolerance or projection_t > 1.0 + endpoint_tolerance
    near_endpoint = (-endpoint_tolerance <= projection_t <= endpoint_tolerance or
                     1.0 - endpoint_tolerance <= projection_t <= 1.0 + endpoint_tolerance)
    lateral_boundary = POSITION_BOUNDARY_TOLERANCE_M
    outside_lateral = perpendicular_distance > lateral_tolerance + lateral_boundary
    near_lateral_boundary = abs(perpendicular_distance - lateral_tolerance) <= lateral_boundary
    if outside_segment or outside_lateral:
        status = 'fail'
        reason = None
    elif near_endpoint or near_lateral_boundary:
        status = 'unknown'
        reason = ('target is near a between-corridor endpoint boundary' if near_endpoint else
                  'target is near the between-corridor lateral boundary')
    else:
        status = 'pass'
        reason = None
    within_segment = 0.0 <= projection_t <= 1.0
    record = {'target_id': target['bbox_id'], 'anchor_ids': [anchor_a['bbox_id'], anchor_b['bbox_id']],
              'relation': 'between', 'projection_t': projection_t,
              'perpendicular_distance_m': perpendicular_distance,
              'lateral_tolerance_m': lateral_tolerance,
              'within_segment': within_segment,
              'passes': True if status == 'pass' else False if status == 'fail' else None,
              'status': status,
              'reference_center_distance_m': float(np.linalg.norm(a_box[:2] - b_box[:2])),
              'reference_endpoint_distance_m': length, **wall_record}
    if reason:
        record['reason'] = reason
    return record


def _vertical_match(pair, relation):
    return _vertical_status(pair, relation) == 'pass'


def _vertical_status(pair, relation):
    """Classify a vertical predicate without forcing near-boundary geometry."""
    gap = pair['horizontal_gap_m']
    if gap > HORIZONTAL_ALIGNMENT_GAP_M + POSITION_BOUNDARY_TOLERANCE_M:
        return 'fail'
    if gap > HORIZONTAL_ALIGNMENT_GAP_M:
        return 'unknown'
    if relation == 'above':
        delta = pair['center_z_delta_m']
        clearance = pair['target_bottom_minus_anchor_top_m']
    elif relation == 'below':
        delta = -pair['center_z_delta_m']
        clearance = pair['anchor_bottom_minus_target_top_m']
    else:
        return 'unknown'
    # Coincident vertical centres are a definite non-match for an above/below
    # predicate.  Only a small non-zero displacement near the boundary is
    # uncertain; treating exact alignment as unknown would retain overlapping
    # candidates and make a clear vertical failure look unresolved.
    if abs(delta) <= 1e-9:
        return 'fail'
    if delta < 0.0:
        return 'unknown' if delta >= -POSITION_BOUNDARY_TOLERANCE_M else 'fail'
    if delta <= POSITION_BOUNDARY_TOLERANCE_M:
        return 'unknown'
    # Separation is sufficient evidence, but overlapping AABBs do not refute
    # ordinary above/below language. A higher window can overlap the top of a
    # door box, for example. Keep these cases unproved rather than filtering
    # them out; a positive centre displacement alone is not a passing proof.
    cutoff = -VERTICAL_CLEARANCE_TOLERANCE_M
    if clearance < cutoff + POSITION_BOUNDARY_TOLERANCE_M:
        return 'unknown'
    return 'pass'


def _direction_status(value, relation):
    """Classify a signed observer-frame displacement with an uncertainty band."""
    if abs(value) <= POSITION_BOUNDARY_TOLERANCE_M:
        return 'unknown'
    wants_positive = relation in ('right', 'front')
    return 'pass' if (value > 0.0) == wants_positive else 'fail'


def _floor_reference(objects):
    bottoms = sorted(_box_metrics(obj)['bottom_z_m'] for obj in objects)
    if not bottoms:
        return 0.0
    # Use the median of the lowest fifth rather than one potentially corrupt box.
    count = max(1, ceil(len(bottoms) * 0.2))
    return float(np.median(bottoms[:count]))


def size_category_scope(objects):
    """Different annotated subtypes do not establish one language comparison pool."""
    labels = sorted({canonical_category(obj['target']) for obj in objects})
    if len(labels) <= 1:
        return None
    return {'status': 'unknown', 'category_labels': labels,
            'unknown_ids': sorted(obj['bbox_id'] for obj in objects),
            'reason': 'size comparison pool contains distinct category labels: ' + ', '.join(labels)}


def candidate_removal_gate(record, *, source_quote=None, target_role='target',
                           reference_role=None, role_binding_status=None,
                           comparison_scope_ids=None, comparison_scope_status=None,
                           measurement_source=None,
                           explicit_fail_criterion=None):
    """Return an auditable candidate record and whether its removal is allowed.

    Candidate removal is deliberately stricter than a ``status == fail`` check.
    A fail is actionable only when the language role, frozen comparison scope,
    measurement provenance and fail criterion are all present and bound.  This
    helper is shared by initial diagnosis and the active evidence controller so
    unknown, incomplete or conflicting evidence remains live.
    """
    item = dict(record or {})
    status = item.get('status', 'unknown')
    quote = (item.get('source_quote') or item.get('original_quote') or
             item.get('phrase') or source_quote or '')
    role_status = (item.get('role_binding_status') or item.get('binding_status') or
                   role_binding_status or 'unknown')
    scope_status = (item.get('comparison_scope_status') or item.get('scope_status') or
                    comparison_scope_status or 'unknown')
    if role_binding_status is not None and role_status != role_binding_status:
        role_status = 'conflicting'
    if comparison_scope_status is not None and scope_status != comparison_scope_status:
        scope_status = 'conflicting'
    scope = item.get('comparison_scope_ids')
    if scope is None:
        scope = item.get('comparison_pool_ids')
    if scope is None:
        scope = comparison_scope_ids
    scope = sorted({value for value in (scope or []) if type(value) is int})
    source = item.get('measurement_source') or measurement_source or ''
    criterion = item.get('explicit_fail_criterion') or explicit_fail_criterion or ''
    recorded_roles = [item[key] for key in ('target_role', 'role') if item.get(key)]
    bound_role = recorded_roles[0] if recorded_roles else target_role
    scope_id = sha256(json.dumps([bound_role, quote, scope], sort_keys=True,
                                ensure_ascii=False).encode('utf-8')).hexdigest()
    item.update(source_quote=quote, target_role=bound_role,
                reference_role=reference_role,
                role_binding_status=role_status,
                comparison_scope_ids=scope,
                comparison_scope_status=scope_status,
                comparison_scope_id='scope-v2-' + scope_id,
                comparison_scope_frozen=scope_status == 'bound',
                removal_rule_version=CANDIDATE_REMOVAL_RULE_VERSION,
                measurement_source=source,
                explicit_fail_criterion=criterion)
    missing = []
    if status != 'fail':
        missing.append('candidate status is not an explicit fail')
    if not isinstance(quote, str) or not quote.strip():
        missing.append('original query condition is missing')
    if (role_status != 'bound' or bound_role not in ('target', 'reference') or
            any(role != target_role for role in recorded_roles)):
        missing.append('target/reference role is not bound')
    if scope_status != 'bound':
        missing.append('comparison scope is not bound')
    if not scope:
        missing.append('comparison scope is empty or missing')
    if type(item.get('candidate_id')) is not int or item.get('candidate_id') not in scope:
        missing.append('candidate is not a member of the bound comparison scope')
    if (not isinstance(source, str) or not source.strip() or
            source.strip().casefold() in ('unknown', 'unspecified', 'missing')):
        missing.append('measurement source is missing')
    if not isinstance(criterion, str) or not criterion.strip():
        missing.append('explicit fail criterion is missing')
    reliability = item.get('measurement_reliability')
    if (reliability is not None and reliability != 'usable' and
            not (isinstance(reliability, dict) and reliability.get('status') == 'usable')):
        missing.append('measurement is unavailable or unreliable')
    item['removal_allowed'] = not missing
    item['removal_blocked_reason'] = '; '.join(missing) if missing else None
    return item


def _attribute_comparison_pool(parsed, objects, category, checks, role='target'):
    """Bind a comparison subset using all explicitly checked role attributes."""
    source_attributes = attributes_for(parsed, category, role=role)
    modifiers = required_visual_attributes(appearance_attributes_for(parsed, category, role=role))
    if not modifiers:
        return list(objects), None
    generation = parsed.get('candidate_generation', {})
    unverified_fallback = (role == 'target' and isinstance(generation, dict) and
                           generation.get('source') == 'broad_fallback' and
                           generation.get('status') != 'bound')
    scoped, unknown, audit = [], [], []
    for obj in objects:
        check = aggregate_visual_checks(
            appearance_checks_for([item for item in _checks_for_role(parsed, checks, category, role) if isinstance(item, dict)
             and item.get('id') == obj['bbox_id']
             and retrieval_category(item.get('category', '')) == retrieval_category(category)], parsed, category, role),
            modifiers, source_phrases=source_attributes)
        attrs = {item.get('attribute'): item for item in (check or {}).get('attribute_checks', [])
                 if isinstance(item, dict)}
        verdicts = [attrs.get(modifier, {}).get('verdict') for modifier in modifiers]
        failed_modifiers = [modifier for modifier, value in zip(modifiers, verdicts) if value == 'no']
        explicit_no = bool((check and check.get('category_verdict') == 'no') or failed_modifiers)
        item = candidate_removal_gate(
            {'candidate_id': obj['bbox_id'], 'status': 'fail' if explicit_no else
             'pass' if check and all(value == 'yes' for value in verdicts) else 'unknown',
             'attribute_verdicts': dict(zip(modifiers, verdicts)),
             'category_verdict': (check or {}).get('category_verdict', 'uncertain')},
            source_quote=next((phrase for phrase in source_attributes
                               if any(modifier.casefold() in phrase.casefold()
                                      for modifier in failed_modifiers)),
                              parsed.get('source_query', '')),
            target_role=role, reference_role=role if role == 'reference' else None,
            role_binding_status='bound' if source_attributes and not unverified_fallback else 'unknown',
            comparison_scope_ids=[item['bbox_id'] for item in objects],
            comparison_scope_status='unknown' if unverified_fallback else 'bound',
            measurement_source='RGB appearance attribute checks',
            explicit_fail_criterion=('explicit visible no for ' + ', '.join(failed_modifiers)
                                     if failed_modifiers else 'explicit visible category no'
                                     if explicit_no else None))
        audit.append(item)
        if item['removal_allowed']:
            continue
        if check and all(value == 'yes' for value in verdicts):
            scoped.append(obj)
        else:
            unknown.append(obj['bbox_id'])
    if unknown:
        return [], {'status': 'unknown', 'modifiers': modifiers,
                    'unknown_ids': sorted(unknown),
                    'candidate_removal_audit': audit,
                    'reason': 'appearance comparison scope is not fully bound'}
    return scoped, {'status': 'bound', 'modifiers': modifiers,
                    'candidate_removal_audit': audit,
                    'selected_ids': sorted(obj['bbox_id'] for obj in scoped)}


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


def _plural_category_matches(phrase, category):
    phrase_category = canonical_category(phrase)
    category = canonical_category(category)
    if phrase_category == category:
        return True
    words = category.split()
    if not words:
        return False
    final = words[-1]
    if re.search(r'[^aeiou]y$', final):
        plural = final[:-1] + 'ies'
    elif final.endswith(('s', 'x', 'z', 'ch', 'sh')):
        plural = final + 'es'
    else:
        plural = final + 's'
    return phrase_category == ' '.join(words[:-1] + [plural])


def _other_two_hypothesis_support(query, parsed, category, objects, candidates,
                                  eligible, hypotheses, relation_anchor_slots,
                                  trusted_categories):
    """Summarize target-applicable hypotheses only for a literal three-peer query."""
    summary = {'enabled': False, 'policy': 'other_two_same_category_exact_three',
               'reason': 'query does not meet the narrow applicability policy',
               'hypotheses': [], 'candidates': []}
    if not trusted_categories:
        summary['reason'] = 'category annotations are not trusted'
        return summary
    observer = parsed.get('Observer') or {}
    origin = observer.get('origin') or {}
    facing = observer.get('facing') or {}
    observer_condition_present = (
        bool(origin.get('category') or origin.get('phrase') or
             origin.get('kind') not in (None, '', 'unspecified')) or
        bool(facing.get('category') or facing.get('phrase') or facing.get('group')))
    if (not isinstance(query, str) or len(parsed.get('Relations', [])) != 1 or
            len(relation_anchor_slots) != 1 or parsed.get('diagnostic_missing') or
            parsed.get('Unhandled') or parsed.get('Attributes') or
            parsed.get('Viewpoint') or observer_condition_present):
        return summary
    relation = parsed['Relations'][0]
    anchor_categories = relation.get('anchor_categories', [])
    if (relation.get('relation') != 'between' or
            retrieval_category(relation.get('target_category', '')) != category or
            not isinstance(anchor_categories, list) or len(anchor_categories) != 2 or
            any(retrieval_category(name) != category for name in anchor_categories)):
        return summary
    target_name = re.escape(category).replace(r'\ ', r'\s+')
    match = re.fullmatch(
        r"\s*(?:(?:it['’]s|it is)\s+)?(?:the\s+)?" + target_name +
        r"\s+(?:(?:that|which)(?:\s+is|['’]s)\s+)?between\s+the\s+other\s+two\s+"
        r"(?P<references>[a-z][a-z -]*?)\s*[.!?]?\s*", query, re.I)
    if not match or not _plural_category_matches(match['references'].strip(), category):
        return summary
    pool = [obj for obj in match_retrieval_categories([category], objects)
            if retrieval_category(obj.get('target', '')) == category]
    pool_ids = sorted(obj['bbox_id'] for obj in pool)
    if len(pool) != 3:
        summary['reason'] = 'full exact-category pool does not contain exactly three instances'
        summary['category_pool_ids'] = pool_ids
        return summary
    if (set(pool_ids) != {obj['bbox_id'] for obj in candidates} or
            set(pool_ids) != {obj['bbox_id'] for obj in eligible}):
        summary['reason'] = 'trusted exact-category pool is not fully eligible'
        summary['category_pool_ids'] = pool_ids
        return summary

    relation_index, slots = next(iter(relation_anchor_slots.items()))
    hypothesis_rows = []
    candidate_rows = []
    for index, hypothesis in enumerate(hypotheses):
        reference_ids = [hypothesis['references'].get(slot) for slot in slots]
        applicable = [oid for oid in pool_ids if oid not in reference_ids]
        hypothesis_rows.append({
            'index': index, 'reference_ids': reference_ids,
            'applicable_candidate_ids': applicable,
            'inapplicable_candidate_ids': sorted(set(pool_ids) - set(applicable)),
        })
    for oid in pool_ids:
        applicable_rows = [row for row in hypothesis_rows
                           if oid in row['applicable_candidate_ids']]
        by_hypothesis = []
        for row in applicable_rows:
            measurements = [item for item in hypotheses[row['index']]['measurements']
                            if item.get('relation') == 'between' and
                            item.get('relation_index') == relation_index and
                            item.get('target_id') == oid]
            statuses = {item.get('status', 'unknown') for item in measurements}
            status = ('fail' if 'fail' in statuses else
                      'unknown' if not measurements or 'unknown' in statuses else
                      'pass' if statuses == {'pass'} else 'unknown')
            by_hypothesis.append({'hypothesis_index': row['index'], 'status': status})
        statuses = [item['status'] for item in by_hypothesis]
        candidate_status = ('fail' if statuses and all(item == 'fail' for item in statuses) else
                            'unknown' if not statuses or 'unknown' in statuses or 'fail' in statuses else
                            'pass')
        candidate_rows.append({
            'candidate_id': oid, 'status': candidate_status,
            'applicable_hypothesis_indices': [item['hypothesis_index'] for item in by_hypothesis],
            'hypothesis_statuses': by_hypothesis,
        })
    summary.update(enabled=True, reason=None, category_pool_ids=pool_ids,
                   hypotheses=hypothesis_rows, candidates=candidate_rows)
    return summary


def candidate_hypothesis_support(diagnosis, candidate_id):
    """Return the narrow between-applicability record for one candidate, if active."""
    summary = diagnosis.get('candidate_hypothesis_support', {})
    if not summary.get('enabled'):
        return None
    return next((item for item in summary.get('candidates', [])
                 if item.get('candidate_id') == candidate_id), None)


def diagnose(parsed, objects, candidates, room_center, checks=(), trusted_categories=False,
             apply_size_constraints=True, query=None, size_measurements=None):
    """Unknown category evidence stays eligible; predicates are ANDed per interpretation."""
    parsed.setdefault('_size_scope_bindings', {})
    if query and not parsed.get('source_query'):
        parsed = {**parsed, 'source_query': query}
    rewritten = parsed.get('query_rewrite') if isinstance(parsed, dict) else None
    if isinstance(rewritten, dict):
        from seeground.query.query_rewrite import constraints_as_relations
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
    category_pool = match_retrieval_categories([category], objects) if category else []
    generation = parsed.get('candidate_generation', {})
    unverified_fallback = (isinstance(generation, dict) and
                           generation.get('source') == 'broad_fallback' and
                           generation.get('status') != 'bound')
    category_bound = bool(category_pool) and not unverified_fallback
    if category and not category_bound:
        missing.append('target category recall/binding missing: ' + category)
    checks_by_key = {}
    for check in checks:
        key = (check['id'], retrieval_category(check['category']))
        checks_by_key.setdefault(key, []).append(check)
    by_id = {o['bbox_id']:o for o in objects}
    geometry = {
        'target_role_binding_status': 'bound' if category_bound else 'unknown',
        'size_constraints': [], 'floor_constraint': None, 'relation_pairs': [],
        'between_triplets': [], 'proximity_pairs': [], 'direction_evidence': [],
        'distance_rankings': [], 'distance_ignored': [],
        'candidate_removal_audit': [],
        'vertical_policy': {
            'horizontal_overlap_required': False,
            'horizontal_gap_max_m': HORIZONTAL_ALIGNMENT_GAP_M,
            'vertical_clearance_tolerance_m': VERTICAL_CLEARANCE_TOLERANCE_M,
            'requires_vertical_separation': True,
            'overlap_policy': 'positive center displacement with overlapping boxes is unknown, not fail or pass',
        },
        'size_policy': {
            'metric': 'AABB volume, footprint and height',
            'adjective_policy': 'predeclared relative comparisons and descriptive unknown use shared size predicates',
            'shape_policy': 'XY extents and ratios are soft AABB proxies, not intrinsic object width/depth; '
                            'skinny/narrow is not a minimum-short-edge rule',
        },
    }
    def verdict(oid, name, role='target'):
        if trusted_categories and not match_retrieval_categories([name], [by_id[oid]]):
            return 'no'
        language_role = 'target' if role == 'target' else 'reference'
        source_attributes = attributes_for(parsed, name, role=language_role)
        expected = required_visual_attributes(appearance_attributes_for(parsed, name, role=language_role))
        visual_checks = _checks_for_role(
            parsed, checks_by_key.get((oid, retrieval_category(name)), []), name, role)
        checked = aggregate_visual_checks(
            appearance_checks_for(visual_checks, parsed, name, language_role), expected,
            source_phrases=source_attributes)
        if trusted_categories and not expected:
            if checked and checked.get('category_verdict') == 'no':
                return 'no'
            if not checked or any(check.get('visibility') in
                                  ('unknown', 'low_detail', 'occluded', 'unmarked', 'missing_view')
                                  for check in visual_checks):
                return 'yes'
        if checked:
            return checked.get('verdict', 'uncertain')
        return 'uncertain'
    eligible = []
    initial_scope_ids = sorted({o['bbox_id'] for o in candidates
                                if type(o.get('bbox_id')) is int})
    for obj in candidates:
        verdict_value = verdict(obj['bbox_id'], category)
        check_rows = _checks_for_role(
            parsed, checks_by_key.get((obj['bbox_id'], category), []), category, 'target')
        failed_attributes = [item.get('attribute', '') for check in check_rows
                             for item in check.get('attribute_checks', [])
                             if isinstance(item, dict) and item.get('verdict') == 'no']
        initial_evidence = candidate_removal_gate(
            {'candidate_id': obj['bbox_id'],
             'status': 'fail' if verdict_value == 'no' else
                       'unknown' if verdict_value == 'uncertain' else 'pass',
             'reason': 'category or appearance evidence contradicted this candidate',
             'observed_category': canonical_category(obj['target']),
             'required_category': category, 'appearance_checks': check_rows},
            source_quote=next((attribute.get('phrase', '') for attribute in parsed.get('Attributes', [])
                               if retrieval_category(attribute.get('category', '')) == category and
                               attribute_role(parsed, attribute) == 'target' and
                               any(unit and unit.casefold() in attribute.get('phrase', '').casefold()
                                   for unit in failed_attributes)),
                              parsed.get('source_query', '')),
            target_role='target', role_binding_status='bound' if category_bound else 'unknown',
            comparison_scope_ids=initial_scope_ids,
            comparison_scope_status='bound' if category_bound else 'unknown',
            measurement_source='annotated category' if trusted_categories and not
                match_retrieval_categories([category], [obj]) else 'RGB appearance checks',
            explicit_fail_criterion=('annotated category does not match the bound target category'
                                     if trusted_categories and not match_retrieval_categories([category], [obj])
                                     else 'category/appearance verdict is an explicit visible no'))
        geometry['candidate_removal_audit'].append(initial_evidence)
        if initial_evidence['removal_allowed']:
            continue
        if initial_evidence['status'] == 'fail':
            missing.append('category/appearance failure lacks complete exclusion evidence: ' + str(obj['bbox_id']))
        eligible.append(obj)
    distance_pool, distance_scope = _attribute_comparison_pool(
        parsed, list(candidates), category, checks)
    if distance_scope:
        geometry['candidate_removal_audit'].extend(distance_scope.get('candidate_removal_audit', []))
    if distance_scope and distance_scope['status'] == 'unknown':
        distance_pool = list(candidates)
    # Explicit superlatives and "on the floor" are deterministic box predicates.
    size_allowed = None
    floor_allowed = None
    pending_size_conditions = []
    measurements = (list(size_measurements) if size_measurements is not None else
                    [{'id': o['bbox_id'], 'status': 'known', **_box_metrics(o)} for o in objects])
    local_contrast = parsed.get('relative_height_binding')
    for attribute in parsed.get('Attributes', []):
        if (retrieval_category(attribute.get('category', '')) != category or
                attribute_role(parsed, attribute) != 'target'):
            continue
        phrase = attribute.get('phrase', '')
        if re.search(r'\b(on the floor|on floor|on the ground|grounded)\b', phrase, re.I):
            floor_z = _floor_reference(objects)
            floor_ids = {o['bbox_id'] for o in objects
                         if _box_metrics(o)['bottom_z_m'] <= floor_z + 0.15}
            geometry['floor_constraint'] = {'floor_z_m': floor_z, 'tolerance_m': 0.15,
                                            'selected_ids': sorted(floor_ids)}
            floor_allowed = floor_ids if floor_allowed is None else floor_allowed & floor_ids
    for spec in size_conditions(parsed, category):
        if not apply_size_constraints:
            pending_size_conditions.append({'attribute': spec['word'], 'phrase': spec['original_quote'],
                                            'role': 'target'})
        if local_contrast and spec['word'] == local_contrast['target_word']:
            continue  # Evaluated jointly with its two bound references below.
        pool, scope_record = size_condition_scope(parsed, objects, category, spec, checks)
        geometry['candidate_removal_audit'].extend(scope_record.get('candidate_removal_audit', []))
        record = (evaluate_scope_hypotheses(parsed, objects, category, spec, checks, measurements,
                    [o['bbox_id'] for o in candidates]) if spec.get('scope_hypotheses') else
                  evaluate_size_condition(spec, measurements, [o['bbox_id'] for o in pool],
            scope_bound=scope_record['status'] == 'bound', candidate_ids=[o['bbox_id'] for o in candidates],
            scope_source=scope_record['source'], scope_unknown_ids=scope_record.get('unknown_ids', [])))
        geometry['size_constraints'].append(record)
        if apply_size_constraints:
            gated_evidence = []
            for item in record['candidate_evidence']:
                gated = candidate_removal_gate(
                    item,
                    source_quote=record.get('original_quote'),
                    target_role='target',
                    role_binding_status=('bound' if record.get('scope_status') == 'bound'
                                         else 'unknown'),
                    comparison_scope_ids=record.get('comparison_pool_ids', []),
                    comparison_scope_status=record.get('scope_status', 'unknown'),
                    measurement_source=record.get('measurement_sources', {}).get(
                        str(item.get('candidate_id')), ''),
                    explicit_fail_criterion=(
                        'reliable measurement contradicts the quoted size condition'
                        if item.get('status') == 'fail' else None))
                gated_evidence.append(gated)
                geometry['candidate_removal_audit'].append(gated)
                if gated['status'] == 'fail' and not gated['removal_allowed']:
                    missing.append('target size failure lacks complete exclusion evidence: ' + str(item['candidate_id']))
            record['candidate_evidence'] = gated_evidence
            flags = {item['candidate_id']: item for item in gated_evidence}
            allowed = {o['bbox_id'] for o in candidates
                       if not flags.get(o['bbox_id'], {}).get('removal_allowed')}
            size_allowed = allowed if size_allowed is None else size_allowed & allowed
            if record['status'] == 'unknown':
                missing.append(record['reason'])
    if size_allowed is not None and apply_size_constraints:
        eligible = [o for o in eligible if o['bbox_id'] in size_allowed]
    if floor_allowed is not None:
        kept = []
        for obj in eligible:
            floor_fail = obj['bbox_id'] not in floor_allowed
            audit = candidate_removal_gate(
                {'candidate_id': obj['bbox_id'],
                 'status': 'fail' if floor_fail else 'pass',
                 'reason': 'bottom of box is outside the declared floor tolerance',
                 'bottom_z_m': _box_metrics(obj)['bottom_z_m'],
                 'floor_z_m': geometry['floor_constraint']['floor_z_m'], 'tolerance_m': .15,
                 'measurement_reliability': _box_metrics(obj)['measurement_reliability']},
                source_quote=next((attribute.get('phrase', '')
                                   for attribute in parsed.get('Attributes', [])
                                   if retrieval_category(attribute.get('category', '')) == category and
                                   attribute_role(parsed, attribute) == 'target' and
                                   re.search(r'\b(on the floor|on floor|on the ground|grounded)\b',
                                                attribute.get('phrase', ''), re.I)),
                                  ''),
                target_role='target', role_binding_status='bound' if category_bound else 'unknown',
                comparison_scope_ids=sorted({item['bbox_id'] for item in objects}),
                comparison_scope_status='bound' if category_bound else 'unknown',
                measurement_source='annotated_box bottom_z_m',
                explicit_fail_criterion=('bottom_z_m exceeds floor tolerance'
                                         if floor_fail else None))
            geometry['candidate_removal_audit'].append(audit)
            if not audit['removal_allowed']:
                kept.append(obj)
                if floor_fail:
                    missing.append('floor failure lacks complete exclusion evidence: ' + str(obj['bbox_id']))
        eligible = kept
    if local_contrast and not apply_size_constraints:
        pending_size_conditions.append({'attribute': local_contrast['reference_word'],
                                        'phrase': local_contrast['reference_word'], 'role': 'reference'})
    if pending_size_conditions:
        missing.append('target size conditions lack complete evidence')
    directions = any(r['relation'] in ('left', 'right', 'front', 'back') for r in parsed['Relations'])
    observer = parsed['Observer']
    for observer_role in ('origin', 'facing'):
        entry = observer.get(observer_role, {})
        name = entry.get('category', '') if isinstance(entry, dict) else ''
        for spec in size_conditions(parsed, name, role='reference') if name else []:
            pending_size_conditions.append({'attribute': spec['word'],
                'phrase': spec['original_quote'], 'role': 'reference', 'category': name,
                'observer_role': observer_role, 'binding_status': 'unknown'})
            missing.append('observer reference size comparison is not jointly bound to its frame: ' +
                           spec['original_quote'])
    frames, frame_missing = observer_frames(parsed, objects, room_center) if directions else ([None], [])
    if directions:
        kept_frames = []
        for frame in frames:
            scope_ids = [*frame['facing_ids']]
            if type(frame.get('origin_id')) is int:
                scope_ids.append(frame['origin_id'])
            frame_allowed = True
            frame_checks = [('facing', oid) for oid in frame['facing_ids']]
            if type(frame.get('origin_id')) is int:
                frame_checks.append(('origin', frame['origin_id']))
            for observer_role, oid in frame_checks:
                entry = observer[observer_role]
                name = retrieval_category(entry.get('category', ''))
                value = verdict(oid, name, 'observer_' + observer_role)
                audit = candidate_removal_gate(
                    {'candidate_id': oid, 'status': 'fail' if value == 'no' else
                     'unknown' if value == 'uncertain' else 'pass',
                     'observer_role': observer_role,
                     'appearance_checks': _checks_for_role(parsed,
                         checks_by_key.get((oid, name), []), name, 'observer_' + observer_role)},
                    source_quote=entry.get('phrase', ''), target_role='reference',
                    reference_role='reference',
                    role_binding_status='bound' if name and quoted(parsed.get('source_query', ''),
                        entry.get('phrase', '')) else 'unknown',
                    comparison_scope_ids=scope_ids, comparison_scope_status='bound',
                    measurement_source='RGB observer reference checks',
                    explicit_fail_criterion='observer reference has an explicit visible category/appearance no')
                geometry['candidate_removal_audit'].append(audit)
                if audit['removal_allowed']:
                    frame_allowed = False
                elif value == 'no':
                    missing.append('observer reference failure lacks complete exclusion evidence')
            if frame_allowed:
                kept_frames.append(frame)
        frames = kept_frames
        if not frames and not frame_missing:
            frame_missing.append('observer categories contradicted by RGB evidence')
        missing.extend(frame_missing)
        if not frames:
            for target in eligible:
                for index, relation in enumerate(parsed['Relations']):
                    if relation.get('relation') in ('left', 'right', 'front', 'back'):
                        geometry['direction_evidence'].append({
                            'target_id': target['bbox_id'], 'relation': relation['relation'],
                            'relation_index': index, 'phrase': relation.get('phrase', ''),
                            'scope': relation.get('scope', ''), 'anchor_id': None,
                            'signed_m': None, 'passes': None, 'status': 'unknown',
                            'reason': 'observer frame is unavailable',
                        })
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
    unverified_reference_bindings = set()
    for name in sorted(reference_names):
        reference_pool = match_retrieval_categories([name], objects)
        matches = []
        for obj in reference_pool:
            reference_verdict = verdict(obj['bbox_id'], name, 'reference')
            check_rows = _checks_for_role(parsed,
                checks_by_key.get((obj['bbox_id'], name), []), name, 'reference')
            failed_attributes = [item.get('attribute', '') for check in check_rows
                                 for item in check.get('attribute_checks', [])
                                 if isinstance(item, dict) and item.get('verdict') == 'no']
            audit = candidate_removal_gate(
                {'candidate_id': obj['bbox_id'],
                 'status': 'fail' if reference_verdict == 'no' else
                           'unknown' if reference_verdict == 'uncertain' else 'pass',
                 'appearance_checks': check_rows},
                source_quote=next((attribute.get('phrase', '') for attribute in parsed.get('Attributes', [])
                                   if retrieval_category(attribute.get('category', '')) == name and
                                   attribute_role(parsed, attribute) == 'reference' and
                                   any(unit and unit.casefold() in attribute.get('phrase', '').casefold()
                                       for unit in failed_attributes)),
                                  parsed.get('source_query', '')),
                target_role='reference', reference_role='reference', role_binding_status='bound',
                comparison_scope_ids=[item['bbox_id'] for item in reference_pool],
                comparison_scope_status='bound', measurement_source='RGB appearance checks',
                explicit_fail_criterion='reference category/appearance has an explicit visible no')
            geometry['candidate_removal_audit'].append(audit)
            if not audit['removal_allowed']:
                matches.append(obj)
                if reference_verdict == 'no':
                    missing.append('reference appearance failure lacks complete exclusion evidence: ' + str(obj['bbox_id']))
        explicit_anchor_ids = {
            relation.get('anchor_id', relation.get('anchor_instance_id'))
            for relation in parsed['Relations']
            if (relation.get('relation') in ('nearest', 'farthest') and
                retrieval_category(relation.get('anchor_category', '')) == name and
                type(relation.get('anchor_id', relation.get('anchor_instance_id'))) is int)
        }
        verified_binding = parsed.get('_verified_reference_bindings', {}).get(name, {})
        verified_id = verified_binding.get('id') if isinstance(verified_binding, dict) else None
        binding_verified = (len(explicit_anchor_ids) == 1 and verified_id in explicit_anchor_ids and
                            verified_binding.get('status') == 'bound' and
                            bool(verified_binding.get('source')) and
                            any(obj['bbox_id'] == verified_id for obj in matches))
        if len(explicit_anchor_ids) > 1:
            missing.append('distance reference has incompatible explicit instance bindings: ' + name)
        elif explicit_anchor_ids and not binding_verified:
            missing.append('distance reference explicit instance binding lacks source verification: ' + name)
        if binding_verified:
            kept = []
            for obj in matches:
                audit = candidate_removal_gate(
                    {'candidate_id': obj['bbox_id'],
                     'status': 'pass' if obj['bbox_id'] == verified_id else 'fail',
                     'verified_reference_binding': verified_binding},
                    source_quote=next((relation.get('phrase', '') for relation in parsed['Relations']
                                       if retrieval_category(relation.get('anchor_category', '')) == name), ''),
                    target_role='reference', reference_role='reference', role_binding_status='bound',
                    comparison_scope_ids=[item['bbox_id'] for item in reference_pool],
                    comparison_scope_status='bound', measurement_source=verified_binding['source'],
                    explicit_fail_criterion='reference instance ID differs from the independently verified binding')
                geometry['candidate_removal_audit'].append(audit)
                if not audit['removal_allowed']:
                    kept.append(obj)
            matches = kept
        elif explicit_anchor_ids:
            unverified_reference_bindings.add(name)
            for obj in matches:
                if obj['bbox_id'] in explicit_anchor_ids and len(explicit_anchor_ids) == 1:
                    continue
                audit = candidate_removal_gate(
                    {'candidate_id': obj['bbox_id'], 'status': 'unknown',
                     'reported_anchor_ids': sorted(explicit_anchor_ids),
                     'reason': 'parser instance IDs do not establish a verified reference binding'},
                    source_quote=next((relation.get('phrase', '') for relation in parsed['Relations']
                                       if retrieval_category(relation.get('anchor_category', '')) == name), ''),
                    target_role='reference', reference_role='reference', role_binding_status='unknown',
                    comparison_scope_ids=[item['bbox_id'] for item in reference_pool],
                    comparison_scope_status='unknown', measurement_source='', explicit_fail_criterion='')
                geometry['candidate_removal_audit'].append(audit)
        if not local_contrast:
            for spec in size_conditions(parsed, name, role='reference'):
                pool, scope_record = size_condition_scope(parsed, objects, name, spec, checks, 'reference')
                geometry['candidate_removal_audit'].extend(scope_record.get('candidate_removal_audit', []))
                if not apply_size_constraints:
                    pending_size_conditions.append({'attribute': spec['word'],
                        'phrase': spec['original_quote'], 'role': 'reference', 'category': name})
                    missing.append('reference size conditions lack complete evidence')
                    continue
                record = (evaluate_scope_hypotheses(parsed, objects, name, spec, checks, measurements,
                            [o['bbox_id'] for o in matches]) if spec.get('scope_hypotheses') else
                          evaluate_size_condition(spec, measurements, [o['bbox_id'] for o in pool],
                    scope_bound=scope_record['status'] == 'bound', candidate_ids=[o['bbox_id'] for o in matches],
                    scope_source=scope_record['source'], scope_unknown_ids=scope_record.get('unknown_ids', [])))
                geometry['size_constraints'].append(record)
                gated_evidence = []
                for item in record['candidate_evidence']:
                    gated = candidate_removal_gate(
                        item,
                        source_quote=record.get('original_quote'),
                        target_role='reference', reference_role='reference',
                        role_binding_status=('bound' if record.get('scope_status') == 'bound'
                                             else 'unknown'),
                        comparison_scope_ids=record.get('comparison_pool_ids', []),
                        comparison_scope_status=record.get('scope_status', 'unknown'),
                        measurement_source=record.get('measurement_sources', {}).get(
                            str(item.get('candidate_id')), ''),
                        explicit_fail_criterion=(
                            'reliable measurement contradicts the quoted reference size condition'
                            if item.get('status') == 'fail' else None))
                    gated_evidence.append(gated)
                    geometry['candidate_removal_audit'].append(gated)
                    if gated['status'] == 'fail' and not gated['removal_allowed']:
                        missing.append('reference size failure lacks complete exclusion evidence: ' + str(item['candidate_id']))
                record['candidate_evidence'] = gated_evidence
                matches = [o for o in matches if not any(
                    e['candidate_id'] == o['bbox_id'] and e.get('removal_allowed')
                    for e in gated_evidence)]
                if any(not any(e['candidate_id'] == obj['bbox_id'] and e['status'] == 'pass'
                               for e in gated_evidence) for obj in matches):
                    pending_size_conditions.append({'attribute': spec['word'],
                        'phrase': spec['original_quote'], 'role': 'reference', 'category': name})
                    missing.append('reference size conditions lack complete evidence')
                if record['status'] == 'unknown':
                    missing.append('reference ' + record['reason'])
        if not name or not matches:
            missing.append('missing reference category: ' + name)
        else:
            references[name] = matches

    # A between predicate may name two references from the same category
    # (for example, "between the tables"). Keep the first occurrence on the
    # normal category key for compatibility, then allocate a private slot for
    # each later occurrence in that same predicate. This lets the hypothesis
    # search bind distinct instances without changing the parser schema.
    relation_anchor_slots = {}
    reference_slot_categories = {name: name for name in references}
    for relation_index, relation in enumerate(parsed['Relations']):
        if relation.get('relation') != 'between':
            continue
        seen = {}
        slots = []
        for position, name in enumerate(relation.get('anchor_categories', [])):
            category_name = retrieval_category(name)
            occurrence = seen.get(category_name, 0)
            seen[category_name] = occurrence + 1
            if occurrence == 0:
                slot = category_name
            else:
                slot = f'__between_{relation_index}_{position}_{category_name}'
            slots.append(slot)
            reference_slot_categories[slot] = category_name
            if slot not in references and references.get(category_name):
                references[slot] = list(references[category_name])
        relation_anchor_slots[relation_index] = slots
        required = {category: count for category, count in seen.items() if count > 1}
        for category_name, count in required.items():
            if len(references.get(category_name, [])) < count:
                missing.append('between reference instances are not distinctly bound: ' + category_name)
    geometry['candidate_metrics'] = _candidate_metrics(objects, category, eligible) if category else []
    geometry['reference_metrics'] = [
        metric for name, matches in references.items()
        for metric in _candidate_metrics(objects, reference_slot_categories.get(name, name), matches)
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

    def hypothesis_fail_proof(hypothesis, oid, hypothesis_index):
        references_complete = all(
            verdict(value, reference_slot_categories.get(name, name), 'reference') == 'yes'
            for name, value in hypothesis.get('references', {}).items())
        bound_frame = hypothesis.get('observer')
        if bound_frame:
            references_complete = references_complete and all(
                verdict(value, observer['facing'].get('category', ''), 'observer_facing') == 'yes'
                for value in bound_frame.get('facing_ids', []))
            if type(bound_frame.get('origin_id')) is int:
                references_complete = references_complete and verdict(
                    bound_frame['origin_id'], observer['origin'].get('category', ''), 'observer_origin') == 'yes'
        rows = [row for row in hypothesis.get('measurements', []) if row.get('target_id') == oid]
        for row in rows:
            candidate_row = next((item for item in row.get('candidate_evidence', [])
                                  if item.get('candidate_id') == oid), row)
            if candidate_row.get('status') != 'fail':
                continue
            anchors = ([row['anchor_id']] if type(row.get('anchor_id')) is int else
                       row.get('anchor_ids') or row.get('frame_facing_ids') or [])
            scope = row.get('comparison_pool_ids') or [oid, *anchors]
            scope_bound = (row.get('scope_status') == 'bound' if 'scope_status' in row else
                           bool(anchors) and all(type(value) is int for value in anchors))
            scope_bound = scope_bound and references_complete and not any(
                item.get('role') == 'reference' for item in pending_size_conditions)
            # Local short/tall is one joint comparison. A reliable target-size
            # fail or a distinct-instance contradiction is already sufficient
            # for this assignment; unknown size rows for other targets cannot
            # invalidate it. Ordinary geometry still needs this target's
            # reference size evidence to be complete.
            if (row.get('interpretation') != 'reference_comparison' and
                    row.get('measurement_source') != 'bound instance identity'):
                scope_bound = scope_bound and not any(
                    item.get('status') == 'unknown' for measurement in rows
                    if measurement.get('role') == 'reference'
                    for item in measurement.get('candidate_evidence', []))
            proof = candidate_removal_gate(
                {**candidate_row, 'candidate_id': oid, 'hypothesis_index': hypothesis_index,
                 'evidence': row}, source_quote=row.get('original_quote') or row.get('phrase', ''),
                target_role='target', role_binding_status='bound' if category_bound and scope_bound else 'unknown',
                comparison_scope_ids=scope, comparison_scope_status='bound' if scope_bound else 'unknown',
                measurement_source=(row.get('measurement_source') or
                    row.get('measurement_sources', {}).get(str(oid)) or 'annotated_box relation geometry'),
                explicit_fail_criterion=(row.get('explicit_fail_criterion') or candidate_row.get('reason') or
                    'bound ' + str(row.get('relation', 'relation')) + ' geometry fails its declared predicate'))
            if proof['removal_allowed']:
                return proof
        return {'hypothesis_index': hypothesis_index, 'removal_allowed': False,
                'reason': 'this interpretation has no bound explicit candidate failure'}

    hypotheses = []
    distance_by_anchor = {}
    choices = list(references)
    combinations = prod(len(references[n]) for n in choices) * max(len(frames), 1)
    if combinations > 256:
        missing.append('hypothesis budget exceeded (256); no truncated stability claim')
    interpretations = product(frames or [None], product(*(references[n] for n in choices))) if combinations <= 256 else []
    for frame, refs in interpretations:
        binding = dict(zip(choices, refs))
        invalid_duplicate_reference = False
        for slots in relation_anchor_slots.values():
            bound_ids = [binding[slot]['bbox_id'] for slot in slots if slot in binding]
            if len(bound_ids) != len(set(bound_ids)):
                invalid_duplicate_reference = True
                break
        if invalid_duplicate_reference:
            continue
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
                relation_anchor_names = (relation_anchor_slots.get(index, [])
                                         if kind == 'between' else
                                         [retrieval_category(relation.get('anchor_category', ''))])
                relation_anchors = [binding.get(name)
                                    for name in relation_anchor_names]
                if any(reference_slot_categories.get(name, name) in unverified_reference_bindings
                       for name in relation_anchor_names):
                    measured.append({'target_id': oid, 'relation': kind, 'relation_index': index,
                                     'phrase': relation.get('phrase', ''), 'status': 'unknown',
                                     'passes': None,
                                     'reason': 'reference instance ID binding is unverified or conflicting'})
                    continue
                anchor = relation_anchors[0] if relation_anchors else None
                if anchor and anchor['bbox_id'] == oid:
                    measured.append({'target_id': oid, 'anchor_id': anchor['bbox_id'],
                                     'relation': kind, 'relation_index': index,
                                     'phrase': relation.get('phrase', ''), 'status': 'fail',
                                     'passes': False, 'measurement_source': 'bound instance identity',
                                     'explicit_fail_criterion': 'target and relation reference must be distinct instances'})
                    passed = False
                    break
                if kind == 'between':
                    if len(relation_anchors) != 2 or any(item is None for item in relation_anchors):
                        unbound = [reference_slot_categories.get(name, name)
                                   for name, item in zip(relation_anchor_names, relation_anchors)
                                   if item is None]
                        reason = ('between reference instance is unbound: ' + ', '.join(unbound)
                                  if unbound else 'between requires two bound reference instances')
                        missing.append(reason)
                        between = {'target_id': oid,
                                   'anchor_ids': [item['bbox_id'] if item else None
                                                  for item in relation_anchors],
                                   'relation': kind, 'relation_index': index,
                                   'phrase': relation.get('phrase', ''),
                                   'passes': None, 'status': 'unknown', 'reason': reason}
                        geometry['between_triplets'].append(between)
                        measured.append(between)
                        continue
                    if any(item['bbox_id'] == oid for item in relation_anchors):
                        measured.append({'target_id': oid,
                                         'anchor_ids': [item['bbox_id'] for item in relation_anchors],
                                         'relation': kind, 'relation_index': index,
                                         'phrase': relation.get('phrase', ''), 'status': 'fail',
                                         'passes': False, 'measurement_source': 'bound instance identity',
                                         'explicit_fail_criterion': 'between target must be distinct from both references'})
                        passed = False
                        break
                    if local_contrast and apply_size_constraints:
                        spec = size_specs(local_contrast['target_word'], category,
                                          query=parsed.get('source_query', ''), role='target')[0]
                        spec.update(interpretation='reference_comparison',
                                    scope_kind='bound_relation_hypothesis',
                                    scope_phrase=local_contrast['relation_phrase'],
                                    reference_ids=[o['bbox_id'] for o in relation_anchors])
                        height = evaluate_size_condition(spec, measurements,
                            [oid, *spec['reference_ids']], scope_bound=True, candidate_ids=[oid],
                            scope_source='same between hypothesis; target and two distinct references')
                        height.update(target_id=oid, anchor_ids=spec['reference_ids'], relation_index=index)
                        geometry['size_constraints'].append(height)
                        measured.append(height)
                        reference_spec = size_specs(local_contrast['reference_word'], category,
                            query=parsed.get('source_query', ''), role='reference',
                            source='same quoted short/tall between clause')[0]
                        reference_spec.update(interpretation='reference_comparison',
                            scope_kind='bound_relation_hypothesis',
                            scope_phrase=local_contrast['relation_phrase'],
                            reference_ids=[oid], required_reference_count=1,
                            metric_reason='bound references taller than this target in the same between hypothesis; no absolute tallness claim')
                        reference_height = evaluate_size_condition(reference_spec, measurements,
                            [oid, *spec['reference_ids']], scope_bound=True,
                            candidate_ids=spec['reference_ids'],
                            scope_source='same between hypothesis; references compared with the bound target')
                        reference_height.update(target_id=oid, anchor_ids=spec['reference_ids'], relation_index=index)
                        geometry['size_constraints'].append(reference_height)
                        measured.append(reference_height)
                        flag = height['candidate_evidence'][0]['status']
                        passed = passed and flag != 'fail'
                    between = _between_geometry(target, relation_anchors[0], relation_anchors[1])
                    between.update(relation='between', relation_index=index,
                                   phrase=relation.get('phrase', ''))
                    geometry['between_triplets'].append(between)
                    measured.append(between)
                    if (local_contrast and apply_size_constraints and between['passes'] is not False
                            and flag == 'unknown'):
                        missing.append(height['reason'])
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
                    frame_origin = tuple(frame['origin']) if frame else None
                    key = (index, anchor['bbox_id'], frame_origin)
                    if key not in distance_by_anchor:
                        bound = {**relation, 'anchor_id': anchor['bbox_id']}
                        result = distance_rankings(
                            rewritten.get('query', relation.get('phrase', ''))
                            if isinstance(rewritten, dict) else relation.get('phrase', ''),
                            {'Relations': [bound]}, distance_pool, [anchor],
                            viewpoint=frame or parsed.get('Observer'))
                        for ranking in result['rankings']:
                            ranking['relation_index'] = index
                            ranking['comparison_pool_ids'] = sorted(o['bbox_id'] for o in distance_pool)
                            if distance_scope and distance_scope['status'] == 'unknown':
                                ranking['comparison_status'] = 'unknown'
                                ranking['scope_status'] = 'unknown'
                                ranking['scope_unknown_ids'] = distance_scope['unknown_ids']
                                ranking['best_ids'] = []
                                ranking['unknown_target_ids'] = sorted(o['bbox_id'] for o in distance_pool)
                        if distance_scope and distance_scope['status'] == 'unknown':
                            result['ignored'].append({
                                'relation': bound,
                                'reason': 'distance comparison appearance scope is not fully bound',
                                'unknown_scope_ids': distance_scope['unknown_ids'],
                            })
                        distance_by_anchor[key] = result
                        geometry['distance_rankings'].extend(result['rankings'])
                        geometry['distance_ignored'].extend(result['ignored'])
                    result = distance_by_anchor[key]
                    ranking = next(iter(result['rankings']), None)
                    requested_id = relation.get('target_id', relation.get('target_instance_id'))
                    if type(requested_id) is int:
                        geometry['target_instance_binding_status'] = 'unknown'
                    if type(requested_id) is int and requested_id != oid:
                        distance_status = 'unknown'
                        missing.append('distance target instance binding lacks source verification')
                    elif ranking is None or ranking['comparison_status'] == 'unknown':
                        distance_status = 'unknown'
                    else:
                        distance_status = 'pass' if oid in ranking['best_ids'] else 'fail'
                    distance = next((row['distance_m'] for row in
                                     ranking['ordered_targets'] if row['target_id'] == oid), None) if ranking else None
                    measured.append({
                        'target_id': oid, 'anchor_id': anchor['bbox_id'], 'relation': kind,
                        'relation_index': index, 'phrase': relation.get('phrase', ''),
                        'distance_m': distance, 'metric': ranking['metric'] if ranking else None,
                        'status': distance_status,
                        'passes': None if distance_status == 'unknown' else distance_status == 'pass',
                        'best_ids': ranking['best_ids'] if ranking else [],
                        'unknown_target_ids': ranking['unknown_target_ids'] if ranking else [],
                        'comparison_pool_ids': (ranking.get('comparison_pool_ids', []) + [anchor['bbox_id']]
                                                if ranking else []),
                        'scope_status': ('bound' if ranking and ranking.get('comparison_status') != 'unknown'
                                         else 'unknown'),
                        'explicit_fail_criterion': ('candidate is outside best_ids under the bound distance metric'
                                                    if distance_status == 'fail' else None),
                    })
                    if distance_status == 'unknown':
                        missing.extend(item['reason'] for item in result['ignored'])
                        if ranking and ranking['unknown_target_ids']:
                            missing.append('distance comparison has unknown target measurements: ' +
                                           str(ranking['unknown_target_ids']))
                    else:
                        passed = passed and distance_status == 'pass'
                elif kind in ('nearest', 'farthest'):
                    missing.append(kind + ' relation reference instance is unbound')
                elif kind in ('above', 'below') and anchor:
                    pair = _pair_geometry(target, anchor)
                    pair['relation'] = kind
                    pair.update(relation_index=index, phrase=relation.get('phrase', ''))
                    pair['horizontal_alignment_passes'] = (
                        pair['horizontal_overlap_m2'] > 0.0 or
                        pair['horizontal_gap_m'] <= HORIZONTAL_ALIGNMENT_GAP_M)
                    vertical_status = _vertical_status(pair, kind)
                    pair['vertical_separation_passes'] = vertical_status == 'pass'
                    pair['passes'] = (True if vertical_status == 'pass' else
                                      False if vertical_status == 'fail' else None)
                    pair['status'] = vertical_status
                    geometry['relation_pairs'].append(pair)
                    measured.append(pair)
                    if vertical_status == 'unknown':
                        missing.append('vertical relation geometry is uncertain at an alignment boundary or with unproved box separation')
                    else:
                        passed = passed and pair['passes']
                elif kind in ('above', 'below'):
                    missing.append(kind + ' relation reference instance is unbound')
                    measured.append({'target_id': oid, 'relation': kind, 'anchor_id': None,
                                     'relation_index': index, 'phrase': relation.get('phrase', ''),
                                     'passes': None, 'status': 'unknown',
                                     'reason': 'vertical reference instance is unbound'})
                elif kind in ('left', 'right', 'front', 'back') and frame:
                    if relation.get('scope') == 'relative_anchor' and not anchor:
                        geometry['direction_evidence'].append({
                            'target_id': oid, 'relation': kind, 'scope': relation.get('scope', ''),
                            'relation_index': index, 'phrase': relation.get('phrase', ''),
                            'anchor_id': None, 'signed_m': None, 'passes': None,
                            'status': 'unknown', 'reason': 'direction reference instance is unbound',
                        })
                        missing.append('direction reference instance is unbound')
                        continue
                    base = anchor['bbox_3d'][:3] if relation.get('scope') == 'relative_anchor' else frame['origin']
                    delta = np.asarray(target['bbox_3d'][:3]) - base
                    axis = frame['right'] if kind in ('left', 'right') else frame['forward']
                    value = float(np.dot(delta, axis))
                    direction_status = _direction_status(value, kind)
                    direction = {'target_id': oid, 'relation': kind, 'scope': relation['scope'],
                                 'relation_index': index, 'phrase': relation.get('phrase', ''),
                                 'anchor_id': anchor['bbox_id'] if anchor else None,
                                 'frame_origin_id': frame['origin_id'],
                                 'frame_facing_ids': list(frame['facing_ids']),
                                 'signed_m': value,
                                 'passes': (True if direction_status == 'pass' else
                                            False if direction_status == 'fail' else None),
                                 'status': direction_status}
                    geometry['direction_evidence'].append(direction)
                    measured.append(direction)
                    if direction_status == 'unknown':
                        missing.append('direction relation geometry is uncertain near the observer-frame boundary')
                    else:
                        passed = passed and direction['passes']
            if passed:
                possible.append(oid)
        hypothesis = {'observer': frame, 'references': {n: o['bbox_id'] for n, o in binding.items()},
                      'candidate_ids': possible, 'measurements': measured}
        if local_contrast and apply_size_constraints and not possible:
            proofs = [hypothesis_fail_proof(hypothesis, obj['bbox_id'], len(hypotheses))
                      for obj in eligible]
            if proofs and all(proof['removal_allowed'] for proof in proofs):
                geometry.setdefault('rejected_hypotheses', []).append({
                    **hypothesis, 'candidate_removal_audit': proofs})
                continue
        hypotheses.append(hypothesis)
    sets = [set(h['candidate_ids']) for h in hypotheses]
    possible = sorted(set.union(*sets)) if sets else []
    stable = sorted(set.intersection(*sets)) if sets else []
    # Hypothesis evaluation is another candidate-removal path.  Reconcile
    # excluded IDs through the same evidence gate so an unbound relation or a
    # partially evaluated hypothesis cannot silently erase a candidate.
    possible_set = set(possible)
    for target in eligible:
        oid = target['bbox_id']
        if oid in possible_set:
            continue
        hypothesis_proofs = []
        for hypothesis_index, hypothesis in enumerate(
                hypotheses + geometry.get('rejected_hypotheses', [])):
            hypothesis_proofs.append(hypothesis_fail_proof(hypothesis, oid, hypothesis_index))
        relation_fail = bool(hypothesis_proofs) and all(proof['removal_allowed'] for proof in hypothesis_proofs)
        scopes = sorted({value for proof in hypothesis_proofs
                         for value in proof.get('comparison_scope_ids', [])})
        audit = candidate_removal_gate(
            {'candidate_id': oid, 'status': 'fail' if relation_fail else 'unknown',
             'hypothesis_fail_evidence': hypothesis_proofs,
             'reason': 'all bound relation interpretations fail the predicate'
                       if relation_fail else 'relation interpretation is incomplete'},
            source_quote=next((proof.get('source_quote', '') for proof in hypothesis_proofs
                               if proof.get('source_quote')), ''),
            target_role='target',
            role_binding_status='bound' if relation_fail else 'unknown',
            comparison_scope_ids=scopes,
            comparison_scope_status='bound' if relation_fail else 'unknown',
            measurement_source='; '.join(sorted({proof.get('measurement_source', '')
                                                for proof in hypothesis_proofs if proof.get('measurement_source')})),
            explicit_fail_criterion=('all bound relation measurements are explicit failures'
                                     if relation_fail else None))
        geometry['candidate_removal_audit'].append(audit)
        if not audit['removal_allowed']:
            possible_set.add(oid)
            missing.append('candidate relation exclusion lacks complete fail evidence: ' + str(oid))
    possible = sorted(possible_set)
    geometry['reference_binding_status'] = ('unknown' if unverified_reference_bindings
                                            else 'bound')
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
    candidate_hypothesis_support = _other_two_hypothesis_support(
        query, parsed, category, objects, candidates, eligible, hypotheses,
        relation_anchor_slots, trusted_categories)
    # VLM support covers category AND its supplied appearance attributes, not identity proof.
    needs = []
    def need(oid, name, role):
        name = retrieval_category(name)
        key = (oid, name, role)
        if verdict(oid, name, role) == 'uncertain' and not any((v['id'], v['category'], v['role']) == key for v in needs):
            language_role = 'target' if role == 'target' else 'reference'
            source_attributes = attributes_for(parsed, name, role=language_role)
            needs.append({'id': oid, 'category': name,
                          'attributes': appearance_attributes_for(parsed, name, role=language_role),
                          'source_attributes': source_attributes, 'role': role})
    for target in eligible:
        need(target['bbox_id'], category, 'target')
    for hypothesis in hypotheses:
        for name, oid in hypothesis['references'].items():
            need(oid, reference_slot_categories.get(name, name), 'reference')
        frame = hypothesis['observer']
        if frame:
            observer = parsed['Observer']
            if frame['origin_id'] is not None:
                need(frame['origin_id'], observer['origin']['category'], 'observer_origin')
            for oid in frame['facing_ids']:
                need(oid, observer['facing']['category'], 'observer_facing')
    anchor_names = parsed.get('Anchor', [])
    anchor_names = [anchor_names] if isinstance(anchor_names, str) else anchor_names
    for name in anchor_names:
        name = retrieval_category(name)
        if (name and name not in reference_names and
                attributes_for(parsed, name, role='reference')):
            for reference in match_retrieval_categories([name], objects):
                need(reference['bbox_id'], name, 'reference')
    if needs:
        missing.append('category/appearance evidence is uncertain or unreviewed')
    missing = list(dict.fromkeys(missing))
    if missing:
        status = 'missing'
    elif not eligible or ((sets or geometry.get('rejected_hypotheses')) and not possible):
        status = 'conflict'
    elif len(possible) != 1 or any(len(s) != 1 for s in sets):
        status = 'ambiguous'
    else:
        status = 'resolved'
    return {'status': status, 'missing': missing, 'candidate_ids': possible, 'stable_candidate_ids': stable,
            'pending_size_conditions': pending_size_conditions,
            'candidate_removal_audit': list(geometry['candidate_removal_audit']),
            'hypotheses': hypotheses, 'visual_needs': needs, 'checks': list(checks),
            'candidate_hypothesis_support': candidate_hypothesis_support,
            'geometry': geometry,
            'scope': 'bounded proposed instances, selected distance metrics, parser and VLM evidence; not linguistic uniqueness',
            'room_center': list(room_center),
            'room_center_definition': 'center of point-cloud AABB, approximation only',
            'hypotheses_are_votes': False, 'category_source': 'annotated' if trusted_categories else 'predicted',
            'category_policy': 'annotated categories trusted; appearance still requires evidence' if trusted_categories
                               else 'category and appearance require visual support'}
