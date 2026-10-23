"""Quoted query obligations and a broad local graph; language never deletes IDs."""
from categories_v9 import category_fields
import re

from evidence import match_retrieval_categories, retrieval_category


NEGATION = r"\b(?:no|not|nothing|never|without|except)\b|n['’]t\b"


def exclusion_clauses(query, syntax):
    if not re.search(NEGATION, query, re.I):
        return []
    if syntax.get('backend') != 'spacy':
        return [{'source_quote': query, 'source_span': [0, len(query)],
                 'polarity': 'unknown', 'modified_category': None}]
    from query_syntax import _pipeline, _span
    doc = _pipeline()(query)
    clauses = []
    for token in doc:
        if not re.search(NEGATION, token.text, re.I):
            continue
        head = token if token.lower_ in ('without', 'except') else token.head
        tokens = list(head.subtree)
        if head.pos_ in ('NOUN', 'PROPN', 'NUM'):
            tokens = [t for t in tokens if t.i >= token.i]
        span = _span(doc, tokens)
        # A predicate's relative-clause subject identifies the modified noun.
        noun = head if head.pos_ in ('NOUN', 'PROPN') else head.head
        category = noun.lemma_ if noun.pos_ in ('NOUN', 'PROPN') else None
        predicate = [head, token] + [t for t in head.children if t.dep_ in ('amod', 'acomp', 'attr')]
        from v4_policy import v7_enabled
        if v7_enabled():
            # Postnominal contrast (table not the white one) does not negate
            # the earlier adjectives of the syntactic head.
            predicate = tokens if head.pos_ in ('NOUN', 'PROPN', 'NUM') or head.lower_ in ('without', 'except') else predicate
        clauses.append({'source_quote': span['text'], 'source_span': [span['start'], span['end']],
                        'predicate_spans': [[t.idx, t.idx + len(t)] for t in predicate],
                        'polarity': 'exclude', 'modified_category': category})
    return clauses or [{'source_quote': query, 'source_span': [0, len(query)],
                        'polarity': 'unknown', 'modified_category': None}]


def quote_spans(query, quote):
    return [[m.start(), m.end()] for m in re.finditer(re.escape(quote), query, re.I)] if quote else []


def polarity(query, quote, exclusions, attribute=False):
    spans = quote_spans(query, quote)
    if not spans:
        return 'unknown'
    if len(spans) > 1:
        if not exclusions:
            return 'positive'
        return 'unknown'
    start, end = spans[0]
    overlapping = [c for c in exclusions if any(start < b and end > a for a, b in
        (c.get('predicate_spans', [c['source_span']]) if attribute else [c['source_span']]))]
    if not overlapping:
        return 'positive'
    return 'exclude' if any(c['polarity'] == 'exclude' for c in overlapping) else 'unknown'


def retain_positive_predicates(query, parsed):
    """Keep exclusions in the ledger, outside the existing positive executors."""
    exclusions = exclusion_clauses(query, parsed.get('query_syntax', {}))
    from v4_policy import v7_enabled
    if v7_enabled():
        from v7_evidence import bind_negative_attributes
        bind_negative_attributes(query, parsed, exclusions)
        parsed['Relations'] = [r for r in parsed.get('Relations', [])
            if polarity(query, r.get('phrase', ''), exclusions) == 'positive']
        return exclusions
    for field in ('Attributes', 'Relations'):
        parsed[field] = [row for row in parsed.get(field, [])
                         if polarity(query, row.get('phrase', ''), exclusions,
                                     attribute=field == 'Attributes') == 'positive']
    return exclusions


def structure(query, parsed):
    exclusions = exclusion_clauses(query, parsed.get('query_syntax', {}))
    conditions = []
    for record in parsed.get('condition_records', []):
        quote = record.get('source_quote')
        value = polarity(query, quote, exclusions, attribute=record['condition_type'] in ('appearance', 'size'))
        role = record.get('target_role') or record.get('reference_role') or record.get('other_role')
        from v4_policy import v7_enabled
        negative_attribute = (v7_enabled() and record['condition_type'] == 'appearance' and
            any(a['phrase'] == quote and a.get('polarity') == 'exclude' for a in parsed.get('Attributes', [])))
        bound = record.get('binding_status') == 'bound' and (value == 'positive' or negative_attribute)
        conditions.append({**record, 'condition_id': f'c{len(conditions)}',
            'source_spans': quote_spans(query, quote), 'polarity': value,
            'modified_role': role, 'parse_status': 'bound' if bound else 'unknown',
            **({'evidence_status': 'unsupported', 'binding_status': 'unknown'} if value != 'positive' and not negative_attribute else {}),
            **({'executor': 'negative_color', 'evidence_status': 'pending'} if negative_attribute else {})})
    for clause in exclusions:
        category = retrieval_category(clause.get('modified_category') or '')
        roles = [role for role, names in (('target', parsed.get('Target', [])),
                                          ('reference', parsed.get('Anchor', [])))
                 if category and category in {retrieval_category(n) for n in names}]
        executed = v7_enabled() and clause['source_quote'] in parsed.get('_v7_executed_negations', [])
        conditions.append({**clause, 'condition_id': f'c{len(conditions)}',
            'condition_type': 'exclusion', 'parse_status': 'unknown', 'evidence_status': 'unsupported',
            'modified_role': {'role': roles[0], 'category': category} if len(roles) == 1 else None,
            'reason': 'exclusion requires a source-bound predicate; positive evidence cannot execute it',
            **({'parse_status': 'bound', 'evidence_status': 'pending', 'executor': 'negative_color',
                'reason': 'source-bound negative color is sent to appearance verification'} if executed else {})})
    # Unrepresented language is visible to the planner instead of being dropped.
    covered = {i for c in conditions for start, end in c.get('source_spans', [c.get('source_span', [0, 0])])
               for i in range(start, end)}
    covered.update(i for c in conditions for hint in c.get('language_comparisons', [])
                   for i in range(*hint['source_span']))
    gaps = [t['text'] for t in parsed.get('query_syntax', {}).get('dependencies', [])
            if (t['pos'] in ('ADJ', 'ADV') or t['dependency'] in ('acl', 'amod')) and t['text'].casefold() not in
            ('the', 'a', 'an', 'there', 'please') and
            not any(i in covered for i in range(t['start'], t['end']))]
    for quote in dict.fromkeys(gaps):
        conditions.append({'condition_id': f'c{len(conditions)}', 'condition_type': 'coverage_gap',
            'source_quote': quote, 'source_spans': quote_spans(query, quote), 'polarity': 'unknown',
            'modified_role': None, 'parse_status': 'unknown', 'evidence_status': 'missing',
            'reason': 'modifier or connector is not represented by a quoted condition'})
    return {'target_categories': parsed.get('Target', []), 'reference_categories': parsed.get('Anchor', []),
            'conditions': conditions, 'uncovered_modifiers': list(dict.fromkeys(gaps)),
            'policy': 'quoted language hints only; unknown/exclusion conditions cannot remove candidates'}


def local_graph(parsed, objects, candidates, diagnosis):
    obligations = parsed.get('query_structure') or structure(parsed.get('source_query', ''), parsed)
    categories = list(parsed.get('Anchor', []))
    for c in obligations['conditions']:
        role = c.get('reference_role') or c.get('other_role') or {}
        categories += role.get('categories', [])
        if role.get('category'):
            categories.append(role['category'])
    # Include quoted noun categories even when the parser missed a reference.
    categories += [n['head'] for n in parsed.get('query_syntax', {}).get('noun_phrases', [])]
    categories += ['wall', 'floor']
    targets = {o['bbox_id'] for o in candidates}
    relevant = {o['bbox_id'] for o in match_retrieval_categories(categories, objects)} | targets
    bindings = []
    for category in sorted({retrieval_category(c) for c in categories if c}):
        pool = match_retrieval_categories([category], objects)
        if not pool:
            continue
        attrs = [c for c in obligations['conditions'] if
                 (c.get('reference_role') or {}).get('category') == category]
        proofs = [r for r in diagnosis.get('geometry', {}).get('candidate_removal_audit', [])
                  if r.get('target_role') == 'reference' and r.get('candidate_id') in
                  {o['bbox_id'] for o in pool}]
        excluded = {r['candidate_id'] for r in proofs if r.get('removal_allowed')}
        bindings.append({'category': category, 'candidate_ids': [o['bbox_id'] for o in pool if o['bbox_id'] not in excluded],
            'initial_candidate_ids': [o['bbox_id'] for o in pool],
            'excluded_ids': sorted(excluded),
            'description_condition_ids': [c['condition_id'] for c in attrs],
            'binding_status': 'pending_evidence',
            'policy': 'retain every category instance; no nearest or first-instance binding'})
    edges = [{'condition_id': c['condition_id'], 'target_role': c.get('target_role'),
              'reference_role': c.get('reference_role'), 'modified_role': c.get('modified_role'),
              'polarity': c['polarity'], 'parse_status': c['parse_status']}
             for c in obligations['conditions']]
    return {'nodes': [{'bbox_id': o['bbox_id'], 'category': o['target'], **category_fields(o),
                       'role': 'target' if o['bbox_id'] in targets else 'reference'}
                      for o in objects if o['bbox_id'] in relevant],
            'initial_target_ids': sorted(targets),
            'remaining_target_ids': diagnosis.get('candidate_ids', []),
            'reference_bindings': bindings, 'edges': edges,
            'comparison_scope_ids': diagnosis.get('size_comparison_pool_ids', sorted(targets)),
            'policy': 'local context only; complete target and reference pools stay in evidence executors'}
