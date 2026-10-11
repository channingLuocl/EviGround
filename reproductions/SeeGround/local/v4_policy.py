"""Recorded v4 ablations and fixed resource bounds (no learned scores)."""
import os


def enabled(feature=None):
    active = os.getenv('SEEGROUND_V4', '1') == '1'
    return active and (feature is None or os.getenv('SEEGROUND_V4_' + feature.upper(), '1') == '1')


def settings():
    return {'version': 'relative-size-v4' if enabled() else 'relative-size-v3',
            'inventory': enabled('inventory'), 'backtrack': enabled('backtrack'),
            'final_review': enabled('final_review'), 'size': enabled('size'),
            'relations': enabled('relations'),
            'dynamic_threshold': enabled() and os.getenv('SEEGROUND_V4_DYNAMIC_THRESHOLD', '0') == '1',
            'backtrack_limit': 1, 'backtrack_evidence_reserve': 1,
            'size_visual_calls_limit': 2, 'relation_visual_calls_limit': 2,
            'final_review_calls_limit': 2, 'views_per_visual_call': 1,
            'relaxed_relative_margin_floor': .05,
            'relaxed_length_margin_floor_m': .03}


def evidence_inventory(query, parsed):
    """Retain parser coverage gaps and roles, without an execution schedule."""
    from diagnostic import size_conditions
    rows = []
    for condition in parsed.get('condition_records', []):
        kind = condition.get('condition_type', '')
        dimension = ('size' if kind == 'size' else 'distance' if kind in
                     ('relation:near', 'relation:nearest', 'relation:farthest') else
                     'position' if kind.startswith('relation:') else
                     'viewpoint' if 'observer' in kind or 'viewpoint' in kind else
                     'appearance' if 'attribute' in kind or 'category' in kind else 'unresolved')
        rows.append({'source_quote': condition.get('source_quote'), 'dimension': dimension,
                     'target_role': condition.get('target_role'),
                     'reference_role': condition.get('reference_role'),
                     'condition': kind, 'binding_status': condition.get('binding_status'),
                     'dependencies': ['bind reference'] if condition.get('reference_role') else []})
    for category in parsed.get('Target', []):
        for spec in size_conditions(parsed, category):
            rows.append({'source_quote': spec.get('source_phrase', spec['original_quote']),
                         'dimension': 'size', 'target_role': {'category': category},
                         'condition': spec['word'], 'metric': spec['metric'],
                         'scope_kind': spec['scope_kind'], 'scope_phrase': spec.get('scope_phrase'),
                         'member_count': spec.get('required_member_count'),
                         'attachment': spec.get('syntax_attachment'),
                         'dependencies': ['bind comparison members'] + spec.get('scope_relation_quotes', [])})
    for phrase in parsed.get('Unhandled', []):
        if phrase.casefold() in query.casefold():
            rows.append({'source_quote': phrase, 'dimension': 'unresolved',
                         'condition': 'unhandled', 'dependencies': ['obtain quoted predicate evidence']})
    return {'items': rows, 'role_coreferences': parsed.get('target_coreferences', []),
            'policy': 'requirements and dependencies only; planner chooses order'}


def deletion_events(before, after, dimension, evidence, diagnosis, index):
    events = []
    for oid in before:
        if oid in after:
            continue
        proofs = [row for row in evidence if row.get('candidate_id') == oid]
        events.append({'event_id': index + len(events), 'candidate_id': oid,
                       'dimension': dimension, 'source_quotes': list(dict.fromkeys(
                           row.get('source_quote', '') for row in proofs)),
                       'evidence': proofs, 'comparison_bindings': diagnosis.get('geometry', {}).get('size_constraints', []),
                       'before_ids': list(before), 'after_ids': list(after),
                       'before_state': {'status': diagnosis.get('status'),
                                        'missing': list(diagnosis.get('missing', [])),
                                        'hypotheses': diagnosis.get('hypotheses', [])},
                       'invalidated': False})
    return events


def backtrack_events(events, problem, used=False):
    """Select affected deletions only; a bare model disagreement is insufficient."""
    if used or not problem.get('supported'):
        return []
    ids = set(problem.get('candidate_ids', []))
    quotes = set(problem.get('source_quotes', []))
    return [event for event in events if not event['invalidated'] and
            (event['candidate_id'] in ids or quotes.intersection(event['source_quotes']))]
