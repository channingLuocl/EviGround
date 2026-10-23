"""Source-bound position, distance and size evidence; no model or rendering calls."""
from copy import deepcopy
import json

from eviground.grounding.diagnostic import (candidate_removal_gate, size_category_scope, size_conditions,
                        size_condition_scope)
from eviground.grounding.evidence import retrieval_category
from eviground.geometry.size_semantics import evaluate_size_condition
from eviground.grounding.v4_policy import v5_enabled, v7_enabled


def _status(flags):
    """Summarize conditional evidence without treating unknown as failure."""
    if not flags:
        return 'not_applicable'
    if any(flag is True for flag in flags):
        return 'pass'
    if any(flag is None for flag in flags):
        return 'unknown'
    return 'fail'


def _overall_status(groups):
    statuses = [group['status'] for group in groups.values()
                if group['status'] != 'not_applicable']
    if not statuses:
        return 'not_applicable'
    if 'fail' in statuses:
        return 'fail'
    if 'unknown' in statuses:
        return 'unknown'
    return 'pass'


def _relation_clause_key(item):
    if type(item.get('relation_index')) is int:
        return 'relation:' + str(item['relation_index'])
    phrase = item.get('phrase')
    if isinstance(phrase, str) and phrase.strip():
        anchor_category = item.get('anchor_category',
                                   item.get('anchor_binding', {}).get('anchor_category', ''))
        return json.dumps([item.get('relation'), phrase, item.get('metric'), anchor_category])
    return None


def _unresolved_relation_bindings(diagnosis):
    """A geometric fail cannot exclude through an unfinished language binding."""
    reasons = []
    for key in ('target_role_binding_status', 'reference_binding_status',
                'target_instance_binding_status'):
        if diagnosis.get('geometry', {}).get(key) == 'unknown':
            reasons.append(key)
    if any(item.get('role') == 'reference'
           for item in diagnosis.get('pending_size_conditions', [])):
        reasons.append('reference size is unresolved')
    if any(condition.get('role') == 'reference' and any(
            item.get('status') == 'unknown' for item in condition.get('candidate_evidence', []))
            for condition in diagnosis.get('geometry', {}).get('size_constraints', [])):
        reasons.append('reference size measurement is unknown')
    if any(item.get('role') != 'target' for item in diagnosis.get('visual_needs', [])):
        reasons.append('reference category/appearance is unresolved')
    return reasons


def position_candidate_evidence(diagnosis, candidate_ids=None):
    """AND clauses after retaining alternative bindings within each clause."""
    geometry = diagnosis.get('geometry', {})
    ids = list(candidate_ids if candidate_ids is not None
               else diagnosis.get('candidate_ids', []))
    rows = (list(geometry.get('relation_pairs', [])) +
            list(geometry.get('direction_evidence', [])))
    triplets = geometry.get('between_triplets', [])
    if triplets:
        rows.extend(triplets)
    else:
        for item in geometry.get('between_candidate_evidence', []):
            for key, passes in (('passing_reference_pairs', True),
                                ('unknown_reference_pairs', None),
                                ('failing_reference_pairs', False)):
                rows.extend({**item, 'relation': 'between', 'passes': passes}
                            for _ in item.get(key, []))
    clauses = {_relation_clause_key(row) or 'unidentified_condition' for row in rows}
    records = []
    for target_id in ids:
        target_rows = [pair for pair in rows if pair.get('target_id') == target_id]
        groups = {clause: [] for clause in clauses}
        for pair in target_rows:
            clause = _relation_clause_key(pair)
            groups[clause or 'unidentified_condition'].append(
                pair.get('passes') if clause else None)
        normalized = {
            clause: {'status': _status(flags or [None]), 'observations': len(flags)}
            for clause, flags in groups.items()
        }
        records_for_id = {
            'candidate_id': target_id,
            'status': _overall_status(normalized),
            'relations': normalized,
            'evidence_scope': 'predicate-local clause summary; complete-query support requires one shared hypothesis',
        }
        records_for_id['exclusion_reason'] = (
            'all available reference interpretations fail the position predicate'
            if records_for_id['status'] == 'fail' else None)
        anchors = [pair.get('anchor_id') for pair in target_rows
                   if type(pair.get('anchor_id')) is int]
        for pair in target_rows:
            for key in ('anchor_ids', 'passing_reference_pairs',
                        'unknown_reference_pairs', 'failing_reference_pairs', 'frame_facing_ids'):
                values = pair.get(key, [])
                if key in ('anchor_ids','frame_facing_ids'):
                    values = [values]
                for value in values:
                    if isinstance(value, (list, tuple)):
                        anchors.extend(item for item in value if type(item) is int)
        records_for_id.update(
            source_quote=next((pair.get('phrase', '') for pair in target_rows
                               if pair.get('phrase') and normalized.get(
                                   _relation_clause_key(pair), {}).get('status') == 'fail'),
                              next((pair.get('phrase', '') for pair in target_rows
                                    if pair.get('phrase')), '')),
            role_binding_status=('bound' if target_rows and anchors and all(
                group['status'] != 'unknown' for group in normalized.values()) else 'unknown'),
            reference_role=('reference' if anchors else None),
            comparison_scope_ids=sorted(set([target_id, *anchors])),
            comparison_scope_status=('bound' if target_rows and anchors and all(
                group['status'] != 'unknown' for group in normalized.values()) else 'unknown'),
            measurement_source='annotated_box relation geometry',
            explicit_fail_criterion=records_for_id['exclusion_reason'],
            measurements=target_rows)
        unresolved = _unresolved_relation_bindings(diagnosis)
        if unresolved:
            records_for_id.update(status='unknown', role_binding_status='unknown',
                                  comparison_scope_status='unknown',
                                  binding_missing=unresolved)
        records.append(records_for_id)
    return records


def distance_candidate_evidence(rankings, candidate_ids, diagnosis=None):
    """AND distance clauses after ORing their alternative reference bindings."""
    result = []
    clauses = {_relation_clause_key(row) or 'unidentified_condition' for row in rankings}
    for candidate_id in candidate_ids:
        groups = {clause: [] for clause in clauses}
        for ranking in rankings:
            ordered = {item.get('target_id') for item in ranking.get('ordered_targets', [])}
            included = ordered | set(ranking.get('unknown_target_ids', []))
            if candidate_id not in included:
                continue
            clause = _relation_clause_key(ranking)
            groups[clause or 'unidentified_condition'].append(
                None if not clause or ranking.get('comparison_status') == 'unknown' else
                candidate_id in set(ranking.get('best_ids', [])))
        normalized = {clause: {'status': _status(flags or [None]), 'observations': len(flags)}
                      for clause, flags in groups.items()}
        result.append({'candidate_id': candidate_id,
                       'status': _overall_status(normalized),
                       'relations': normalized,
                       'evidence_scope': 'predicate-local clause summary; complete-query support requires one shared hypothesis',
                       'exclusion_reason': (
                           'all available bindings fail at least one explicit distance clause'
                           if _overall_status(normalized) == 'fail' else None),
                       'source_quote': next((row.get('phrase', '') for row in rankings
                                             if candidate_id in ({item.get('target_id') for item in row.get('ordered_targets', [])} |
                                                                 set(row.get('unknown_target_ids', []))) and row.get('phrase') and
                                             normalized.get(_relation_clause_key(row), {}).get('status') == 'fail'),
                                            next((row.get('phrase', '') for row in rankings if row.get('phrase')), '')),
                       'role_binding_status': ('bound' if any(
                           candidate_id in ({item.get('target_id') for item in row.get('ordered_targets', [])} |
                                             set(row.get('unknown_target_ids', []))) and
                           type(row.get('anchor_id')) is int for row in rankings) else 'unknown'),
                       'comparison_scope_ids': sorted({candidate_id, *[
                           row.get('anchor_id') for row in rankings
                           if type(row.get('anchor_id')) is int], *[
                           oid for row in rankings for oid in row.get('comparison_pool_ids', [])
                           if type(oid) is int], *[
                           item['target_id'] for row in rankings for item in row.get('ordered_targets', [])
                           if type(item.get('target_id')) is int]}),
                       'comparison_scope_status': ('bound' if rankings and all(
                           type(row.get('anchor_id')) is int and
                           row.get('scope_status', 'bound') == 'bound' and
                           row.get('comparison_status') != 'unknown'
                           for row in rankings) and all(
                           group['status'] != 'unknown' for group in normalized.values()) else 'unknown'),
                       'measurement_source': 'distance ranking geometry',
                       'measurements': [row for row in rankings if candidate_id in (
                           {item.get('target_id') for item in row.get('ordered_targets', [])} |
                           set(row.get('unknown_target_ids', [])))],
                       'explicit_fail_criterion': (
                           'candidate is not in the best_ids of a bound nearest/farthest ranking'
                           if _overall_status(normalized) == 'fail' else None)})
        unresolved = _unresolved_relation_bindings(diagnosis or {})
        if unresolved:
            result[-1].update(status='unknown', role_binding_status='unknown',
                              comparison_scope_status='unknown', binding_missing=unresolved)
    return result


def _target_size_scope(parsed, diagnosis, candidates):
    """Keep the initial complete category pool; each quoted condition owns its scope."""
    seed = list(diagnosis.get('size_comparison_pool_ids') or [o['bbox_id'] for o in candidates])
    category_scope = size_category_scope([o for o in candidates if o['bbox_id'] in seed])
    return seed, category_scope is None, category_scope.get('unknown_ids', []) if category_scope else []


def size_candidate_evidence(parsed, candidates, measurements, comparison_pool_ids=None,
                            comparison_scope_bound=None, objects=None, checks=()):
    """Use the same source-bound predicates as ordinary diagnosis and replay."""
    by_id = {row['id']: row for row in measurements}
    pool_ids = list(dict.fromkeys(comparison_pool_ids if comparison_pool_ids is not None else by_id))
    if comparison_scope_bound is None:
        comparison_scope_bound = True
    names = parsed.get('Target', [])
    category = retrieval_category(names[0]) if names else ''
    constraints = []
    contrast = parsed.get('relative_height_binding')
    for spec in size_conditions(parsed, category):
        if contrast and spec['word'] == contrast['target_word']:
            continue  # compute_observation binds the two references in the joint diagnosis.
        if v5_enabled('scope') and spec.get('scope_hypotheses') and objects is not None:
            from eviground.grounding.diagnostic import evaluate_scope_hypotheses
            constraints.append(evaluate_scope_hypotheses(parsed, objects, category, spec, checks,
                measurements, [o['bbox_id'] for o in candidates], comparison_scope_bound))
            continue
        scoped_ids, bound, source, unknown = pool_ids, comparison_scope_bound, 'frozen initial category pool', []
        if objects is not None:
            pool, scope_record = size_condition_scope(parsed, objects, category, spec, checks)
            scoped_ids = [o['bbox_id'] for o in pool]
            bound = bound and scope_record['status'] == 'bound'
            source, unknown = scope_record['source'], scope_record.get('unknown_ids', [])
        elif spec.get('scope_kind') == 'attribute_subset':
            bound = False
        if v7_enabled() and not bound and spec.get('measurement_policy') == 'provided_instance_box':
            scoped_ids = pool_ids
            source = 'diagnostic full-category box ordering; quoted modifier scope unresolved'
        if parsed.get('_v4_relax_once'):
            spec = {**spec, 'relax_once': True}
        constraints.append(evaluate_size_condition(spec, measurements, scoped_ids,
            scope_bound=bound, candidate_ids=[o['bbox_id'] for o in candidates],
            scope_source=source, scope_unknown_ids=unknown))
        if spec['original_quote'] in parsed.get('_v4_invalidated_conditions', []):
            constraint = constraints[-1]
            constraint.update(status='unknown', selected_ids=[], tentative_selected_ids=[],
                              reason='predicate invalidated by supported backtrack; new binding/measurement evidence required')
            for item in constraint['candidate_evidence']:
                item['status'] = 'unknown'
    records = []
    for obj in candidates:
        oid = obj['bbox_id']
        proof = [next((e for e in c['candidate_evidence'] if e['candidate_id'] == oid),
                       {'candidate_id': oid, 'status': 'unknown'}) for c in constraints]
        flags = [e['status'] for e in proof]
        status = ('fail' if 'fail' in flags else 'unknown' if 'unknown' in flags else
                  'pass' if flags else 'not_applicable')
        decisive = next((c for e, c in zip(proof, constraints) if e['status'] == 'fail'),
                        constraints[0] if constraints else {})
        scopes = decisive.get('comparison_pool_ids', [])
        role_status = ('bound' if decisive.get('scope_status') == 'bound' and
                       decisive.get('role', 'target') == 'target' else 'unknown')
        source = decisive.get('measurement_sources', {}).get(str(oid), '')
        criterion = 'reliable measurement contradicts the quoted size condition' if status == 'fail' else None
        records.append({'candidate_id': oid, 'status': status, 'metrics': by_id.get(oid, {}),
                        'attribute_evidence': [{**e, 'attribute': c['attribute'],
                            'original_quote': c['original_quote']} for e, c in zip(proof, constraints)],
                        'exclusion_reason': 'explicit size predicate failed' if status == 'fail' else None,
                        'source_quote': decisive.get('original_quote', ''),
                        'target_role': decisive.get('role', 'target'),
                        'role_binding_status': role_status,
                        'comparison_scope_ids': scopes,
                        'comparison_scope_status': role_status,
                        'measurement_source': source,
                        'explicit_fail_criterion': criterion})
    unknown = sorted({oid for c in constraints for oid in c['unknown_measurement_ids']})
    return records, constraints, {
        'attributes': constraints, 'candidate_count': len(candidates),
        'comparison_pool_ids': sorted(pool_ids), 'comparison_pool_size': len(pool_ids),
        'comparison_pool_frozen': bool(comparison_scope_bound),
        'scope_status': 'bound' if comparison_scope_bound else 'unknown',
        'scope_missing': list(dict.fromkeys(c['reason'] for c in constraints
                                            if c['scope_status'] == 'unknown' and c['reason'])),
        'unknown_measurement_ids': unknown,
        'metric_ambiguities': {c['attribute']: c['metric_ambiguity'] for c in constraints
                              if c.get('metric_ambiguity')},
        'candidate_relative_ids': [r['candidate_id'] for r in records if r['status'] == 'pass'],
        'tentative_selected_ids': sorted(set.intersection(*[
            set(c.get('tentative_selected_ids') or c.get('selected_ids', [])) for c in constraints]))
            if constraints and any(c.get('tentative_selected_ids') for c in constraints) else [],
        'policy': 'source words and per-condition frozen scopes; only explicit fail excludes; '
                  'size adjectives never become superlatives and count is not a size metric'}


def apply_computed_evidence(diagnosis, evidence, dimension, record=True, size_constraints=()):
    """Apply only definite failures from a computed dimension to the live scope."""
    unresolved = (_unresolved_relation_bindings(diagnosis)
                  if dimension in ('position', 'distance') else [])
    for reason in unresolved:
        missing = 'relation evidence has unresolved binding: ' + reason
        if missing not in diagnosis.setdefault('missing', []):
            diagnosis['missing'].append(missing)
    scope_ids = []
    for constraint in size_constraints:
        scope_ids.extend(constraint.get('comparison_pool_ids', []))
    scope_ids = sorted({oid for oid in scope_ids if type(oid) is int})
    gated = []
    conflicting_ids = {item.get('candidate_id') for item in evidence
                       if sum(other.get('candidate_id') == item.get('candidate_id')
                              for other in evidence) > 1}
    for item in evidence:
        if unresolved:
            item = {**item, 'status': 'unknown', 'role_binding_status': 'unknown',
                    'comparison_scope_status': 'unknown', 'binding_missing': unresolved}
        if item.get('candidate_id') in conflicting_ids:
            item = {**item, 'status':'unknown', 'role_binding_status':'conflict',
                    'reason':'duplicate candidate evidence records conflict'}
        constraint = next((row for row in size_constraints
                           if row.get('role', 'target') == 'target' and
                           item.get('candidate_id') in row.get('comparison_pool_ids', []) and
                           any(e.get('candidate_id') == item.get('candidate_id') and
                               e.get('status') == item.get('status')
                               for e in row.get('candidate_evidence', []))), None)
        constraint_sources = (constraint or {}).get('measurement_sources', {})
        constraint_role = (constraint or {}).get('role', 'target')
        constraint_scope = (constraint or {}).get('comparison_pool_ids', scope_ids)
        # Every computed dimension supplies its own predicate-local scope.  A
        # missing scope is intentional evidence of an unbound condition and
        # must not become a hard filter through the shared update path.
        gated.append(candidate_removal_gate(
            item,
            source_quote=item.get('source_quote') or item.get('original_quote') or
                (constraint or {}).get('original_quote', ''),
            target_role=item.get('target_role') or constraint_role,
            reference_role=item.get('reference_role'),
            # Legacy/synthetic records without an explicit role binding are
            # intentionally retained; a status string alone is not proof.
            role_binding_status=item.get('role_binding_status') or
                ('bound' if constraint and constraint.get('scope_status') == 'bound' else 'unknown'),
            comparison_scope_ids=item.get('comparison_scope_ids') or constraint_scope or
                scope_ids or
                item.get('scope_ids'),
            comparison_scope_status=item.get('comparison_scope_status') or
                (constraint or {}).get('scope_status', 'unknown'),
            measurement_source=(item.get('measurement_source') or
                constraint_sources.get(str(item.get('candidate_id')), '') or
                ((item.get('render_provenance') or {}).get('source_type', '')
                 if isinstance(item.get('render_provenance'), dict) else '')),
            explicit_fail_criterion=item.get('explicit_fail_criterion') or
                ('reliable measurement contradicts the quoted size condition'
                 if item.get('status') == 'fail' and constraint else None)))
        if gated[-1].get('target_role') != 'target':
            gated[-1]['removal_allowed'] = False
            gated[-1]['removal_blocked_reason'] = 'reference evidence cannot remove a target candidate'
    by_id = {item['candidate_id']: item for item in gated}
    before = list(diagnosis.get('candidate_ids', []))
    if conflicting_ids & set(before):
        diagnosis.setdefault('missing', []).append('conflicting candidate evidence records')
    for item in gated:
        if item.get('candidate_id') in before and item.get('status') == 'fail' and not item['removal_allowed']:
            reason = 'candidate exclusion blocked: ' + str(item['candidate_id']) + ': ' + item['removal_blocked_reason']
            if reason not in diagnosis.setdefault('missing', []):
                diagnosis['missing'].append(reason)
    removed = [oid for oid in before if by_id.get(oid, {}).get('removal_allowed')]
    remaining = [oid for oid in before if oid not in set(removed)]
    hypotheses = diagnosis.get('hypotheses', [])
    candidate_sets = []
    if hypotheses:
        allowed = set(remaining)
        for hypothesis in hypotheses:
            hypothesis['candidate_ids'] = sorted(
                allowed & set(hypothesis.get('candidate_ids', [])))
            candidate_sets.append(set(hypothesis['candidate_ids']))
        diagnosis['stable_candidate_ids'] = sorted(set.intersection(*candidate_sets))
    else:
        diagnosis['stable_candidate_ids'] = [oid for oid in diagnosis.get('stable_candidate_ids', [])
                                             if oid in remaining]
    diagnosis['candidate_ids'] = remaining
    pending_size = diagnosis.get('pending_size_conditions', [])
    if dimension == 'size' and pending_size and remaining:
        evaluated = {(item.get('attribute'), item.get('role', 'target')) for item in size_constraints}
        # A request alone is not evidence: every deferred predicate must be
        # represented and every surviving candidate must have a definite pass.
        if (all((item['attribute'], item.get('role', 'target')) in evaluated for item in pending_size) and
                all(by_id.get(oid, {}).get('status') == 'pass' for oid in remaining)):
            diagnosis['pending_size_conditions'] = []
            diagnosis['missing'] = [reason for reason in diagnosis.get('missing', [])
                                    if reason != 'target size conditions lack complete evidence']
    if record:
        diagnosis.setdefault('active_computed_evidence', []).append({
            'dimension': dimension, 'evidence': gated, 'removed_ids': removed,
            'candidate_removal_audit': [item for item in gated
                                        if item.get('candidate_id') in before],
            **({'size_constraints': list(size_constraints)} if dimension == 'size' else {}),
        })
    if not remaining:
        diagnosis['status'] = ('missing' if not before and diagnosis.get('missing')
                               and diagnosis.get('status') == 'missing' else 'conflict')
    elif diagnosis.get('missing'):
        diagnosis['status'] = 'missing'
    elif len(remaining) != 1 or any(len(ids) != 1 for ids in candidate_sets):
        diagnosis['status'] = 'ambiguous'
    else:
        diagnosis['status'] = 'resolved'
    return {'before': before, 'removed_ids': removed, 'after': remaining}


def apply_size_observation(diagnosis, details, record=True):
    """Apply the shared joint diagnosis or ordinary per-condition evidence."""
    joint = details.get('joint_size_diagnosis')
    if joint is not None:
        return deepcopy(joint)
    apply_computed_evidence(diagnosis, details['candidate_evidence'], 'size', record=record,
                            size_constraints=details.get('size_constraints', []))
    return diagnosis

