"""Typed JEV decisions for the text-only outer evidence loop."""
from seeground.categories.categories_v9 import category_fields, CATEGORY_INSTRUCTIONS
import json
import os
from collections import Counter

from seeground.api.decision_api import DecisionAPIError, decide, jev_model
from seeground.api.qwen_api import api_provider
from seeground.grounding.v4_policy import v5_enabled


def react_backend():
    backend = os.getenv('SEEGROUND_REACT_BACKEND', 'jev').strip()
    if backend not in ('jev', 'qwen', 'openai'):
        raise ValueError('SEEGROUND_REACT_BACKEND must be jev, qwen or openai')
    return backend


def validate_react_config():
    backend = react_backend()
    if backend in ('qwen', 'openai') and backend != api_provider():
        raise ValueError('SEEGROUND_REACT_BACKEND must match SEEGROUND_API_PROVIDER')
    if backend == 'jev' and (not os.getenv('TYPESAFE_API_KEY', '').strip() or not jev_model()):
        raise ValueError('JEV ReAct requires TYPESAFE_API_KEY and SEEGROUND_JEV_MODEL; no Qwen fallback')
    return backend


def text_state(value):
    """Keep structured observations; image pixels and file paths stay in the trace."""
    if isinstance(value, dict):
        if value.get('joint_size_diagnosis') is not None:
            # All assignment proofs stay in the local trace. The planner uses
            # surviving candidate proofs and the controller's joint verdict.
            joint = value['joint_size_diagnosis']
            value = {key: item for key, item in value.items()
                     if key not in ('joint_size_diagnosis', 'size_constraints', 'size_decision')}
            value['joint_size_summary'] = {
                'status': joint['status'], 'missing': joint['missing'],
                'candidate_ids': joint['candidate_ids'],
                'stable_candidate_ids': joint['stable_candidate_ids'],
                'hypotheses': [{'references': h['references'], 'candidate_ids': h['candidate_ids']}
                               for h in joint['hypotheses']],
                'pending_size_conditions': joint['pending_size_conditions'],
            }
        return {key: text_state(item) for key, item in value.items()
                if key not in ('image', 'images', 'image_url', 'call', 'internal_steps')}
    if isinstance(value, list):
        return [text_state(item) for item in value]
    return value


def compact_repeated_evidence(result):
    """Share identical proof blocks across every planner evidence container."""
    if result.get('evidence_encoding'):
        return result, len(result.get('shared_evidence', {}))
    table_fields = {'candidate_evidence', 'candidate_removal_audit', 'relation_tasks',
                    '_v4_relation_evidence', 'reference_bindings', 'anchor_hypotheses',
                    'member_binding', 'scope_hypotheses'}
    # These values are read locally before transport or retained as direct
    # obligations. Encoding them as references would change execution shape.
    direct_fields = {'next_evidence', 'condition_records', 'missing', 'target_table',
                     'reference_table', 'category_candidates', 'remaining_candidate_ids',
                     'available_dimensions', 'evidence_capabilities', 'tools_remaining', 'deletion_events',
                     'distance_rankings'}
    def tables(value, field=None):
        if field in direct_fields:
            return value
        if isinstance(value, dict):
            return {key: tables(item, key) for key, item in value.items()}
        if isinstance(value, list):
            rows = [tables(item) for item in value]
            if (field in table_fields and len(rows) >= 3 and all(isinstance(row, dict) for row in rows)
                    and all(row.keys() == rows[0].keys() for row in rows)):
                shared = {key: value for key, value in rows[0].items()
                          if all(row[key] == value for row in rows[1:])}
                columns = [key for key in rows[0] if key not in shared]
                return {'table_shared': shared, 'table_columns': columns,
                        'table_rows': [[row[key] for key in columns] for row in rows]}
            return rows
        return value
    evidence = {key: tables(value, key) for key, value in result.items()}
    counts = Counter()
    def fingerprint(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    def count(value, field=None):
        if field in direct_fields:
            return
        if isinstance(value, (dict, list)):
            encoded = fingerprint(value)
            if len(encoded.encode()) >= 96:
                counts[encoded] += 1
            if isinstance(value, dict):
                for key, child in value.items():
                    count(child, key)
            else:
                for child in value:
                    count(child)
    for key, value in evidence.items():
        count(value, key)
    definitions, identifiers = {}, {}
    def share(value, field=None, container=False):
        if field in direct_fields:
            return value
        if isinstance(value, (dict, list)):
            encoded = fingerprint(value)
            if not container and counts[encoded] > 1:
                if encoded not in identifiers:
                    identifier = 'e' + str(len(identifiers) + 1)
                    identifiers[encoded] = identifier
                    definitions[identifier] = ({key: share(item, key) for key, item in value.items()}
                                               if isinstance(value, dict)
                                               else [share(item) for item in value])
                return {'shared_evidence_ref': identifiers[encoded]}
            if isinstance(value, dict):
                return {key: share(item, key, container=(key == 'observation'))
                        for key, item in value.items()}
            return [share(item, container=(field == 'history')) for item in value]
        return value
    # Keep each top-level container directly accessible to evidence_tasks and
    # jev_state; only their repeated structured contents use shared references.
    result = {key: share(value, key, container=True) for key, value in evidence.items()}
    if definitions:
        result['shared_evidence'] = definitions
    result['evidence_encoding'] = ('table_rows follow table_columns in order plus table_shared fields. shared_evidence_ref resolves '
        'the complete block in shared_evidence. All candidate/anchor IDs, roles, member counts, '
        'unknown/fail states and reasons are preserved; encoding never proves a condition.')
    return result, len(definitions)



def planner_state(state, tighter=False, instruction_bytes=0):
    """Keep obligations and proof states, not repeated point/measurement payloads."""
    if state.get('evidence_encoding'):
        # References and tables already encode complete summaries. Projecting
        # their keys again would erase proof fields; only trim summary history.
        result = {**state, 'history': state.get('history', [])[-(1 if tighter else 3):]}
        summary_bytes = len(json.dumps(result, ensure_ascii=False, separators=(',', ':')).encode())
        audit = {'method': 'utf8_bytes_upper_bound', 'input_units': summary_bytes + instruction_bytes + 64,
            'summary_bytes': summary_bytes, 'protocol_reserve': 64, 'instruction_bytes': instruction_bytes,
            'context_limit': int(os.getenv('SEEGROUND_CONTEXT_LIMIT', '32768')),
            'removed': [], 'history_entries': len(result['history']), 'tighter': tighter,
            'evidence_encoding': 'lossless_tables_and_shared_blocks',
            'shared_evidence_blocks': len(result.get('shared_evidence', {}))}
        if audit['input_units'] > audit['context_limit'] - 2048:
            if not tighter:
                return planner_state(state, tighter=True, instruction_bytes=instruction_bytes)
            raise ValueError('required planner obligations exceed configured text context budget')
        return result, audit
    drop = {'comparison_metrics', 'candidate_metrics', 'geometry_conditions', 'geometry_details',
            'annotated_box_metrics', 'provenance', 'comparison_bindings', 'before_state',
            'before', 'after', 'internal_steps', 'between_triplets', 'metrics', 'size_measurements'}
    drop.update({'_v5_visual_cache', '_v4_metric_choices', '_size_scope_bindings', '_v4_binding_changes',
                 'query_syntax', 'query_rewrite', 'active_computed_evidence', 'candidate_shape_evidence',
                 'render_provenance', 'checks', 'relation_pairs', 'visual_inspection', 'history',
                 'deletion_provenance', 'deletion_events', '_v7_measurement_rows'})
    removed = set()
    def proofs(rows):
        groups = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            # Preserve every obligation, failure reason and unknown candidate;
            # aggregate identical proof states while metrics stay in local trace.
            proof = {k: compact(row[k]) for k in ('status', 'source_quote', 'original_quote',
                'reason', 'target_role', 'role_binding_status', 'comparison_scope_status',
                'removal_allowed', 'explicit_fail_criterion', 'attribute_evidence', 'relation_evidence',
                'binding_missing', 'comparison_scope_ids', 'kind', 'target_category', 'anchor_category',
                'reference_role', 'reference_ids', 'anchor_ids', 'unknown_member_ids', 'unknown_measurement_ids',
                'geometry_evidence', 'visual_evidence', 'observations', 'exclusion_reason', 'missing',
                'condition_checks', 'source_quotes', 'reference_bindings', 'scope_hypotheses',
                'relation', 'support_subject', 'binding_status', 'next_evidence', 'anchor_hypotheses') if k in row}
            if 'attribute_evidence' in row:
                proof['attribute_evidence'] = [{k: compact(e[k]) for k in
                    ('status', 'tentative_status', 'reason', 'original_quote', 'source_quote', 'attribute',
                     'metric', 'role', 'target_role', 'reference_role', 'reference_ids', 'scope_kind',
                     'scope_status', 'scope_unknown_ids', 'comparison_pool_ids', 'unknown_measurement_ids',
                     'scope_hypotheses', 'removal_allowed', 'explicit_fail_criterion', 'role_binding_status',
                     'comparison_scope_status') if k in e}
                    for e in row['attribute_evidence']]
                for summary, original in zip(proof['attribute_evidence'], row['attribute_evidence']):
                    quality = original.get('measurement_reliability')
                    summary['measurement_status'] = quality.get('status') if isinstance(quality, dict) else quality
            key = json.dumps(proof, sort_keys=True)
            groups.setdefault(key, {**proof, 'candidate_ids': []})['candidate_ids'].extend(
                row.get('candidate_ids', [row['candidate_id']] if 'candidate_id' in row else []))
        return list(groups.values())

    def compact(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key in drop:
                    removed.add(key)
                    continue
                if key in ('candidate_evidence', 'candidate_removal_audit', 'relation_tasks', '_v4_relation_evidence') and isinstance(item, list):
                    result[key] = proofs(item)
                elif key in ('size_decision', 'size_comparison') and isinstance(item, dict):
                    result[key] = compact({k: v for k, v in item.items() if k != 'attributes'})
                    result[key]['condition_refs'] = [c.get('original_quote') for c in item.get('attributes', [])]
                elif key == 'scope_hypotheses' and isinstance(item, list):
                    result[key] = [compact({k: hypothesis[k] for k in ('interpretation', 'scope_kind',
                        'scope_status', 'scope_phrase', 'scope_relation_quotes', 'required_member_count',
                        'comparison_pool_ids', 'scope_unknown_ids', 'unknown_measurement_ids', 'selected_ids',
                        'tentative_selected_ids', 'status', 'reason', 'member_binding', 'candidate_evidence',
                        'metric', 'tentative_metric', 'original_quote', 'reference_ids') if k in hypothesis})
                        for hypothesis in item]
                elif key == 'geometry_evidence' and isinstance(item, dict):
                    result[key] = compact({k: v for k, v in item.items() if k != 'geometry'})
                    if 'geometry' in item:
                        result[key]['measured_geometry_fields'] = list(item['geometry'])
                elif key == 'anchor_hypotheses' and isinstance(item, list):
                    result[key] = [{k: compact(binding[k]) for k in ('status', 'anchor_ids', 'anchor_id', 'preferred_ids', 'plausible_band_m', 'reason') if k in binding}
                                   for binding in item]
                else:
                    result[key] = compact(item)
            if isinstance(result.get('details'), dict) and 'candidate_evidence' in result:
                result['details'].pop('candidate_evidence', None)
            return result
        if isinstance(value, list):
            return [compact(v) for v in value]
        return value
    # History and deletion provenance are summarized separately below. Do not
    # recursively copy their repeated snapshots merely to overwrite them.
    result = compact(text_state({k: v for k, v in state.items() if k not in
                                 ('history', 'deletion_provenance', 'deletion_events')}))
    parsed = result.get('parsed_query', {})
    tasks = result.get('observation', {}).get('details', {}).get('relation_tasks')
    if tasks is not None and parsed.get('_v4_relation_evidence') == tasks:
        parsed['_v4_relation_evidence'] = {'same_as_current_relation_tasks': True}
    result['history'] = [{'action': item.get('action'), 'remaining_candidate_ids': item.get('remaining_candidate_ids'),
        'missing': item.get('evidence_state', {}).get('missing', [])}
        for item in state.get('history', [])[-(1 if tighter else 3):]]
    if result.get('observation', {}).get('evidence_state') == result.get('evidence_state'):
        result['observation']['evidence_state'] = {'same_as_current_evidence_state': True}
    for entry in result['history']:
        if entry.get('missing') == result.get('evidence_state', {}).get('missing'):
            entry['missing'] = {'same_as_current_missing': True}
    if ('missing' in result.get('observation', {}) and
            result['observation']['missing'] == result.get('evidence_state', {}).get('missing')):
        result['observation']['missing'] = {'same_as_current_missing': True}
    observation = result.get('observation', {})
    if isinstance(observation.get('details'), dict):
        for key, value in list(observation['details'].items()):
            if key in observation and value == observation[key]:
                observation['details'][key] = {'same_as_current_observation': key}
    result['deletion_events'] = [{k: event.get(k) for k in
        ('event_id', 'candidate_id', 'dimension', 'source_quotes', 'invalidated')}
        for event in state.get('deletion_events', [])]
    for summary, event in zip(result['deletion_events'], state.get('deletion_events', [])):
        summary['proofs'] = [{k: proof.get(k) for k in ('candidate_id', 'status', 'source_quote', 'reason',
            'explicit_fail_criterion', 'role_binding_status', 'comparison_scope_status')}
            for proof in event.get('evidence', event.get('proofs', []))]
    payload = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
    # Byte length is an honest conservative bound when the configured model
    # has no supported local tokenizer. Include protocol overhead as well.
    audit = {'method': 'utf8_bytes_upper_bound', 'input_units': len(payload.encode()) + instruction_bytes + 64,
        'summary_bytes': len(payload.encode()), 'protocol_reserve': 64,
        'instruction_bytes': instruction_bytes,
        'context_limit': int(os.getenv('SEEGROUND_CONTEXT_LIMIT', '32768')),
        'removed': sorted(removed), 'history_entries': len(result['history']), 'tighter': tighter}
    # Each UTF-8 byte is a conservative upper bound for text BPE token count.
    # This is deliberately labelled as an upper bound, never exact tokens.
    if audit['input_units'] > audit['context_limit'] - 2048:
        result, shared_blocks = compact_repeated_evidence(result)
        payload = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
        audit.update(input_units=len(payload.encode()) + instruction_bytes + 64,
                     summary_bytes=len(payload.encode()),
                     evidence_encoding='lossless_tables_and_shared_blocks',
                     shared_evidence_blocks=shared_blocks)
    if audit['input_units'] > audit['context_limit'] - 2048:
        if not tighter:
            return planner_state(state, tighter=True, instruction_bytes=instruction_bytes)
        raise ValueError('required planner obligations exceed configured text context budget')
    return result, audit


def compress_planner_messages(messages):
    from copy import deepcopy
    result = deepcopy(messages)
    instruction_bytes = sum(len(message['content'].encode()) for message in result
                            if message.get('role') == 'system' and isinstance(message.get('content'), str))
    for message in result:
        if message.get('role') != 'user' or not isinstance(message.get('content'), list):
            continue
        for part in message['content']:
            if part.get('type') != 'text':
                continue
            try:
                state = json.loads(part['text'])
            except (ValueError, KeyError):
                continue
            if isinstance(state, dict) and 'evidence_state' in state:
                compact, _ = planner_state(state, tighter=True, instruction_bytes=instruction_bytes)
                part['text'] = json.dumps(compact, ensure_ascii=False, separators=(',', ':'))
    return result


def jev_state(state):
    """Send decision-relevant summaries instead of the full local audit trace."""
    parsed = state.get('parsed_query') or {}
    parsed = {key: parsed[key] for key in (
        'Target', 'Anchor', 'Relations', 'Viewpoint', 'Observer', 'Attributes', 'Unhandled',
        'condition_records', 'target_coreferences'
    ) if key in parsed}
    source_observation = state.get('observation') or {}
    observation = {key: source_observation[key] for key in
                   ('tool', 'dimension', 'status', 'candidate_ids', 'candidate_evidence',
                    'coverage', 'missing', 'next_evidence', 'policy')
                   if key in source_observation}
    geometry = source_observation.get('geometry')
    if isinstance(geometry, dict) and set(geometry) == {'shared_evidence_ref'}:
        geometry = state.get('shared_evidence', {}).get(geometry['shared_evidence_ref'], geometry)
    if isinstance(geometry, dict):
        geometry = {key: value for key, value in geometry.items()
                    if key != 'between_triplets'}
        observation['geometry'] = geometry
    details = source_observation.get('details')
    if isinstance(details, dict) and set(details) == {'shared_evidence_ref'}:
        details = state.get('shared_evidence', {}).get(details['shared_evidence_ref'], details)
    if isinstance(details, dict):
        compact_details = {key: details[key] for key in
                           ('dimension', 'candidate_ids', 'candidate_evidence', 'policy',
                           'candidate_metrics', 'attributes', 'visual_needs', 'size_decision',
                           'joint_size_summary', 'visual_inspection', 'relation_tasks', 'size_constraints')
                           if key in details}
        if 'distance_rankings' in details:
            rankings = details['distance_rankings']
            compact_details['distance_rankings'] = {
                'rankings': [{'relation': row.get('relation'), 'anchor_id': row.get('anchor_id'),
                              'best_ids': row.get('best_ids'), 'ordered_targets': row.get('ordered_targets')}
                             for row in rankings.get('rankings', [])],
                'ignored': rankings.get('ignored', []),
            }
        observation['details'] = compact_details
    history = []
    for item in state.get('history', [])[-8:]:
        row = {key: item[key] for key in ('action', 'remaining_candidate_ids', 'evidence_state')
               if key in item}
        prior_observation = item.get('observation')
        if isinstance(prior_observation, dict):
            row['observation'] = {key: prior_observation[key] for key in
                                  ('dimension', 'candidate_evidence', 'coverage', 'missing')
                                  if key in prior_observation}
        history.append(row)
    def labels(rows):
        return [{'bbox_id': row['bbox_id'], 'category': row.get('category', row.get('target', '')),
                 **category_fields(row)}
                for row in rows if isinstance(row, dict) and 'bbox_id' in row]
    return {
        'query': state.get('query', ''),
        'parsed_query': parsed,
        **({'category_policy': CATEGORY_INSTRUCTIONS}
           if any('category_hypotheses' in r for r in state.get('target_table', [])) else {}),
        'target_candidates': labels(state.get('target_table', [])),
        'reference_candidates': labels(state.get('reference_table', [])),
        'category_candidates': state.get('category_candidates', []),
        'remaining_candidate_ids': state.get('remaining_candidate_ids', []),
        'evidence_state': state.get('evidence_state', {}),
        'observation': observation,
        'history': history,
        'action_attempts_remaining': state.get('action_attempts_remaining'),
        'used_dimensions': state.get('used_dimensions', []),
        'computed_filters': state.get('computed_filters', []),
        'available_dimensions': state.get('available_dimensions', []),
        'evidence_capabilities': state.get('evidence_capabilities', {}),
        'tools_remaining': state.get('tools_remaining', 0),
        'evidence_inventory': state.get('evidence_inventory'),
        'backtrack_available': state.get('backtrack_available', False),
        'internal_visual_budget': state.get('internal_visual_budget'),
        **({'deletion_events': state.get('deletion_events', [])} if v5_enabled() else {}),
        **{key: state[key] for key in ('shared_evidence', 'evidence_encoding') if key in state},
    }


def evidence_tasks(state):
    tasks = {dimension: {'action': 'evidence', 'dimension': dimension,
                        'reason': state['evidence_capabilities'][dimension]}
             for dimension in state['available_dimensions'] if dimension != 'appearance'}
    remaining = state['remaining_candidate_ids']
    # JEV selects supplied tasks, so every live target needs an inspection option.
    groups = [remaining[start:start + 6] for start in range(0, len(remaining), 6)]
    for index, group in enumerate(groups, 1):
        for context in ('scene', 'isolated'):
            for azimuth in (90, 180):
                tasks[f'appearance_{index}_{context}_{azimuth}'] = {
                    'action': 'evidence', 'dimension': 'appearance', 'focus_ids': group,
                    'azimuth_deg': azimuth, 'elevation_deg': 25, 'context': context,
                    'reason': f'Inspect target appearance for {group} in {context} at {azimuth} degrees',
                }
    if state['tools_remaining'] > 0:
        # Import at execution time: the active controller also imports this module.
        from seeground.grounding.active_diagnostic import evidence_request
        known_ids = set(remaining) | set(state.get('category_candidates', []))
        for table in ('target_table', 'reference_table'):
            known_ids.update(row['bbox_id'] for row in state.get(table, [])
                             if isinstance(row, dict) and 'bbox_id' in row)
        objects = [{'bbox_id': oid} for oid in known_ids]
        signatures = {json.dumps({key: value for key, value in task.items() if key != 'reason'},
                                 sort_keys=True) for task in tasks.values()}
        for index, suggestion in enumerate((state.get('observation') or {}).get('next_evidence', []), 1):
            if (not isinstance(suggestion, dict)
                    or suggestion.get('dimension') not in state['available_dimensions']
                    or not isinstance(suggestion.get('reason'), str) or not suggestion['reason'].strip()):
                continue
            try:
                request = evidence_request(suggestion, remaining, objects)
            except ValueError:
                continue
            signature = json.dumps({key: value for key, value in request.items() if key != 'reason'},
                                   sort_keys=True)
            if signature in signatures:
                continue
            if len(tasks) >= 253:  # Reserve the finish and stop options in the JEV Choice.
                break
            tasks[f'suggested_evidence_{index}'] = request
            signatures.add(signature)
    if state['tools_remaining'] <= 0:
        tasks = {}
    tasks['finish'] = {'action': 'finish', 'reason': 'Request a final target choice from supplied evidence'}
    tasks['stop'] = {'action': 'stop', 'reason': 'Stop with insufficient evidence and no useful task remaining'}
    return tasks


def jev_action(state, instructions, call, calls):
    """Select a task, then select a live target only when finish is requested."""
    state = text_state(state)
    tasks = evidence_tasks(state)
    if not 1 <= len(tasks) <= 255:
        raise ValueError('JEV task Choice requires 1-255 options')
    questions = {'next_task': {'type': 'choice', 'instructions':
        'Select the next useful evidence task using the query, missing evidence, coverage, '
        'candidate differences and history. Do not repeat a completed task on unchanged evidence. '
        'Camera angles are inspection parameters, not the speaker reference frame. '
        'Finish before the evidence budget is exhausted only if status is resolved with one '
        'remaining candidate; otherwise collect useful evidence or stop. '
        'Confidence is not evidence or permission to exclude a candidate.', 'criteria': tasks}}
    request_state = jev_state(state)
    request_state['instructions'] = instructions
    call.update(backend='jev', role='planner', questions=questions,
                response_format={'type': 'choice'})
    try:
        result = decide('jev', request_state, questions)
    except DecisionAPIError as exc:
        call['error_response'] = exc.response_data
        if exc.diagnostic:
            call['api_diagnostic'] = dict(exc.diagnostic)
        raise
    call.update(model=result['model'], answers=result['answers'], usage=result.get('usage'),
                latency_s=result['latency_s'], validation_warnings=result['validation_warnings'],
                **{key: result[key] for key in ('api_attempts', 'extra_attempts', 'cache_hit') if key in result})
    action = dict(tasks[result['answers']['next_task']['choice']])
    call['reason_source'] = 'selected_task_description'
    target_data = None
    if action['action'] == 'finish':
        allowed = state['remaining_candidate_ids'] or state['category_candidates']
        if not 1 <= len(allowed) <= 255:
            raise ValueError('JEV target Choice requires 1-255 candidate options')
        questions = {'target': {'type': 'choice', 'instructions':
            'Choose the target best supported by the supplied evidence. Unknown conditions '
            'are not passes. Conditions must hold under the same compatible reference and '
            'observer interpretation. A budget-exhausted guess is not a resolved result.',
            'criteria': {str(oid): {'instance_id': oid} for oid in allowed}}}
        target_call = {'stage': 'active_target_decision', 'step': call['step'],
                       'backend': 'jev', 'role': 'planner', 'questions': questions,
                       'response_format': {'type': 'choice'}}
        calls.append(target_call)
        try:
            target = decide('jev', request_state, questions)
        except (ValueError, OSError) as exc:
            target_call['error'] = str(exc)
            if isinstance(exc, DecisionAPIError):
                target_call['error_response'] = exc.response_data
                if exc.diagnostic:
                    target_call['api_diagnostic'] = dict(exc.diagnostic)
            raise
        target_call.update(model=target['model'], answers=target['answers'], usage=target.get('usage'),
                           latency_s=target['latency_s'], validation_warnings=target['validation_warnings'],
                           **{key: target[key] for key in ('api_attempts', 'extra_attempts', 'cache_hit') if key in target})
        action['predicted_id'] = int(target['answers']['target']['choice'])
        target_call['response'] = json.dumps({'predicted_id': action['predicted_id']})
        target_call['finish_reason'] = 'stop'
        target_data = {'_content': json.dumps(action), '_finish_reason': 'stop',
                       'usage': target.get('usage'), '_baseline': {
                           'model': target['model'], 'latency_s': target['latency_s']}}
    return {'_content': json.dumps(action), '_finish_reason': 'stop', 'usage': result.get('usage'),
            '_baseline': {'model': result['model'], 'latency_s': result['latency_s'],
                **{key: result[key] for key in ('api_attempts', 'extra_attempts', 'cache_hit') if key in result}},
            '_target_decision': target_data}
