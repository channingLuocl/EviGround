"""Typed JEV decisions for the text-only outer evidence loop."""
import json
import os

from decision_api import DecisionAPIError, decide, jev_model
from qwen_api import api_provider


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


def jev_state(state):
    """Send decision-relevant summaries instead of the full local audit trace."""
    parsed = state.get('parsed_query') or {}
    parsed = {key: parsed[key] for key in (
        'Target', 'Anchor', 'Relations', 'Viewpoint', 'Observer', 'Attributes', 'Unhandled'
    ) if key in parsed}
    source_observation = state.get('observation') or {}
    observation = {key: source_observation[key] for key in
                   ('tool', 'dimension', 'status', 'candidate_ids', 'candidate_evidence',
                    'coverage', 'missing', 'next_evidence', 'policy')
                   if key in source_observation}
    geometry = source_observation.get('geometry')
    if isinstance(geometry, dict):
        geometry = {key: value for key, value in geometry.items()
                    if key != 'between_triplets'}
        observation['geometry'] = geometry
    details = source_observation.get('details')
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
        return [{'bbox_id': row['bbox_id'], 'category': row.get('category', row.get('target', ''))}
                for row in rows if isinstance(row, dict) and 'bbox_id' in row]
    return {
        'query': state.get('query', ''),
        'parsed_query': parsed,
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
        from active_diagnostic import evidence_request
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
                latency_s=result['latency_s'], validation_warnings=result['validation_warnings'])
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
                           latency_s=target['latency_s'], validation_warnings=target['validation_warnings'])
        action['predicted_id'] = int(target['answers']['target']['choice'])
        target_call['response'] = json.dumps({'predicted_id': action['predicted_id']})
        target_call['finish_reason'] = 'stop'
        target_data = {'_content': json.dumps(action), '_finish_reason': 'stop',
                       'usage': target.get('usage'), '_baseline': {
                           'model': target['model'], 'latency_s': target['latency_s']}}
    return {'_content': json.dumps(action), '_finish_reason': 'stop', 'usage': result.get('usage'),
            '_baseline': {'model': result['model'], 'latency_s': result['latency_s']},
            '_target_decision': target_data}
