"""Model-directed, bounded evidence loop for candidate disambiguation."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from diagnostic import diagnose
from evidence import distance_rankings, match_retrieval_categories, retrieval_category
from object_geometry import build_geometry
from qwen_api import QwenAPIError, chat, image_part, parse_json_object
from run import render


DIMENSIONS = ('viewpoint', 'position', 'distance', 'appearance', 'size')

INSTRUCTIONS = '''You locate one target object from a natural-language description.
The category-filtered candidates are already supplied. Choose the next useful evidence
dimension from the description and the differences between the remaining candidates.
There is no fixed order: position may be enough, or size/shape, distance, viewpoint,
or appearance may be more useful. Use only the supplied computed evidence and rendered
views. A computed predicate is evidence, not a visual certificate; unknown is never a pass.
Keep candidates that are uncertain. Do not vote over alternative reference bindings.
Never turn small, large, skinny, or narrow into an unstated hard threshold. Never invent
a reference, viewpoint, relation, or target ID. The final ID must be one of the remaining
candidate IDs, unless the remaining set is empty and the initial category candidates are
explicitly provided as fallback.

Actions (return one JSON object with a short evidence-based reason):
{"action":"compute","dimension":"viewpoint|position|distance|appearance|size","reason":"..."}
{"action":"render","focus_ids":[1,2],"azimuth_deg":90,"elevation_deg":25,
 "context":"scene|isolated","reason":"what distinction this view can inspect"}
{"action":"visual_check","checks":[{"id":1,"verdict":"yes|no|uncertain","reason":"visible evidence"}],
 "reason":"..."}
{"action":"finish","predicted_id":1,"reason":"evidence supporting the choice"}
{"action":"stop","reason":"which evidence is still missing and why no useful action remains"}

Use at most 6 focus_ids in a render; prefer 2-4 remaining target candidates plus the
specific reference instances needed for the relation. For visual_check, use only IDs in the most recent rendered focus and use no only when
the rendered view visibly contradicts the target category or quoted appearance. A view
can show shape or appearance, but an isolated view cannot prove room position. Camera
azimuth is a world inspection direction, never the speaker's left/right frame.
Consult evidence_state for current missing evidence and visual needs. A computed
appearance/size summary is not a visual check. After rendering, inspect the image
with visual_check before requesting the same evidence again. Preserve earlier
constraints when switching dimensions. Used dimensions mean inspected, not solved.
If no useful action remains, use stop rather than inventing a target or repeating tools.
One candidate receiving yes does not exclude any alternatives. Compare remaining
alternatives in small rendered groups and record no only for visible contradictions.
Candidate-level pass means at least one reference interpretation supports that
predicate, not that all interpretations or the full description are confirmed.
While tool budget remains, finish is allowed only when evidence_state is resolved
with one candidate. Otherwise continue useful evidence collection or explicitly stop.
'''


def geometry_observation(diagnosis, category_candidates):
    geometry = diagnosis.get('geometry', {})
    return {
        'tool': 'initial_diagnosis',
        'status': diagnosis.get('status'),
        'category_candidates': category_candidates,
        'remaining_candidate_ids': diagnosis.get('candidate_ids', []),
        'stable_candidate_ids': diagnosis.get('stable_candidate_ids', []),
        'missing': diagnosis.get('missing', []),
        'geometry': {
            key: value for key, value in geometry.items()
            if key not in ('candidate_metrics', 'reference_metrics', 'size_policy')
        },
        'policy': 'baseline audit only; choose the next evidence dimension from the query',
    }


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


def position_candidate_evidence(diagnosis, candidate_ids=None):
    """Return per-candidate relation status, preserving alternative bindings."""
    geometry = diagnosis.get('geometry', {})
    ids = list(candidate_ids if candidate_ids is not None
               else diagnosis.get('candidate_ids', []))
    between_flags = {}
    relation_pairs = geometry.get('relation_pairs', [])
    for item in geometry.get('between_candidate_evidence', []):
        target_id = item.get('target_id')
        if target_id not in ids:
            continue
        flags = between_flags.setdefault(target_id, [])
        flags.extend([True] * len(item.get('passing_reference_pairs', [])))
        flags.extend([None] * len(item.get('unknown_reference_pairs', [])))
        flags.extend([False] * len(item.get('failing_reference_pairs', [])))
    records = []
    for target_id in ids:
        groups = {}
        for pair in relation_pairs:
            if pair.get('target_id') != target_id:
                continue
            relation = pair.get('relation', 'relation')
            groups.setdefault(relation, []).append(pair.get('passes'))
        if target_id in between_flags:
            groups.setdefault('between', []).extend(between_flags[target_id])
        normalized = {
            relation: {'status': _status(flags), 'observations': len(flags)}
            for relation, flags in groups.items()
        }
        records_for_id = {
            'candidate_id': target_id,
            'status': _overall_status(normalized),
            'relations': normalized,
        }
        records_for_id['exclusion_reason'] = (
            'all available reference interpretations fail the position predicate'
            if records_for_id['status'] == 'fail' else None)
        records.append(records_for_id)
    return records


def distance_candidate_evidence(rankings, candidate_ids):
    """Summarize explicit nearest/farthest rankings per target candidate."""
    result = []
    for candidate_id in candidate_ids:
        groups = {}
        for ranking in rankings:
            relation = ranking.get('relation', 'distance')
            ordered = {item.get('target_id') for item in ranking.get('ordered_targets', [])}
            if candidate_id not in ordered:
                continue
            groups.setdefault(relation, []).append(
                candidate_id in set(ranking.get('best_ids', [])))
        normalized = {relation: {'status': _status(flags), 'observations': len(flags)}
                      for relation, flags in groups.items()}
        result.append({'candidate_id': candidate_id,
                       'status': _overall_status(normalized),
                       'relations': normalized,
                       'exclusion_reason': (
                           'candidate is never best in any explicit distance ranking'
                           if _overall_status(normalized) == 'fail' else None)})
    return result


def apply_computed_evidence(diagnosis, evidence, dimension, record=True):
    """Apply only definite failures from a computed dimension to the live scope."""
    by_id = {item['candidate_id']: item for item in evidence}
    before = list(diagnosis.get('candidate_ids', []))
    removed = [oid for oid in before if by_id.get(oid, {}).get('status') == 'fail']
    remaining = [oid for oid in before if oid not in set(removed)]
    diagnosis['candidate_ids'] = remaining
    diagnosis['stable_candidate_ids'] = [oid for oid in diagnosis.get('stable_candidate_ids', [])
                                         if oid in remaining]
    if record:
        diagnosis.setdefault('active_computed_evidence', []).append({
            'dimension': dimension, 'evidence': evidence, 'removed_ids': removed,
        })
    if not remaining:
        diagnosis['status'] = 'conflict'
    elif diagnosis.get('missing'):
        diagnosis['status'] = 'missing'
    elif len(remaining) == 1:
        diagnosis['status'] = 'resolved'
    else:
        diagnosis['status'] = 'ambiguous'
    return {'before': before, 'removed_ids': removed, 'after': remaining}


def render_action(action, objects, xyz, rgb, ids, out, index):
    known = {obj['bbox_id']: obj for obj in objects}
    focus = action.get('focus_ids')
    if (not isinstance(focus, list) or not focus or len(focus) > 6
            or any(type(oid) is not int or oid not in known for oid in focus)
            or len(set(focus)) != len(focus)):
        raise ValueError('render needs 1-6 distinct known focus_ids; include a few candidates and references')
    azimuth, elevation = action.get('azimuth_deg'), action.get('elevation_deg')
    if (type(azimuth) not in (int, float) or not np.isfinite(azimuth)
            or not 0 <= azimuth < 360 or type(elevation) not in (int, float)
            or not np.isfinite(elevation) or not 10 <= elevation <= 75):
        raise ValueError('camera angles must be finite; azimuth [0,360), elevation [10,75]')
    context = action.get('context')
    if context not in ('scene', 'isolated'):
        raise ValueError('context must be scene or isolated')
    boxes = np.asarray([known[oid]['bbox_3d'] for oid in focus])
    lower = (boxes[:, :3] - boxes[:, 3:] / 2).min(0)
    upper = (boxes[:, :3] + boxes[:, 3:] / 2).max(0)
    at = (lower + upper) / 2
    radius = max(float(np.linalg.norm(upper - lower)), .5) * 1.5
    azimuth_rad, elevation_rad = np.deg2rad([azimuth, elevation])
    direction = np.array([
        np.cos(elevation_rad) * np.cos(azimuth_rad),
        np.cos(elevation_rad) * np.sin(azimuth_rad),
        np.sin(elevation_rad),
    ])
    eye = at + radius * direction
    ceiling_ids = [oid for oid, obj in known.items()
                   if obj['target'].casefold() == 'ceiling']
    keep = np.isin(ids, focus) if context == 'isolated' else ~np.isin(ids, ceiling_ids)
    if not keep.any():
        raise ValueError('selected focus has no instance points')
    path = out / f'active_view_{index:02d}.png'
    labels = np.where(np.isin(ids, focus), ids, -1)
    meta = render(xyz[keep], rgb[keep], path, eye=eye, at=at,
                  ids=labels[keep], size=1000)
    visible = set(meta['marked_ids'])
    return {
        'tool': 'render', 'image': str(path), 'focus_ids': focus,
        'context': context, 'azimuth_deg': action['azimuth_deg'],
        'elevation_deg': action['elevation_deg'],
        'unmarked_focus_ids': [oid for oid in focus if oid not in visible],
        'ceiling_ids_hidden': ceiling_ids if context == 'scene' else [],
        'provenance': 'CPU colored point-cloud inspection; not a real RGB crop',
        **meta,
    }


def compute_observation(dimension, query, parsed, diagnosis, candidates, geometry_objects):
    geometry = diagnosis.get('geometry', {})
    if dimension == 'viewpoint':
        return {
            'tool': 'compute', 'dimension': dimension,
            'observer': parsed.get('Observer', {}),
            'viewpoint': parsed.get('Viewpoint', ''),
            'hypotheses': [h.get('observer') for h in diagnosis.get('hypotheses', [])],
            'candidate_ids': diagnosis.get('candidate_ids', []),
            'policy': 'only explicit observer/frame evidence is used',
        }
    if dimension == 'position':
        candidate_evidence = position_candidate_evidence(diagnosis)
        return {
            'tool': 'compute', 'dimension': dimension,
            'relation_pairs': geometry.get('relation_pairs', []),
            'between_candidate_evidence': geometry.get('between_candidate_evidence', []),
            'between_triplets': geometry.get('between_triplets', []),
            'candidate_evidence': candidate_evidence,
            'candidate_ids': diagnosis.get('candidate_ids', []),
            'policy': 'candidate-level relation status; only definite fail is removable',
        }
    if dimension == 'distance':
        rankings = distance_rankings(query, parsed, candidates, geometry_objects)
        candidate_evidence = distance_candidate_evidence(
            rankings.get('rankings', []), diagnosis.get('candidate_ids', []))
        return {
            'tool': 'compute', 'dimension': dimension,
            'distance_rankings': rankings,
            'candidate_evidence': candidate_evidence,
            'candidate_ids': diagnosis.get('candidate_ids', []),
            'policy': 'explicit nearest/farthest rankings; only definite fail is removable',
        }
    if dimension == 'size':
        return {
            'tool': 'compute', 'dimension': dimension,
            'candidate_metrics': geometry.get('candidate_metrics', []),
            'reference_metrics': geometry.get('reference_metrics', []),
            'geometry_conditions': build_geometry(query, parsed, geometry_objects),
            'candidate_ids': diagnosis.get('candidate_ids', []),
            'policy': 'soft AABB size and shape evidence; adjectives are not hard filters',
        }
    if dimension == 'appearance':
        return {
            'tool': 'compute', 'dimension': dimension,
            'attributes': parsed.get('Attributes', []),
            'candidate_shape_evidence': geometry.get('candidate_metrics', []),
            'visual_needs': diagnosis.get('visual_needs', []),
            'candidate_ids': diagnosis.get('candidate_ids', []),
            'policy': 'AABB shape is a soft proxy; colors/materials still require a rendered view',
        }
    raise ValueError('dimension must be viewpoint, position, distance, appearance or size')


def _check_payload(action, focus_ids, candidates, parsed):
    checks = action.get('checks')
    if not isinstance(checks, list) or not checks:
        raise ValueError('visual_check needs a non-empty checks list')
    allowed = {obj['bbox_id'] for obj in candidates} & set(focus_ids)
    target_names = parsed.get('Target', [])
    target_category = retrieval_category(target_names[0]) if target_names else ''
    result = []
    seen = set()
    for check in checks:
        if (not isinstance(check, dict) or type(check.get('id')) is not int
                or check['id'] not in allowed or check['id'] in seen
                or check.get('verdict') not in ('yes', 'no', 'uncertain')
                or not isinstance(check.get('reason'), str)
                or not check['reason'].strip()):
            raise ValueError('visual_check IDs must be rendered candidates with yes/no/uncertain reasons')
        seen.add(check['id'])
        result.append({'id': check['id'], 'category': target_category,
                       'verdict': check['verdict'], 'reason': check['reason'],
                       'source': 'active rendered view'})
    return result


def evidence_state(diagnosis):
    """Expose current audit needs without calling inspected dimensions solved."""
    return {
        'status': diagnosis.get('status'),
        'remaining_candidate_ids': list(diagnosis.get('candidate_ids', [])),
        'missing': list(diagnosis.get('missing', [])),
        'visual_needs': deepcopy(diagnosis.get('visual_needs', [])),
    }


def active_ground(query, parsed, objects, geometry_objects, candidates, xyz, rgb, ids,
                  out, calls, trace, limit, trusted_categories):
    room_center = (xyz.min(0) + xyz.max(0)) / 2
    diagnosis = diagnose(parsed, geometry_objects, candidates, room_center,
                         trusted_categories=trusted_categories)
    initial_candidates = sorted(obj['bbox_id'] for obj in candidates)
    category_candidates = initial_candidates[:]
    observation = geometry_observation(diagnosis, category_candidates)
    history, images, used = [], [], set()
    checks, used_dimensions, computed_filters = [], set(), []
    last_render_focus = []
    controller_feedback = None
    trace.update(
        policy='category first; model selects the next evidence dimension; no fixed order',
        tool_budget=limit, category_candidates=category_candidates,
        initial_diagnosis=deepcopy(diagnosis), steps=history,
    )
    target_table = [{'bbox_id': obj['bbox_id'], 'category': obj['target']}
                    for obj in candidates]
    target_ids = set(category_candidates)
    references = [{'bbox_id': obj['bbox_id'], 'category': obj['target']}
                  for obj in geometry_objects if obj['bbox_id'] not in target_ids]
    attempts = 0
    tool_count = 0
    max_attempts = max(limit * 2 + 3, limit + 3)
    trace.update(action_attempt_budget=max_attempts, action_attempts=0,
                 successful_tools=0, rejected_actions=0)
    data = {}

    def finish_result(action, stop_reason, oid=None):
        remaining = diagnosis.get('candidate_ids', [])
        status = diagnosis.get('status') if oid is not None else 'missing'
        if not remaining:
            status = 'conflict'
        elif oid is not None and len(remaining) > 1:
            status = 'missing'
        history.append({'action': action, 'remaining_candidate_ids': list(remaining),
                        'evidence_state': evidence_state(diagnosis)})
        trace.update(stop_reason=stop_reason, final_candidate_ids=list(remaining),
                     final_checks=checks, final_evidence_state=evidence_state(diagnosis))
        obj = next((obj for obj in objects if obj['bbox_id'] == oid), None)
        return {
            'query': query, 'predicted_id': oid,
            'bbox_3d': obj['bbox_3d'] if obj else None,
            'response': data.get('_content', ''), 'usage': data.get('usage'),
            'model': data.get('_baseline', {}).get('model'),
            'latency_s': data.get('_baseline', {}).get('latency_s'),
            'finish_reason': data.get('_finish_reason'), 'images': images,
            'diagnosis': diagnosis, 'decision_status': status,
            'accepted': oid is not None and status == 'resolved',
            'decision_reasons': [action['reason']],
            'diagnosis_policy': 'active_model_selected_evidence', 'active_trace': trace,
        }

    while attempts < max_attempts:
        remaining = diagnosis.get('candidate_ids') or []
        state = {
            'query': query, 'parsed_query': parsed, 'target_table': target_table,
            'reference_table': references, 'annotated_categories': trusted_categories,
            'category_candidates': category_candidates,
            'remaining_candidate_ids': remaining,
            'history': history, 'observation': observation,
            'evidence_state': evidence_state(diagnosis),
            'action_attempts_remaining': max_attempts - attempts,
            'used_dimensions': sorted(used_dimensions),
            'computed_filters': [item['dimension'] for item in computed_filters],
            'available_dimensions': [d for d in DIMENSIONS if d not in used_dimensions],
            'last_render_focus_ids': last_render_focus,
            'tools_remaining': max(0, limit - tool_count),
        }
        instructions = INSTRUCTIONS
        if controller_feedback:
            instructions += '\nController correction: ' + controller_feedback
        if tool_count >= limit:
            instructions += '\nThe tool budget is exhausted. Return action=finish or action=stop now.'
        parts = []
        for path in images:
            parts.extend([{'type': 'text', 'text': f'Inspection {Path(path).name}'}, image_part(path)])
        parts.append({'type': 'text', 'text': json.dumps(state, ensure_ascii=False)})
        call = {'stage': 'active_decision', 'step': attempts + 1,
                'response_format': {'type': 'json_object'}}
        calls.append(call)
        trace['action_attempts'] += 1
        try:
            data = chat([{'role': 'system', 'content': instructions},
                         {'role': 'user', 'content': parts}], max_tokens=900, json_mode=True)
            call.update(response=data['_content'], usage=data.get('usage'),
                        latency_s=data['_baseline']['latency_s'],
                        finish_reason=data.get('_finish_reason'))
            if data.get('_finish_reason') != 'stop':
                raise QwenAPIError('active decision did not complete normally')
            action = parse_json_object(data['_content'])
            if not isinstance(action.get('reason'), str) or not action['reason'].strip():
                raise QwenAPIError('active decision requires a brief evidence-based reason')
        except (ValueError, OSError) as exc:
            call['error'] = str(exc)
            raise
        if action.get('action') == 'stop':
            return finish_result(action, 'model_stop')
        if action.get('action') == 'finish':
            oid = action.get('predicted_id')
            allowed = set(remaining) or set(category_candidates)
            valid_id = type(oid) is int and oid in allowed
            resolved = diagnosis.get('status') == 'resolved' and len(remaining) == 1
            if valid_id and (resolved or tool_count >= limit):
                return finish_result(action, 'budget_final_choice' if tool_count >= limit
                                     else 'model_finish', oid)
            error = ('finish requires an ID from the remaining candidates; use stop if unresolved'
                     if not valid_id else
                     'Premature finish: the evidence has not resolved a unique target. '
                     'A yes for one candidate does not exclude others. Compare unchecked alternatives '
                     'or inspect missing conditions; use stop if no useful evidence can be acquired.')
            observation = {'tool': 'finish', 'no_evidence_acquired': True,
                           'error': error}
            history.append({'action': action, 'observation': observation,
                            'remaining_candidate_ids': list(remaining)})
            trace['rejected_actions'] += 1
            controller_feedback = observation['error']
            attempts += 1
            continue
        if tool_count >= limit:
            history.append({'action': action, 'observation': {
                'tool': action.get('action'), 'error': 'tool budget exhausted',
                'no_evidence_acquired': True}, 'remaining_candidate_ids': list(remaining)})
            trace['rejected_actions'] += 1
            return finish_result(
                {'action': 'stop', 'reason': 'tool budget exhausted without a valid final decision'},
                'tool_budget_exhausted')
        signature = json.dumps({k: v for k, v in action.items() if k != 'reason'}, sort_keys=True)
        if signature in used:
            history.append({
                'action': action,
                'observation': {
                    'tool': action.get('action'),
                    'error': 'this tool action was already executed; choose another observation or finish',
                    'no_evidence_acquired': True,
                },
                'remaining_candidate_ids': diagnosis.get('candidate_ids', []),
            })
            controller_feedback = (
                'The previous action was a duplicate and acquired no evidence. '
                'Do not repeat it. Choose one dimension from available_dimensions, '
                'or render a smaller focus set, or finish if the evidence is sufficient.'
            )
            trace['rejected_actions'] += 1
            attempts += 1
            continue
        entry = {'action': action}
        before = list(diagnosis.get('candidate_ids', []))
        try:
            name = action.get('action')
            if name == 'compute':
                dimension = action.get('dimension')
                if dimension not in DIMENSIONS:
                    raise ValueError('compute needs a supported dimension')
                observation = compute_observation(dimension, query, parsed, diagnosis,
                                                  candidates, geometry_objects)
                used_dimensions.add(dimension)
                if dimension in ('position', 'distance'):
                    candidate_evidence = observation.get('candidate_evidence', [])
                    update = apply_computed_evidence(diagnosis, candidate_evidence, dimension)
                    computed_filters.append({'dimension': dimension,
                                             'evidence': candidate_evidence})
                    observation['candidate_update'] = update
                    observation['candidate_ids'] = diagnosis.get('candidate_ids', [])
            elif name == 'render':
                observation = render_action(action, geometry_objects, xyz, rgb, ids, out, tool_count + 1)
                images.append(observation['image'])
                last_render_focus = list(observation['focus_ids'])
            elif name == 'visual_check':
                if not last_render_focus:
                    raise ValueError('visual_check requires a preceding render action')
                new_checks = _check_payload(
                    action, last_render_focus,
                    [obj for obj in candidates if obj['bbox_id'] in before], parsed)
                checks.extend(new_checks)
                diagnosis = diagnose(parsed, geometry_objects, candidates, room_center, checks,
                                     trusted_categories=trusted_categories)
                for computed in computed_filters:
                    apply_computed_evidence(diagnosis, computed['evidence'], computed['dimension'],
                                            record=False)
                observation = {
                    'tool': 'visual_check', 'checks': new_checks,
                    'candidate_update': {
                        'before': before,
                        'removed_ids': [oid for oid in before if oid not in diagnosis['candidate_ids']],
                        'after': list(diagnosis['candidate_ids']),
                    },
                    'candidate_ids': diagnosis.get('candidate_ids', []),
                    'stable_candidate_ids': diagnosis.get('stable_candidate_ids', []),
                    'status': diagnosis.get('status'),
                }
            else:
                raise ValueError('unknown action; choose compute, render, visual_check or finish')
        except (ValueError, OSError) as exc:
            observation = {'tool': action.get('action'), 'error': str(exc),
                           'no_evidence_acquired': True}
            entry.update(observation=observation,
                         remaining_candidate_ids=diagnosis.get('candidate_ids', []))
            history.append(entry)
            trace['diagnosis_after_step'] = diagnosis
            controller_feedback = (
                'The previous tool request was invalid and acquired no evidence. '
                'Correct its parameters before retrying; do not repeat the invalid request.'
            )
            trace['rejected_actions'] += 1
            attempts += 1
            continue
        used.add(signature)
        tool_count += 1
        trace['successful_tools'] = tool_count
        controller_feedback = None
        entry.update(observation=observation,
                     remaining_candidate_ids=diagnosis.get('candidate_ids', []))
        entry['evidence_state'] = evidence_state(diagnosis)
        history.append(entry)
        trace['diagnosis_after_step'] = deepcopy(diagnosis)
        attempts += 1
    return finish_result(
        {'action': 'stop', 'reason': 'action-attempt budget exhausted without a valid final decision'},
        'action_attempt_budget_exhausted')
