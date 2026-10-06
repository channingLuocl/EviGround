"""Model-directed, bounded evidence loop for candidate disambiguation."""
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

For visual_check, use only IDs in the most recent rendered focus and use no only when
the rendered view visibly contradicts the target category or quoted appearance. A view
can show shape or appearance, but an isolated view cannot prove room position. Camera
azimuth is a world inspection direction, never the speaker's left/right frame.
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


def render_action(action, objects, xyz, rgb, ids, out, index):
    known = {obj['bbox_id']: obj for obj in objects}
    focus = action.get('focus_ids')
    if (not isinstance(focus, list) or not focus or len(focus) > 12
            or any(type(oid) is not int or oid not in known for oid in focus)
            or len(set(focus)) != len(focus)):
        raise ValueError('render needs 1-12 distinct known focus_ids')
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
        return {
            'tool': 'compute', 'dimension': dimension,
            'relation_pairs': geometry.get('relation_pairs', []),
            'between_candidate_evidence': geometry.get('between_candidate_evidence', []),
            'between_triplets': geometry.get('between_triplets', []),
            'candidate_ids': diagnosis.get('candidate_ids', []),
            'policy': 'AABB relation evidence; unknown pairs remain eligible',
        }
    if dimension == 'distance':
        return {
            'tool': 'compute', 'dimension': dimension,
            'distance_rankings': distance_rankings(query, parsed, candidates, geometry_objects),
            'candidate_ids': diagnosis.get('candidate_ids', []),
            'policy': 'only explicit nearest/farthest relations are ranked; no voting',
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
            'visual_needs': diagnosis.get('visual_needs', []),
            'candidate_ids': diagnosis.get('candidate_ids', []),
            'policy': 'appearance cannot be certified without a rendered or supplied view',
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
    for check in checks:
        if (not isinstance(check, dict) or type(check.get('id')) is not int
                or check['id'] not in allowed
                or check.get('verdict') not in ('yes', 'no', 'uncertain')
                or not isinstance(check.get('reason'), str)
                or not check['reason'].strip()):
            raise ValueError('visual_check IDs must be rendered candidates with yes/no/uncertain reasons')
        result.append({'id': check['id'], 'category': target_category,
                       'verdict': check['verdict'], 'reason': check['reason'],
                       'source': 'active rendered view'})
    return result


def active_ground(query, parsed, objects, geometry_objects, candidates, xyz, rgb, ids,
                  out, calls, trace, limit, trusted_categories):
    room_center = (xyz.min(0) + xyz.max(0)) / 2
    diagnosis = diagnose(parsed, geometry_objects, candidates, room_center,
                         trusted_categories=trusted_categories)
    initial_candidates = sorted(obj['bbox_id'] for obj in candidates)
    category_candidates = initial_candidates[:]
    observation = geometry_observation(diagnosis, category_candidates)
    history, images, used = [], [], set()
    checks, used_dimensions = [], set()
    last_render_focus = []
    trace.update(
        policy='category first; model selects the next evidence dimension; no fixed order',
        tool_budget=limit, category_candidates=category_candidates,
        initial_diagnosis=diagnosis, steps=history,
    )
    target_table = [{'bbox_id': obj['bbox_id'], 'category': obj['target']}
                    for obj in candidates]
    target_ids = set(category_candidates)
    references = [{'bbox_id': obj['bbox_id'], 'category': obj['target']}
                  for obj in geometry_objects if obj['bbox_id'] not in target_ids]
    for index in range(limit + 1):
        remaining = diagnosis.get('candidate_ids') or []
        state = {
            'query': query, 'parsed_query': parsed, 'target_table': target_table,
            'reference_table': references, 'annotated_categories': trusted_categories,
            'category_candidates': category_candidates,
            'remaining_candidate_ids': remaining,
            'history': history, 'observation': observation,
            'used_dimensions': sorted(used_dimensions),
            'available_dimensions': [d for d in DIMENSIONS if d not in used_dimensions],
            'last_render_focus_ids': last_render_focus,
            'tools_remaining': max(0, limit - index),
        }
        instructions = INSTRUCTIONS + ('\nThe tool budget is exhausted. Return action=finish now.'
                                       if index == limit else '')
        parts = []
        for path in images:
            parts.extend([{'type': 'text', 'text': f'Inspection {Path(path).name}'}, image_part(path)])
        parts.append({'type': 'text', 'text': json.dumps(state, ensure_ascii=False)})
        call = {'stage': 'active_decision', 'step': index + 1,
                'response_format': {'type': 'json_object'}}
        calls.append(call)
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
        if action.get('action') == 'finish':
            oid = action.get('predicted_id')
            allowed = set(remaining) or set(category_candidates)
            if type(oid) is not int or oid not in allowed:
                raise QwenAPIError('active decision selected an ID outside the remaining candidates')
            history.append({'action': action, 'remaining_candidate_ids': sorted(allowed)})
            status = diagnosis.get('status')
            if not remaining:
                status = 'conflict'
            obj = next(obj for obj in objects if obj['bbox_id'] == oid)
            trace['stop_reason'] = 'budget_final_choice' if index == limit else 'model_finish'
            trace['final_candidate_ids'] = sorted(remaining)
            trace['final_checks'] = checks
            return {
                'query': query, 'predicted_id': oid, 'bbox_3d': obj['bbox_3d'],
                'response': data['_content'], 'usage': data.get('usage'),
                'model': data['_baseline']['model'], 'latency_s': data['_baseline']['latency_s'],
                'finish_reason': data.get('_finish_reason'), 'images': images,
                'diagnosis': diagnosis, 'decision_status': status,
                'accepted': status == 'resolved',
                'decision_reasons': [action['reason']],
                'diagnosis_policy': 'active_model_selected_evidence',
                'active_trace': trace,
            }
        if index == limit:
            raise QwenAPIError('active decision ignored exhausted tool budget')
        signature = json.dumps({k: v for k, v in action.items() if k != 'reason'}, sort_keys=True)
        if signature in used:
            raise QwenAPIError('this tool action was already executed; choose another observation or finish')
        used.add(signature)
        entry = {'action': action}
        try:
            name = action.get('action')
            if name == 'compute':
                dimension = action.get('dimension')
                if dimension not in DIMENSIONS:
                    raise ValueError('compute needs a supported dimension')
                used_dimensions.add(dimension)
                observation = compute_observation(dimension, query, parsed, diagnosis,
                                                  candidates, geometry_objects)
            elif name == 'render':
                observation = render_action(action, geometry_objects, xyz, rgb, ids, out, index + 1)
                images.append(observation['image'])
                last_render_focus = list(observation['focus_ids'])
            elif name == 'visual_check':
                if not last_render_focus:
                    raise ValueError('visual_check requires a preceding render action')
                new_checks = _check_payload(action, last_render_focus, candidates, parsed)
                checks.extend(new_checks)
                diagnosis = diagnose(parsed, geometry_objects, candidates, room_center, checks,
                                     trusted_categories=trusted_categories)
                observation = {
                    'tool': 'visual_check', 'checks': new_checks,
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
    raise QwenAPIError('active decision ended without a target')
