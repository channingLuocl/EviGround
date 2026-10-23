"""A bounded visual ReAct trajectory, using predicted categories and measured geometry."""
import json
from pathlib import Path

import numpy as np

from seeground.categories.categories_v9 import CATEGORY_INSTRUCTIONS
from seeground.grounding.evidence import distance_pair_metrics
from seeground.api.qwen_api import chat, image_part, parse_json_object
from seeground.grounding.r8_grounding import matching_hypotheses
from seeground.grounding.r8_precision import precision_evidence
from seeground.api.response_schemas import obj, array, enum, nullable, TEXT, REASON, schema_kwargs
from seeground.grounding.v4_policy import settings


INSTRUCTIONS = '''Locate the object described by the original utterance. You control evidence acquisition.
Read the scene, compare the plausible target instances, and request the next observation that can
distinguish them. Then reconsider the alternatives using the returned evidence. There is no fixed
sequence of tools. Every request and final choice must be grounded in the original description.

Actions:
- inspect: render 1-12 focus_ids in panels of at most 6, including targets AND references. Use isolated to compare
  category, shape, texture or color without contamination. Use scene to see their physical context.
  Choose azimuth_deg in [0,360) and elevation_deg in [10,75] to expose the unresolved property.
  Isolated tiles have separate cameras and scales; their placement/size is not a spatial relation.
- measure: supply focus_ids for the objects to compare. Returns measured dimensions, pair distances,
  overlap and conditional observer frames. Bind the described references and comparison group first.
  Measurements are observations, not proof that a reference/category interpretation is correct.
- finish: choose one allowed_target_id, or null if essential evidence is unavailable. Give compact
  evidence, name the actual reference IDs, and state unresolved assumptions. Return one final choice.

Use inspect when category alternatives, a reference's color, an object's visible part or facing
direction are unresolved. Use measure when distance, support, height, order or grouping separates
the plausible objects. Inspect references as needed; a convenient reference ID is not a bound noun.
Prefer a discriminating observation to repeating an unchanged view or merely confirming a guess.
All clauses must refer to a compatible assignment of target and reference instances. Keep negation,
the exact comparison group, and noun attachment. The original utterance outranks the parser summary.
Classifier ranks are fallible priors. A rank-2 category can win with visible support; rank-1 gets
preference only if the remaining evidence is comparable. Never use image left as speaker left.
When no observer is given, explicitly state any assumed frame; do not reverse numeric order to
support a preferred ID. AABB dimensions alone do not identify the head or foot of a bed.
An unobserved attribute stays unknown. Return the best supported target despite minor uncertainty,
but do not invent appearance, references or evidence. Unknown does not eliminate an alternative.

Keep reason within 60 words, evidence within 4 entries, and each observation within 35 words.
Every evidence source_quote must be the COMPLETE original description verbatim. For inspect or
measure, use predicted_id=null and evidence=[]. For finish use focus_ids=[], context=null,
azimuth_deg=null and elevation_deg=null. Return only the requested JSON object.
'''


def measure(focus_ids, objects, parsed, candidates, query):
    selected = [o for o in objects if o['bbox_id'] in focus_ids]
    rows, pairs = [], []
    for item in selected:
        x, y, z, dx, dy, dz = map(float, item['bbox_3d'])
        rows.append({'id': item['bbox_id'], 'center_xyz_m': [x, y, z],
                     'extent_xyz_m': [dx, dy, dz], 'observed_box_volume_m3': dx*dy*dz,
                     'observed_footprint_m2': dx*dy, 'bottom_z_m': z-dz/2, 'top_z_m': z+dz/2})
        for reference in selected:
            if item['bbox_id'] == reference['bbox_id']:
                continue
            box, ref = np.asarray(item['bbox_3d'], float), np.asarray(reference['bbox_3d'], float)
            overlap = np.maximum(0, np.minimum(box[:2]+box[3:5]/2, ref[:2]+ref[3:5]/2)
                                 - np.maximum(box[:2]-box[3:5]/2, ref[:2]-ref[3:5]/2))
            distances = distance_pair_metrics(reference, item)
            pairs.append({'object_id': item['bbox_id'], 'reference_id': reference['bbox_id'],
                'delta_xyz_m': [round(float(v), 4) for v in box[:3]-ref[:3]],
                'xy_center_m': round(distances['horizontal_center'], 4),
                'xy_box_gap_m': round(distances['horizontal_gap'], 4),
                'center_3d_m': round(distances['center'], 4),
                'box_gap_3d_m': round(distances['aabb_gap'], 4),
                'bottom_minus_reference_top_m': round(float(box[2]-box[5]/2-ref[2]-ref[5]/2), 4),
                'xy_overlap_fraction': round(float(np.prod(overlap)/np.prod(box[3:5])), 4)})
    frames = precision_evidence(query, parsed, objects, candidates)['observer_frames']
    return {'tool': 'measure', 'objects': rows, 'directed_pair_measurements': pairs,
            'observer_frames': frames, 'policy': 'Observed geometry; comparison pools and frames require language/visual binding.'}


def ground(query, parsed, objects, candidates, xyz, rgb, ids, out, calls, diagnosis, trace, limit):
    from seeground.grounding.active_diagnostic import render_action
    from seeground.grounding.v74_grounding import scene_views, validate_decision
    known = {o['bbox_id'] for o in objects}
    allowed = sorted(o['bbox_id'] for o in candidates)
    schema = obj({'action': enum('inspect', 'measure', 'finish'), 'reason': REASON,
        'focus_ids': array({'type': 'integer', 'enum': sorted(known)}),
        'context': nullable(enum('scene', 'isolated')),
        'azimuth_deg': nullable({'type': 'number'}), 'elevation_deg': nullable({'type': 'number'}),
        'predicted_id': {'type': ['integer', 'null'], 'enum': allowed+[None]},
        'evidence': array(obj({'source_quote': {'type': 'string', 'enum': [query]},
            'object_ids': array({'type': 'integer', 'enum': sorted(known)}), 'observation': REASON})),
        'unresolved': array(TEXT)})
    overview = scene_views(parsed, objects, candidates, xyz, rgb, ids, out)
    views = [overview[0], overview[-1]]
    basic = precision_evidence(query, parsed, objects, candidates)['objects']
    # Keep all original category alternatives available when the parser has missed a reference.
    by_id = {o['bbox_id']: o for o in objects}
    for row in basic:
        row['category_hypotheses'] = [{'category': h['category'], 'rank': h['rank']}
                                     for h in by_id[row['id']].get('category_hypotheses', [])]
    payload = {'description': query, 'allowed_target_ids': allowed, 'objects': basic,
               'parsed_target_categories': parsed.get('Target', []),
               'parsed_reference_categories': parsed.get('Anchor', []),
               'category_policy': 'Target pool and class ranks come only from the classifier; parser bindings are hypotheses.',
               'view_policy': 'Initial overview and top-down view are inspection cameras, not a speaker frame.'}
    payload['target_category_bindings'] = [
        {'id': o['bbox_id'], 'matched_ranks': [h['rank'] for h in matching_hypotheses(parsed.get('Target', []), o)]}
        for o in candidates]
    payload['reference_category_bindings'] = [
        {'category': name, 'candidates': [
            {'id': o['bbox_id'], 'matched_ranks': [h['rank'] for h in matching_hypotheses([name], o)]}
            for o in objects if matching_hypotheses([name], o)]}
        for name in parsed.get('Anchor', [])]
    observations, actions, signatures = [], [], set()
    trace.update(policy='model-selected inspect/measure observations followed by one final decision',
                 interface='multimodal_react', planner_input='text_and_scene_images',
                 r13_initial_input=payload, r13_actions=actions, r13_observations=observations,
                 r13_tool_budget=limit)
    decision = None
    for step in range(limit+1):
        state = {**payload, 'observations': observations, 'previous_actions': actions,
                 'remaining_tool_calls': limit-step,
                 'views': [{k: v for k, v in view.items() if k in
                            ('kind', 'marked_ids', 'focus_ids', 'context', 'azimuth_deg', 'elevation_deg', 'layout')}
                           for view in views]}
        action_schema = json.loads(json.dumps(schema))
        if step == limit:
            action_schema['properties']['action'] = enum('finish')
        call = {'stage': 'r13_react_decision', 'role': 'multimodal_evidence_planner',
                'step': step, 'images': [v['image'] for v in views]}
        calls.append(call)
        response = chat([{'role': 'system', 'content': INSTRUCTIONS+CATEGORY_INSTRUCTIONS},
            {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps(state, ensure_ascii=False)},
                                      *[image_part(v['image']) for v in views]]}],
            max_tokens=1600, json_mode=True, **schema_kwargs('r13_react_action', action_schema))
        call.update(response=response['_content'], usage=response.get('usage'), **response.get('_baseline', {}))
        if response.get('_finish_reason') != 'stop':
            raise ValueError('ReAct action response is incomplete')
        action = parse_json_object(response['_content'])
        actions.append(action)
        if action.get('action') == 'finish':
            decision = validate_decision(action, query, allowed, known)
            break
        if step == limit:
            raise ValueError('ReAct must finish when the evidence budget is exhausted')
        focus = action.get('focus_ids')
        try:
            if (not isinstance(focus, list) or not focus or len(set(focus)) != len(focus)
                    or any(type(i) is not int or i not in known for i in focus)):
                raise ValueError('Supply distinct known focus_ids for evidence acquisition')
            signature = json.dumps({k: action.get(k) for k in
                                    ('action', 'focus_ids', 'context', 'azimuth_deg', 'elevation_deg')}, sort_keys=True)
            if signature in signatures:
                raise ValueError('This observation was already acquired; select different evidence or finish')
            signatures.add(signature)
            if action['action'] == 'inspect':
                if len(focus) > 12:
                    raise ValueError('Inspect at most 12 objects; select the relevant comparison group')
                rendered = [render_action({**action, 'focus_ids': focus[offset:offset+6]},
                    objects, xyz, rgb, ids, Path(out), 2000+2*step+offset//6)
                    for offset in range(0, len(focus), 6)]
                views.extend(rendered)
                observation = {'tool': 'render', 'focus_ids': focus,
                    'views': [{k: v for k, v in view.items() if k not in ('image', 'panels')}
                              for view in rendered]}
                trace['internal_usage']['render'] += sum(v.get('views_rendered', 1) for v in rendered)
            elif action['action'] == 'measure':
                observation = measure(focus, objects, parsed, candidates, query)
                trace['internal_usage']['compute'] += 1
            else:
                raise ValueError('Unknown evidence action')
        except ValueError as exc:
            observation = {'tool': action.get('action'), 'error': str(exc), 'no_evidence_acquired': True}
        observations.append(observation)
        trace['steps'].append({'action': action, 'observation': observation})
    oid = decision['predicted_id']
    selected = next((o for o in candidates if o['bbox_id'] == oid), None)
    trace.update(stop_reason='r13_react_finish', final_candidate_ids=allowed, r13_final_decision=decision,
        final_evidence_state={'status': 'tentative' if oid is not None else 'missing',
                              'missing': decision.get('unresolved', [])})
    trace['steps'].append({'action': 'finish', 'decision': decision})
    trace['internal_usage']['render'] += len(overview)
    trace['internal_usage']['visual_check'] += len(actions)
    return {'query': query, 'predicted_id': oid, 'bbox_3d': selected['bbox_3d'] if selected else None,
        'implementation_version': settings()['version'],
        'prediction_source': 'r13_react_tentative' if oid is not None else 'abstained',
        'diagnosis': diagnosis, 'decision_status': 'tentative' if oid is not None else 'missing',
        'accepted': False, 'images': [v['image'] for v in views], 'response': call.get('response', ''),
        'decision_reasons': [decision['reason']], 'scene_evidence': decision, 'active_trace': trace,
        'final_review': {'status': 'included_in_react_trajectory', 'reason': 'Final choice follows model-selected evidence observations'},
        'backtrack_count': 0, 'internal_budget_usage': dict(parsed['_v4_budget']),
        'diagnosis_policy': 'predicted_category_multimodal_react'}
