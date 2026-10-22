"""A bounded visual ReAct trajectory, using predicted categories and measured geometry."""
import json
from pathlib import Path
import re

import numpy as np

from categories_v9 import CATEGORY_INSTRUCTIONS
from evidence import distance_pair_metrics
from qwen_api import chat, image_part, parse_json_object
from r8_grounding import matching_hypotheses
from r8_precision import frame_coordinates, precision_evidence
from response_schemas import obj, array, enum, nullable, TEXT, REASON, schema_kwargs
from v4_policy import settings


INSTRUCTIONS = '''Locate the object described by the original utterance. You control evidence acquisition.
Read the scene, compare the plausible target instances, and request the next observation that can
distinguish them. Then reconsider the alternatives using the returned evidence. There is no fixed
sequence of tools. Every request and final choice must be grounded in the original description.

This application executes the JSON actions you return. inspect and measure are available through
this JSON interface, even though there are no native API function-call tools. Return the requested
action object; the application will execute it and supply the observation on your next turn.
The state's available_actions and remaining_tool_calls describe what you can request now.

Actions:
- inspect: render 1-12 focus_ids in panels of at most 6, including targets AND references. Use isolated to compare
  category, shape, texture or color without contamination. Use scene to see their physical context.
  Choose azimuth_deg in [0,360) and elevation_deg in [10,75] to expose the unresolved property.
  Null camera angles use the tool's documented defaults: azimuth=315, elevation=30 degrees.
  Isolated tiles have separate cameras and scales; their placement/size is not a spatial relation.
- measure: supply focus_ids for the objects to compare. Returns measured dimensions, pair distances,
  overlap and sorted size orders. For observer-relative order supply frame: origin_ids=[] means room
  center, otherwise the mean center of those instances; facing_ids defines the group being faced.
  For 'from A facing B', origin_ids is A and facing_ids is B. For 'room center facing B', use
  origin_ids=[] and facing_ids=B. An object being faced is NOT the observer origin. For 'looking
  at both targets', use the visually bound target group as facing_ids. Never use identical origin
  and facing groups. Without a directional clause use frame=null; distances need no observer frame.
  Both groups must follow the description and visible layout. Use frame=null for other measurements.
  Always include a nonempty focus_ids list in a measure action, even when a frame is supplied.
  The returned selected_observer_frame is conditional on YOUR chosen groups. Bind the described
  references and comparison group first. Include all plausible comparison members, not just a guess.
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
gap_from_top1 is the difference between uncalibrated classifier scores, NOT a probability. Within
one instance, a small gap denotes a closer classifier alternative than a large gap. It does not
establish the category, and should not be compared using a probability cutoff.
Top-5 recall expands the pool; it does not make all five names true. A rank-1 label is still only a
prior: do not discard a rank-2--5 candidate merely because its point-cloud appearance is small,
folded, occluded or visually uncertain. Before choosing an alternative-only target category, inspect
that instance and plausible primary-category rivals. Geometry alone cannot
turn a monitor stand into a monitor, or a toilet into a stool. A finish without that inspection is
deferred while a tool call remains. Check reference categories and their qualifiers independently;
do not select whichever door, desk or shelf happens to make a convenient distance minimum.
Some allowed IDs may be marked candidate_role=visual_fallback. Their classifier category did not
match the parsed target; they were added only from a bounded geometric neighborhood because a
Top-5 miss is possible. Treat them as unknown-category visual hypotheses: inspect them before
choosing them, and compare them with the matched target candidates.
For a nested relation such as an object ON a desk, first bind the object to its actual support desk
using overlap and vertical surface gap; only then compare targets relative to that SAME desk.
Preserve every clause: color, visible part, group, negation and relative order cannot be replaced by
nearest, largest or highest. If one clause distinguishes the current best two candidates, acquire
evidence for that clause before finishing. State what distinguishes the winner from its best rival.
When no observer is given, explicitly state any assumed frame; do not reverse numeric order to
support a preferred ID. AABB dimensions alone do not identify the head or foot of a bed.
An unobserved attribute stays unknown. Return the best supported target despite minor uncertainty,
but do not invent appearance, references or evidence. Unknown does not eliminate an alternative.

Before finish, perform a complete comparison for every spatial, distance, order, support or size
clause: use measure with all allowed target candidates and the bound reference instances. A scene
view or an object table is not a substitute for that comparison. If the chosen target is a rank-2+
category hypothesis, inspect it together with every rank-1 target-category rival before deciding.
Never finish immediately from the initial overview when more than one target candidate remains.
In the final evidence, cite the measured/observed IDs that distinguish the selected target from its
strongest rival; do not turn an assumption into a fact merely to satisfy the schema.

Keep reason within 60 words, evidence within 4 entries, and each observation within 35 words.
Every evidence source_quote must be the COMPLETE original description verbatim. For inspect or
measure, use predicted_id=null and evidence=[]. For inspect and finish use frame=null. For finish use
focus_ids=[], context=null, azimuth_deg=null and elevation_deg=null. Return a single top-level JSON
OBJECT with the schema keys, never an array containing an object.
'''


_MEASURE_RELATIONS = {
    'nearest', 'farthest', 'near', 'left', 'right', 'front', 'back', 'above',
    'below', 'underneath', 'between', 'overlap', 'over', 'support', 'inside',
}
_MEASURE_WORDS = re.compile(
    r'\b(?:closest|nearest|farthest|farther|left|right|front|behind|back|above|below|'
    r'under|underneath|over|between|next\s+to|near|beside|alongside|away\s+from|'
    r'larger|smaller|taller|shorter|lowest|highest|middle|first|second|third|end|corner)\b',
    re.IGNORECASE)


def requires_measure(query, parsed):
    """Return whether the description contains a clause whose comparison needs geometry."""
    relations = {str(item.get('relation', '')).casefold()
                 for item in parsed.get('Relations', []) if isinstance(item, dict)}
    return bool(relations & _MEASURE_RELATIONS) or bool(_MEASURE_WORDS.search(query))


def required_measure_ids(parsed, allowed, objects):
    """Bind all target candidates and the visible instances matching parsed anchors."""
    ids = set(allowed)
    for name in parsed.get('Anchor', []):
        ids.update(item['bbox_id'] for item in objects if matching_hypotheses([name], item))
    return sorted(ids)


def reference_review_ids(parsed, objects):
    """Return classifier-covered reference instances that can change a relation binding."""
    names = list(parsed.get('Anchor', []))
    for relation in parsed.get('Relations', []):
        if not isinstance(relation, dict):
            continue
        names.extend(relation.get('anchor_categories', []) or [])
        if relation.get('anchor_category'):
            names.append(relation['anchor_category'])
    names = list(dict.fromkeys(name for name in names if name))
    ids = []
    for name in names:
        matched = [item for item in objects if matching_hypotheses([name], item)]
        matched.sort(key=lambda item: (min((h['rank'] for h in matching_hypotheses([name], item)),
                                           default=99), item['bbox_id']))
        ids.extend(item['bbox_id'] for item in matched)
    return list(dict.fromkeys(ids))[:12]


def measured_together(required_ids, observations):
    return any(item.get('tool') == 'measure' and set(required_ids).issubset(
        row['id'] for row in item.get('objects', [])) for item in observations)


def compact_observations(observations):
    """Encode geometry as columnar rows; retain the original evidence in the trace."""
    compact, covered = [], set()
    for item in reversed(observations):
        if item.get('tool') != 'measure':
            compact.append(item)
            continue
        ids = frozenset(row['id'] for row in item['objects'])
        frame = json.dumps(item.get('selected_observer_frame'), sort_keys=True)
        if any(ids <= previous_ids and frame == previous_frame
               for previous_ids, previous_frame in covered):
            continue
        covered.add((ids, frame))
        pairs = item['directed_pair_measurements']
        columns = list(pairs[0]) if pairs else []
        compact.append({**item, 'directed_pair_measurements': {
            'columns': columns, 'rows': [[row[key] for key in columns] for row in pairs]}})
    return list(reversed(compact))


def current_views(views):
    """Keep the overview and latest three images; old views can be inspected again."""
    recent = views[-3:]
    return recent if views[0] in recent else [views[0], *recent]


def latest_frame_hint(actions):
    """Reuse a model-selected observer frame when a later guard completes its measurement."""
    for action in reversed(actions):
        frame = action.get('frame') if isinstance(action, dict) else None
        if isinstance(frame, dict):
            return frame
    return None


def category_review_ids(oid, best_ranks):
    """Return the selected alternative and all primary-category target rivals."""
    if best_ranks.get(oid) not in (2, 3, 4, 5):
        return []
    return sorted([oid] + [item for item, rank in best_ranks.items() if rank == 1])


def observer_frame(frame, objects, selected):
    groups = (frame.get('origin_ids'), frame.get('facing_ids'))
    known = {o['bbox_id'] for o in objects}
    if any(not isinstance(group, list) or len(set(group)) != len(group)
           or any(type(i) is not int or i not in known for i in group) for group in groups):
        raise ValueError('Observer frame needs distinct known origin_ids and facing_ids')
    origin_ids, facing_ids = groups
    if not facing_ids:
        raise ValueError('Observer frame needs a nonempty facing group')
    boxes = np.array([o['bbox_3d'] for o in objects], float)
    room_center = ((boxes[:, :3]-boxes[:, 3:]/2).min(0)
                   +(boxes[:, :3]+boxes[:, 3:]/2).max(0))/2
    origin = (np.mean([o['bbox_3d'][:3] for o in objects if o['bbox_id'] in origin_ids], axis=0)
              if origin_ids else room_center)
    facing = np.mean([o['bbox_3d'][:3] for o in objects if o['bbox_id'] in facing_ids], axis=0)
    result = frame_coordinates(origin, facing, selected)
    if result is None:
        raise ValueError('Observer origin and facing group coincide; choose a supported frame')
    result.update(origin_ids=origin_ids, facing_ids=facing_ids,
        assumption='model-selected origin and facing groups; verify against the original description')
    return result


def measure(focus_ids, objects, parsed, candidates, query, frame=None):
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
    selected_frame, frame_error = None, None
    if frame is not None:
        try:
            selected_frame = observer_frame(frame, objects, selected)
        except ValueError as exc:
            # An invalid viewing frame does not invalidate independent distances.
            frame_error = str(exc)
    orders = {key+'_ascending': [[row['id'], row[key]] for row in sorted(rows, key=lambda r: (r[key], r['id']))]
              for key in ('bottom_z_m', 'top_z_m', 'observed_box_volume_m3', 'observed_footprint_m2')}
    return {'tool': 'measure', 'objects': rows, 'directed_pair_measurements': pairs,
            'sorted_object_orders': orders, 'selected_observer_frame': selected_frame,
            'observer_frame_error': frame_error,
            'conditional_observer_frames': precision_evidence(query, parsed, objects, candidates)['observer_frames'],
            'policy': 'Observed geometry; comparison pools and frames require language/visual binding.'}


def ground(query, parsed, objects, candidates, xyz, rgb, ids, out, calls, diagnosis, trace, limit):
    from active_diagnostic import render_action
    from v74_grounding import scene_views, validate_decision
    known = {o['bbox_id'] for o in objects}
    allowed = sorted(o['bbox_id'] for o in candidates)
    schema = obj({'action': enum('inspect', 'measure', 'finish'), 'reason': REASON,
        'focus_ids': array({'type': 'integer', 'enum': sorted(known)}),
        'context': nullable(enum('scene', 'isolated')),
        'azimuth_deg': nullable({'type': 'number'}), 'elevation_deg': nullable({'type': 'number'}),
        'frame': nullable(obj({'origin_ids': array({'type': 'integer', 'enum': sorted(known)}),
                               'facing_ids': array({'type': 'integer', 'enum': sorted(known)})})),
        'predicted_id': {'type': ['integer', 'null'], 'enum': allowed+[None]},
        'evidence': array(obj({'source_quote': {'type': 'string', 'enum': [query]},
            'object_ids': array({'type': 'integer', 'enum': sorted(known)}), 'observation': REASON})),
        'unresolved': array(TEXT)})
    overview = scene_views(parsed, objects, candidates, xyz, rgb, ids, out)
    views = [overview[0], overview[-1]]
    basic = precision_evidence(query, parsed, objects, candidates)['objects']
    # Keep all original category alternatives available when the parser has missed a reference.
    by_id = {o['bbox_id']: o for o in objects}
    fallback_ids = set(parsed.get('candidate_generation', {}).get('visual_fallback_ids', []))
    for row in basic:
        row['category_hypotheses'] = [{'category': h['category'], 'rank': h['rank'],
                                       'gap_from_top1': round(h['gap_from_top1'], 4)}
                                     for h in by_id[row['id']].get('category_hypotheses', [])]
        if row['id'] in fallback_ids:
            row['candidate_role'] = 'visual_fallback'
    payload = {'description': query, 'allowed_target_ids': allowed, 'objects': basic,
               'visual_fallback_target_ids': sorted(fallback_ids),
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
    observations, actions, signatures, inspected = [], [], set(), set()
    best_ranks = {b['id']: min(b['matched_ranks'], default=None)
                  for b in payload['target_category_bindings']}
    trace.update(policy='model-selected inspect/measure observations followed by one final decision',
                 interface='multimodal_react', planner_input='text_and_scene_images',
                 r18_initial_input=payload, r18_actions=actions, r18_observations=observations,
                 r18_tool_budget=limit)
    decision = None
    guard_attempts = 0
    final_review_done = False
    reserved_evidence = False
    tool_calls = 0
    guard_prompted = False
    budget_reminder_sent = False
    for step in range(limit+12):
        attached = current_views(views)
        remaining = max(0, limit-tool_calls) + int(reserved_evidence)
        final_only = remaining == 0
        state = {**payload, 'observations': compact_observations(observations),
                 'previous_actions': actions, 'remaining_tool_calls': remaining,
                 'available_actions': ['finish'] if final_only else ['inspect', 'measure', 'finish'],
                 'decision_phase': 'final_choice' if final_only else 'evidence_acquisition',
                 'instruction': ('Tool budget exhausted: return action=finish using the supplied evidence. '
                                 'Disclose unresolved assumptions; do not request inspect/measure.'
                                 if final_only else 'Request inspect or measure by returning its JSON action; '
                                 'the application will execute it. Select useful evidence or finish.'),
                 'views': [{k: v for k, v in view.items() if k in
                            ('kind', 'marked_ids', 'focus_ids', 'context', 'azimuth_deg', 'elevation_deg', 'layout')}
                           for view in attached]}
        action_schema = json.loads(json.dumps(schema))
        if final_only:
            action_schema['properties']['action'] = enum('finish')
        call = {'stage': 'r18_react_decision', 'role': 'multimodal_evidence_planner',
                'step': step, 'images': [v['image'] for v in attached],
                'model_tool_calls': tool_calls, 'controller_tool_calls': guard_attempts,
                'final_only': final_only, 'state_chars': len(json.dumps(state, ensure_ascii=False))}
        calls.append(call)
        final_instruction = ('\nFor this turn evidence acquisition is CLOSED. The only allowed action is finish. '
            'The supplied observations are the evidence already acquired; do not restart the scene analysis. '
            'If an evidence obligation remains unmet, disclose it in unresolved and choose the best supported '
            'allowed target, or null if essential evidence is unavailable. Do not call inspect or measure.'
            if final_only else '')
        response = chat([{'role': 'system', 'content': INSTRUCTIONS+CATEGORY_INSTRUCTIONS+final_instruction},
            {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps(state, ensure_ascii=False)},
                                      *[image_part(v['image']) for v in attached]]}],
            max_tokens=1600, json_mode=True, **schema_kwargs('r18_react_action', action_schema))
        call.update(response=response['_content'], usage=response.get('usage'), **response.get('_baseline', {}))
        if response.get('_finish_reason') != 'stop':
            raise ValueError('ReAct action response is incomplete')
        action = parse_json_object(response['_content'])
        actions.append(action)
        if action.get('action') == 'finish':
            decision = validate_decision(action, query, allowed, known)
            oid = decision['predicted_id']
            missing = []
            required_ids = required_measure_ids(parsed, allowed, objects)
            reference_ids = reference_review_ids(parsed, objects)
            if len(allowed) > 1 and not observations:
                missing.append('compare the multiple target candidates with one evidence observation before finishing')
            if (len(allowed) > 1 and requires_measure(query, parsed)
                    and not measured_together(required_ids, observations)):
                missing.append('measure all target candidates and bound references for the spatial/size clause')
            unseen_references = set(reference_ids) - inspected
            if (len(allowed) > 1 and requires_measure(query, parsed) and unseen_references):
                missing.append('inspect all classifier-covered reference candidates before binding a relation or qualifier')
            category_ids = category_review_ids(oid, best_ranks)
            if category_ids and not set(category_ids).issubset(inspected):
                missing.append('inspect the selected alternative category together with every rank-1 target-category rival')
            unseen_fallback = sorted(fallback_ids - inspected)
            if (len(allowed) > 1 and requires_measure(query, parsed) and unseen_fallback):
                missing.append('inspect geometry-only fallback candidates before accepting the spatial comparison')
            if missing and guard_attempts < 4:
                # Controller evidence is separately bounded; reminders never consume tools.
                if (len(allowed) > 1 and requires_measure(query, parsed)
                        and not measured_together(required_ids, observations)):
                    observation = measure(required_ids, objects, parsed, candidates, query,
                                          latest_frame_hint(actions))
                    observation.update(controller_forced=True,
                                      reason='complete the comparison pool before accepting finish')
                    observations.append(observation)
                    trace['steps'].append({'action': action, 'observation': observation})
                    trace['internal_usage']['compute'] += 1
                    guard_attempts += 1
                    continue
                focus = set(category_ids) - inspected
                if len(allowed) > 1 and requires_measure(query, parsed):
                    focus.update(unseen_references)
                    focus.update(unseen_fallback)
                if len(allowed) > 1 and not observations:
                    focus.update(allowed)
                if focus and not guard_prompted and tool_calls < limit:
                    observation = {'tool': 'finish_check', 'accepted': False,
                                   'missing': missing, 'focus_ids': sorted(focus)}
                    observations.append(observation)
                    trace['steps'].append({'action': action, 'observation': observation})
                    guard_prompted = True
                    continue
                if focus:
                    focus = sorted(focus)[:12]
                    camera = {'azimuth_deg': 315, 'elevation_deg': 30, 'context': 'isolated'}
                    try:
                        rendered = [render_action({**camera, 'focus_ids': focus[offset:offset+6]},
                            objects, xyz, rgb, ids, Path(out), 3200+12*step+offset//6)
                            for offset in range(0, len(focus), 6)]
                        views.extend(rendered)
                        inspected.update(i for view in rendered for i in view.get('marked_ids', []))
                        observation = {'tool': 'render', 'focus_ids': focus,
                            'controller_forced': True,
                            'views': [{k: v for k, v in view.items() if k not in ('image', 'panels')}
                                      for view in rendered]}
                        trace['internal_usage']['render'] += sum(v.get('views_rendered', 1) for v in rendered)
                    except ValueError as exc:
                        observation = {'tool': 'render', 'focus_ids': focus,
                            'controller_forced': True, 'error': str(exc), 'no_evidence_acquired': True}
                    observations.append(observation)
                    trace['steps'].append({'action': action, 'observation': observation})
                    guard_attempts += 1
                    continue
            if missing:
                decision.setdefault('unresolved', []).extend(missing)
            elif len(allowed) > 1 and not final_review_done:
                review = {
                    'tool': 'finish_check', 'accepted': False,
                    'proposed_id': oid,
                    'missing': ('Adversarially re-check the proposed target against its strongest rival. '
                                'Re-read every clause, category rank, measured relation and observer frame; '
                                'then finish again or acquire one discriminating observation.')}
                observations.append(review)
                trace['steps'].append({'action': action, 'observation': review})
                final_review_done = True
                reserved_evidence = True
                continue
            break
        if final_only:
            if budget_reminder_sent:
                raise ValueError('ReAct must finish after the budget-exhausted reminder')
            observation = {'tool': 'budget_exhausted', 'requested_tool': action['action'],
                'no_evidence_acquired': True,
                'message': 'No evidence calls remain; finish using the observations already acquired.'}
            observations.append(observation)
            trace['steps'].append({'action': action, 'observation': observation})
            budget_reminder_sent = True
            continue
        if reserved_evidence and tool_calls >= limit:
            # One extra model-selected observation is reserved after the adversarial
            # finish check; this is still bounded and is the only budget extension.
            reserved_evidence = False
        focus = action.get('focus_ids')
        try:
            if (not isinstance(focus, list) or not focus or len(set(focus)) != len(focus)
                    or any(type(i) is not int or i not in known for i in focus)):
                raise ValueError('Supply distinct known focus_ids for evidence acquisition')
            signature = json.dumps({k: action.get(k) for k in
                                    ('action', 'focus_ids', 'context', 'azimuth_deg', 'elevation_deg', 'frame')}, sort_keys=True)
            if signature in signatures and action['action'] != 'inspect':
                raise ValueError('This observation was already acquired; select different evidence or finish')
            tool_calls += 1
            if action['action'] == 'inspect':
                if len(focus) > 12:
                    raise ValueError('Inspect at most 12 objects; select the relevant comparison group')
                camera = {key: action.get(key) if action.get(key) is not None else default
                          for key, default in (('azimuth_deg', 315), ('elevation_deg', 30))}
                rendered = [render_action({**action, **camera, 'focus_ids': focus[offset:offset+6]},
                    objects, xyz, rgb, ids, Path(out), 2000+12*step+offset//6)
                    for offset in range(0, len(focus), 6)]
                views.extend(rendered)
                inspected.update(i for view in rendered for i in view.get('marked_ids', []))
                observation = {'tool': 'render', 'focus_ids': focus,
                    'camera_defaults_used': [key for key in camera if action.get(key) is None],
                    'views': [{k: v for k, v in view.items() if k not in ('image', 'panels')}
                              for view in rendered]}
                trace['internal_usage']['render'] += sum(v.get('views_rendered', 1) for v in rendered)
            elif action['action'] == 'measure':
                observation = measure(focus, objects, parsed, candidates, query, action.get('frame'))
                trace['internal_usage']['compute'] += 1
            else:
                raise ValueError('Unknown evidence action')
            signatures.add(signature)
        except ValueError as exc:
            observation = {'tool': action.get('action'), 'error': str(exc), 'no_evidence_acquired': True}
        observations.append(observation)
        trace['steps'].append({'action': action, 'observation': observation})
    else:
        raise ValueError('ReAct failed to produce a final decision within its bounded trajectory')
    trace.update(r18_model_tool_calls=tool_calls, r18_controller_tool_calls=guard_attempts)
    oid = decision['predicted_id']
    selected = next((o for o in candidates if o['bbox_id'] == oid), None)
    trace.update(stop_reason='r18_react_finish', final_candidate_ids=allowed, r18_final_decision=decision,
        final_evidence_state={'status': 'tentative' if oid is not None else 'missing',
                              'missing': decision.get('unresolved', [])})
    trace['steps'].append({'action': 'finish', 'decision': decision})
    trace['internal_usage']['render'] += len(overview)
    trace['internal_usage']['visual_check'] += len(actions)
    return {'query': query, 'predicted_id': oid, 'bbox_3d': selected['bbox_3d'] if selected else None,
        'implementation_version': settings()['version'],
        'prediction_source': 'r18_react_tentative' if oid is not None else 'abstained',
        'diagnosis': diagnosis, 'decision_status': 'tentative' if oid is not None else 'missing',
        'accepted': False, 'images': [v['image'] for v in views], 'response': call.get('response', ''),
        'decision_reasons': [decision['reason']], 'scene_evidence': decision, 'active_trace': trace,
        'final_review': {'status': 'included_in_react_trajectory', 'reason': 'Final choice follows model-selected evidence observations'},
        'backtrack_count': 0, 'internal_budget_usage': dict(parsed['_v4_budget']),
        'diagnosis_policy': 'predicted_category_multimodal_react'}
