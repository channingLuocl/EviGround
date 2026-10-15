"""Model-directed, bounded evidence loop for candidate disambiguation."""
from copy import deepcopy
import hashlib
import json
import re

import numpy as np
from PIL import Image

from diagnostic import (attribute_role, attributes_for, appearance_attributes_for, size_conditions,
                        candidate_hypothesis_support,
                        diagnose, viewpoint_evidence, size_category_scope,
                        candidate_removal_gate)
from evidence import distance_rankings, match_retrieval_categories, retrieval_category
from appearance_evidence import (aggregate_visual_checks, attribute_contexts, attribute_units,
                                  normalize_visual_check, required_visual_attributes)
from object_geometry import build_geometry, point_cloud_measurements
from qwen_api import QwenAPIError, chat, image_part, parse_json_object, api_provider
from response_schemas import APPEARANCE, planner_schema, schema_kwargs
from react_decision import jev_action, react_backend, text_state
from run import render
from v4_policy import enabled, v5_enabled, v6_enabled, v7_enabled, v73_enabled, v74_enabled, settings as v4_settings, evidence_inventory, deletion_events, backtrack_events
from v4_visual import boxed_view, visual_report
from v4_relations import bind_tasks
from planner_prompt import DIMENSIONS, SIZE_WORDS, SIZE_EXTREMES, EVIDENCE_CAPABILITIES, INSTRUCTIONS
from computed_evidence import (_target_size_scope, position_candidate_evidence,
                               distance_candidate_evidence, size_candidate_evidence,
                               apply_computed_evidence, apply_size_observation)


def geometry_observation(diagnosis, category_candidates):
    geometry = diagnosis.get('geometry', {})
    return {
        'tool': 'initial_diagnosis',
        'status': diagnosis.get('status'),
        'category_candidates': category_candidates,
        'remaining_candidate_ids': diagnosis.get('candidate_ids', []),
        'stable_candidate_ids': diagnosis.get('stable_candidate_ids', []),
        'missing': diagnosis.get('missing', []),
        'pending_size_conditions': diagnosis.get('pending_size_conditions', []),
        'geometry': {
            key: value for key, value in geometry.items()
            if key not in ('candidate_metrics', 'reference_metrics', 'size_policy')
        },
        'policy': 'baseline audit only; choose the next evidence dimension from the query',
    }


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
    panels = []
    if context == 'isolated':
        # Each requested instance gets its own local camera, so distant objects
        # do not shrink each other or obscure a color comparison.
        tile_size = 600
        columns = min(3, len(focus))
        rows = (len(focus) + columns - 1) // columns
        canvas = Image.new('RGB', (columns*tile_size, rows*tile_size), (245, 245, 245))
        for offset, oid in enumerate(focus):
            instance = ids == oid
            if not instance.any():
                continue
            box = np.asarray(known[oid]['bbox_3d'])
            local_at = box[:3]
            local_eye = local_at + max(float(np.linalg.norm(box[3:])), .5)*1.5*direction
            local_path = out / f'active_view_{index:02d}_instance_{oid}.png'
            local = render(xyz[instance], rgb[instance], local_path,
                           eye=local_eye, at=local_at, ids=ids[instance],
                           size=tile_size, point_radius=4)
            tile_xy = ((offset % columns)*tile_size, (offset // columns)*tile_size)
            with Image.open(local_path) as tile:
                canvas.paste(tile, tile_xy)
            panels.append({'id': oid, 'image': str(local_path),
                           'tile_xy_px': list(tile_xy), 'tile_size_px': tile_size, **local})
        canvas.save(path)
        meta = {'points': sum(panel['points'] for panel in panels),
                'visible_pixels': sum(panel['visible_pixels'] for panel in panels),
                'marked_ids': [oid for panel in panels for oid in panel['marked_ids']],
                'point_radius_px': 4,
                'visible_instance_pixels': {key: value for panel in panels
                                            for key, value in panel['visible_instance_pixels'].items()},
                'panels': panels, 'views_rendered': len(panels),
                'layout': 'one independently framed tile per instance; no shared spatial frame'}
    else:
        meta = render(xyz[keep], rgb[keep], path, eye=eye, at=at,
                      ids=labels[keep], size=1000, point_radius=4)
    visible = set(meta['marked_ids'])
    return {
        'tool': 'render', 'image': str(path), 'focus_ids': focus,
        'context': context, 'azimuth_deg': action['azimuth_deg'],
        'elevation_deg': action['elevation_deg'],
        'unmarked_focus_ids': [oid for oid in focus if oid not in visible],
        'ceiling_ids_hidden': ceiling_ids if context == 'scene' else [],
        'provenance': 'CPU colored point-cloud inspection with depth-tested 4px point disks; not a real RGB crop',
        **meta,
    }


def compute_observation(dimension, query, parsed, diagnosis, candidates, geometry_objects,
                        xyz=None, ids=None):
    geometry = diagnosis.get('geometry', {})
    if dimension == 'viewpoint':
        evidence = viewpoint_evidence(parsed, diagnosis)
        return {
            'tool': 'compute',
            **evidence,
            'candidate_ids': diagnosis.get('candidate_ids', []),
            'policy': 'language reference frames only; capture cameras are inspection evidence and never speaker frames',
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
        if 'distance_rankings' in geometry:
            rankings = {
                'rankings': geometry['distance_rankings'],
                'ignored': geometry.get('distance_ignored', []),
                'scope': 'the diagnosis reference and observer hypotheses',
            }
        else:
            rankings = distance_rankings(query, parsed, candidates, geometry_objects,
                                         viewpoint=parsed.get('Observer'))
        candidate_evidence = distance_candidate_evidence(
            rankings.get('rankings', []), diagnosis.get('candidate_ids', []), diagnosis)
        return {
            'tool': 'compute', 'dimension': dimension,
            'distance_rankings': rankings,
            'candidate_evidence': candidate_evidence,
            'candidate_ids': diagnosis.get('candidate_ids', []),
            'policy': 'explicit nearest/farthest rankings per bound anchor and selected metric; '
                      'only definite fail is removable; near/far and missing viewpoints remain unknown',
        }
    if dimension == 'size':
        scope_ids = set(diagnosis.get('candidate_ids', []))
        scope = [obj for obj in candidates if obj['bbox_id'] in scope_ids]
        comparison_pool_ids, comparison_scope_bound, scope_unknown_ids = _target_size_scope(
            parsed, diagnosis, candidates)
        reference_size_categories = {retrieval_category(obj['target']) for obj in geometry_objects
                                    if size_conditions(parsed, obj['target'], role='reference')}
        if v7_enabled() and not comparison_scope_bound:
            comparison_pool_ids = list(diagnosis.get('size_comparison_pool_ids', [o['bbox_id'] for o in candidates]))
        measurement_scope = [obj for obj in geometry_objects
                             if obj['bbox_id'] in set(comparison_pool_ids or [])
                             or retrieval_category(obj['target']) in reference_size_categories
                             or (parsed.get('relative_height_binding') and
                                 retrieval_category(obj['target']) == retrieval_category(parsed['Target'][0]))]
        measurements = point_cloud_measurements(measurement_scope, xyz, ids, scene_objects=geometry_objects)
        if v7_enabled():
            from v7_evidence import box_rows
            measurements = box_rows(measurement_scope, measurements)
        if v5_enabled():
            for row in measurements:
                previous = parsed.get('_v5_measurement_observations', {}).get(str(row['id']))
                if previous and previous['measurement_version'] == measurement_version(row):
                    row['observed_completeness'] = deepcopy(previous['observed_completeness'])
        candidate_evidence, constraints, decision = size_candidate_evidence(
            parsed, scope, measurements, comparison_pool_ids=comparison_pool_ids,
            comparison_scope_bound=comparison_scope_bound, objects=geometry_objects,
            checks=diagnosis.get('checks', []))
        joint_size_diagnosis = None
        if parsed.get('relative_height_binding') or reference_size_categories:
            joint_size_diagnosis = diagnose(parsed, geometry_objects, candidates,
                diagnosis.get('room_center', np.mean(xyz, axis=0) if xyz is not None else [0., 0., 0.]),
                diagnosis.get('checks', []), trusted_categories=diagnosis.get('category_source') == 'annotated',
                apply_size_constraints=True, query=query, size_measurements=measurements)
            joint_size_diagnosis['size_comparison_pool_ids'] = comparison_pool_ids
            constraints = joint_size_diagnosis['geometry']['size_constraints']
            remaining = set(joint_size_diagnosis['candidate_ids'])
            candidate_evidence = []
            for obj in scope:
                oid = obj['bbox_id']
                proofs = [e for h in joint_size_diagnosis['hypotheses'] if oid in h['candidate_ids']
                          for m in h.get('measurements', []) if m.get('target_id') == oid
                          and m.get('interpretation') == 'reference_comparison'
                          for e in m.get('candidate_evidence', []) if e['candidate_id'] == oid]
                if not parsed.get('relative_height_binding'):
                    proofs = [e for c in constraints for e in c.get('candidate_evidence', [])
                              if e['candidate_id'] == oid and c.get('role') == 'target']
                state = ('fail' if oid not in remaining else 'unknown' if not proofs or any(
                    e['status'] == 'unknown' for e in proofs) else 'pass')
                candidate_evidence.append({'candidate_id': oid, 'status': state,
                    'metrics': next((r for r in measurements if r['id'] == oid), {}),
                    'attribute_evidence': proofs,
                    'exclusion_reason': 'joint between/relative-height predicates fail in all legal assignments'
                                        if state == 'fail' else None,
                    'source_quote': next((c.get('original_quote', '') for c in constraints), ''),
                    'target_role': 'target',
                    'role_binding_status': ('bound' if constraints and all(
                        c.get('scope_status') == 'bound' for c in constraints) else 'unknown'),
                    'comparison_scope_ids': sorted({sid for c in constraints
                                                    for sid in c.get('comparison_pool_ids', [])
                                                    if type(sid) is int}),
                    'comparison_scope_status': ('bound' if constraints and all(
                        c.get('scope_status') == 'bound' for c in constraints) else 'unknown'),
                    'measurement_source': 'joint bound relation measurements',
                    'explicit_fail_criterion': ('all legal bound hypotheses fail the quoted predicate'
                                                if state == 'fail' else None)})
            decision.update(attributes=constraints, joint_hypotheses=joint_size_diagnosis['hypotheses'])
        category_scope = size_category_scope(measurement_scope)
        if category_scope:
            decision['comparison_category_labels'] = category_scope['category_labels']
            decision['scope_missing'] = [category_scope['reason']]
        if scope_unknown_ids:
            decision['scope_unknown_ids'] = scope_unknown_ids
        candidate_measurements = [row for row in measurements
                                  if row['id'] in scope_ids]
        return {
            'tool': 'compute', 'dimension': dimension,
            'candidate_metrics': candidate_measurements,
            'comparison_metrics': measurements,
            'candidate_evidence': candidate_evidence,
            'size_constraints': constraints,
            'size_decision': decision,
            'joint_size_diagnosis': joint_size_diagnosis,
            'reference_metrics': geometry.get('reference_metrics', []),
            'geometry_conditions': build_geometry(query, parsed, geometry_objects),
            'candidate_ids': diagnosis.get('candidate_ids', []),
            'comparison_pool_ids': decision.get('comparison_pool_ids', []),
            'policy': 'shared source-bound size predicates and reliable instance geometry; '
                      'comparison pool is frozen at initial category candidates',
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


def visual_attributes(parsed):
    phrases = []
    for name in parsed.get('Target', []):
        values = required_visual_attributes(appearance_attributes_for(parsed, name, role='target'))
        phrases.extend(value for value in values if value not in phrases)
    return phrases


def _appearance_attributes_for(parsed, category):
    """Return independently checkable appearance modifiers for the target role."""
    return required_visual_attributes(appearance_attributes_for(parsed, category, role='target'))


def _appearance_review_requirements(parsed, diagnosis, focus_ids, objects):
    """Build the per-instance visual contract for one rendered request.

    Target candidates are always reviewable when focused. References are added
    only when the current diagnosis has a bound visual need for that instance.
    The role and category are kept with the requirement so an answer about a
    reference cannot be mistaken for evidence about the target.
    """
    known = {obj['bbox_id']: obj for obj in objects}
    target_names = parsed.get('Target', [])
    target_category = retrieval_category(target_names[0]) if target_names else ''
    target_ids = set(diagnosis.get('candidate_ids', []))
    needs = []
    for need in diagnosis.get('visual_needs', []):
        if not isinstance(need, dict) or type(need.get('id')) is not int:
            continue
        oid = need['id']
        if oid not in known or oid not in set(focus_ids):
            continue
        category = retrieval_category(need.get('category', '')) or known[oid].get('target', '')
        source_attributes = [value for value in
                             need.get('source_attributes', need.get('attributes', []))
                             if isinstance(value, str)]
        attrs = required_visual_attributes(need.get('attributes', source_attributes))
        record = {'id': oid, 'category': category, 'role': need.get('role', 'reference'),
                  'attributes': attrs, 'source_attributes': source_attributes,
                  'attribute_contexts': attribute_contexts(attrs, source_attributes)}
        if record not in needs:
            needs.append(record)
    result = []
    for oid in focus_ids:
        if oid not in known:
            continue
        if oid in target_ids:
            source_attributes = attributes_for(parsed, target_category, role='target')
            attrs = _appearance_attributes_for(parsed, target_category)
            result.append({'id': oid, 'category': target_category or known[oid].get('target', ''),
                           'role': 'target', 'attributes': attrs,
                           'source_attributes': source_attributes,
                           'attribute_contexts': attribute_contexts(attrs, source_attributes)})
        result.extend(need for need in needs if need['id'] == oid
                      and (oid not in target_ids or need['role'] != 'target'))
    return result


def _attribute_words(phrase):
    return attribute_units(phrase)


def _check_payload(action, focus_ids, candidates, parsed, requirements=None,
                   render_provenance=None):
    checks = action.get('checks')
    if not isinstance(checks, list) or not checks:
        raise ValueError('visual_check needs a non-empty checks list')
    allowed = {obj['bbox_id'] for obj in candidates} & set(focus_ids)
    target_names = parsed.get('Target', [])
    target_category = retrieval_category(target_names[0]) if target_names else ''
    requirements = requirements or []
    requirement_by_id = {}
    for item in requirements:
        requirement_by_id.setdefault(item['id'], []).append(item)
    if requirements:
        allowed = set(requirement_by_id) & set(focus_ids)
    result = []
    seen = set()
    for check in checks:
        if (not isinstance(check, dict) or type(check.get('id')) is not int
                or check['id'] not in allowed
                or check.get('verdict') not in ('yes', 'no', 'uncertain')
                or not isinstance(check.get('reason'), str)
                or not check['reason'].strip()):
            raise ValueError('visual_check IDs must be rendered candidates with yes/no/uncertain reasons')
        matching = requirement_by_id.get(check['id'], [{}])
        if 'category' in check:
            matching = [item for item in matching if
                        item.get('category') == retrieval_category(check['category'])]
        if 'role' in check:
            matching = [item for item in matching if item.get('role') == check['role']]
        if len(matching) != 1:
            raise ValueError('visual_check category and role must identify one supplied requirement')
        requirement = matching[0]
        category = requirement.get('category', target_category)
        key = (check['id'], category, requirement.get('role', 'target'))
        if key in seen:
            raise ValueError('visual_check cannot repeat the same instance/category/role')
        seen.add(key)
        record = {'id': check['id'], 'category': category,
                  'role': requirement.get('role', 'target'),
                  'verdict': check['verdict'], 'reason': check['reason'],
                  'source': 'active rendered view'}
        expected = list(requirement.get('attributes', [])) if requirements else visual_attributes(parsed)
        source_attributes = (requirement.get('source_attributes') if requirements else
                             attributes_for(parsed, target_category, role='target'))
        record['attribute_checks'] = check.get('attribute_checks', [])
        if 'category_verdict' in check:
            record['category_verdict'] = check['category_verdict']
        if 'visibility' in check:
            record['visibility'] = check['visibility']
        if render_provenance:
            record['render_provenance'] = dict(render_provenance)
        record = normalize_visual_check(record, expected, source_phrases=source_attributes)
        result.append(record)
    return result


def evidence_request(action, candidate_ids, geometry_objects):
    """Validate the public request and make inspection defaults explicit."""
    if v73_enabled():
        action = {key: value for key, value in action.items() if value is not None}
        dimension = action.get('dimension')
        # The strict planner schema shares optional fields across dimensions.
        # Appearance always renders; geometry inspections use fixed comparison
        # panels. These two fields therefore carry no executable choice here.
        if dimension == 'appearance' and 'inspect' in action:
            if type(action['inspect']) is not bool:
                raise ValueError('inspect must be a boolean')
            action.pop('inspect')
        if dimension in ('size', 'position', 'distance') and 'context' in action:
            if action['context'] not in ('scene', 'isolated'):
                raise ValueError('context must be scene or isolated')
            action.pop('context')
        # Viewpoint only binds an observer frame; it never renders a view.
        # Validate unused visual fields before removing them from this request.
        if dimension == 'viewpoint':
            if 'inspect' in action and type(action['inspect']) is not bool:
                raise ValueError('inspect must be a boolean')
            if 'context' in action and action['context'] not in ('scene', 'isolated'):
                raise ValueError('context must be scene or isolated')
            if 'focus_ids' in action:
                focus, known = action['focus_ids'], {o['bbox_id'] for o in geometry_objects}
                if (not isinstance(focus, list) or not 1 <= len(focus) <= 6 or
                        any(type(i) is not int or i not in known for i in focus) or
                        len(set(focus)) != len(focus)):
                    raise ValueError('viewpoint visual fields need 1-6 distinct known IDs')
            for key, lower, upper in (('azimuth_deg', 0, 360), ('elevation_deg', 10, 76)):
                if key in action and (type(action[key]) not in (int, float) or
                        not np.isfinite(action[key]) or not lower <= action[key] < upper):
                    raise ValueError('invalid common-camera inspection angle')
            for key in ('inspect', 'context', 'focus_ids', 'azimuth_deg', 'elevation_deg'):
                action.pop(key, None)
    dimension = action.get('dimension')
    if action.get('action') != 'evidence' or dimension not in DIMENSIONS:
        raise ValueError('choose action=evidence with a supported dimension, finish or stop')
    allowed = {'action', 'dimension', 'reason'}
    if enabled() and dimension in ('size', 'position', 'distance'):
        allowed.update(('inspect', 'focus_ids', 'azimuth_deg', 'elevation_deg'))
        if 'inspect' in action and type(action['inspect']) is not bool:
            raise ValueError('inspect must be a boolean')
        known = {o['bbox_id'] for o in geometry_objects}
        focus = action.get('focus_ids')
        if focus is not None and (not isinstance(focus, list) or not 1 <= len(focus) <= 6 or
                any(type(i) is not int or i not in known for i in focus) or len(set(focus)) != len(focus)
                or not set(focus).intersection(candidate_ids)):
            raise ValueError('inspection needs 1-6 known IDs including a remaining target')
        for key, lower, upper in (('azimuth_deg', 0, 360), ('elevation_deg', 10, 76)):
            if key in action and (type(action[key]) not in (int, float) or not np.isfinite(action[key])
                                  or not lower <= action[key] < upper):
                raise ValueError('invalid common-camera inspection angle')
    if dimension == 'appearance':
        allowed.update(('focus_ids', 'azimuth_deg', 'elevation_deg', 'context'))
    if set(action) - allowed:
        if v73_enabled():
            raise ValueError(f'Unsupported fields for {dimension}: {sorted(set(action) - allowed)}. '
                             f'Use only {sorted(allowed)}; set other planner fields to null.')
        raise ValueError('unsupported evidence parameters; camera/focus options are appearance-only')
    request = dict(action)
    if dimension == 'appearance':
        request.setdefault('focus_ids', list(candidate_ids)[:6])
        request.setdefault('azimuth_deg', 90)
        request.setdefault('elevation_deg', 25)
        request.setdefault('context', 'isolated')
        focus = request['focus_ids']
        known = {obj['bbox_id'] for obj in geometry_objects}
        if (not isinstance(focus, list) or not 1 <= len(focus) <= 6
                or any(type(oid) is not int or oid not in known for oid in focus)
                or len(set(focus)) != len(focus) or not set(focus) & set(candidate_ids)):
            raise ValueError('appearance needs 1-6 distinct known focus_ids including a remaining target')
        azimuth, elevation = request['azimuth_deg'], request['elevation_deg']
        if (type(azimuth) not in (int, float) or not np.isfinite(azimuth)
                or not 0 <= azimuth < 360 or type(elevation) not in (int, float)
                or not np.isfinite(elevation) or not 10 <= elevation <= 75
                or request['context'] not in ('scene', 'isolated')):
            raise ValueError('invalid appearance camera: azimuth [0,360), elevation [10,75], scene|isolated')
    return request


def measurement_version(row):
    data = json.dumps({key: value for key, value in row.items()
        if key not in ('observed_completeness', 'visual_observations')}, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()


def execute_v4_inspection(action, query, parsed, diagnosis, candidates, objects,
                          xyz, rgb, ids, out, index, calls, result):
    """One optional common-camera VLM observation inside a chosen evidence task."""
    dimension, details = action['dimension'], result['details']
    budget = parsed.setdefault('_v4_budget', {'size': 0, 'relations': 0, 'review': 0})
    feature = 'size' if dimension == 'size' else 'relations'
    if not enabled(feature):
        return
    rows = details.get('comparison_metrics', [])
    relation_rows = bind_tasks(query, parsed, objects, candidates, diagnosis.get('checks', [])) if feature == 'relations' else []
    trigger = (bool(relation_rows) if feature == 'relations' else any(
        c.get('metric_ambiguity') or c.get('unknown_measurement_ids') or c.get('tentative_metric')
        for c in details.get('size_constraints', [])))
    trigger = trigger or (feature == 'size' and any(r.get('observed_completeness', {}).get('requires_visual_check')
        and (not v5_enabled('call_economy') or r.get('observed_completeness', {}).get('status') != 'supported')
        for r in rows))
    details['relation_tasks'] = relation_rows
    if not trigger and not action.get('inspect'):
        return
    if v5_enabled('call_economy') and feature == 'size' and not trigger:
        return
    cap = v4_settings()[feature + '_visual_calls_limit' if feature == 'size' else 'relation_visual_calls_limit']
    anchors = sorted({oid for r in relation_rows for oid in r['geometry_evidence'].get('anchor_ids', [])})
    pool = [r['id'] for r in rows] if feature == 'size' else [o['bbox_id'] for o in candidates]
    focus = action.get('focus_ids') or list(dict.fromkeys(pool[:max(1, 6 - len(anchors[:2]))] + anchors[:2]))[:6]
    relevant_quotes = {c.get('original_quote') for c in details.get('size_constraints', [])}
    relevant_quotes.update(q for c in details.get('size_constraints', []) for q in c.get('scope_relation_quotes', []))
    relevant_quotes.update(r.get('source_quote') for r in relation_rows)
    cache_key = json.dumps({'query': query, 'feature': feature, 'focus': focus, 'measurements': rows,
        'relations': relation_rows, 'constraints': details.get('size_constraints', []), 'azimuth': action.get('azimuth_deg', 90),
        'elevation': action.get('elevation_deg', 30),
        'invalidated': sorted(set(parsed.get('_v4_invalidated_conditions', [])) & relevant_quotes)}, sort_keys=True)
    cached = parsed.get('_v5_visual_cache', {}).get(cache_key) if v5_enabled('call_economy') else None
    if cached:
        view, report = deepcopy(cached['view']), deepcopy(cached['report'])
        report['cache_hit'] = True
    else:
        if budget[feature] >= cap:
            details['visual_inspection'] = {'status': 'unknown', 'reason': 'internal visual-call budget exhausted'}
            return
        view = boxed_view(objects, focus, xyz, rgb, ids, out, f'v4_{feature}_{index}', rows,
                          action.get('azimuth_deg', 90), action.get('elevation_deg', 30))
        budget[feature] += 1
        report = visual_report('active_' + feature + '_visual_check', query, view,
            {'purpose': feature + ' evidence', 'constraints': details.get('size_constraints', []),
             'measurements': rows, 'relations': relation_rows}, calls, chat_fn=chat)
        if v5_enabled('call_economy'):
            parsed.setdefault('_v5_visual_cache', {})[cache_key] = {'view': deepcopy(view), 'report': deepcopy(report)}
        result['images'].append(view['image'])
        result['internal_steps'].extend([{'tool': 'render', **view},
                                        {'tool': 'visual_check', 'call': calls[-1]}])
    details['visual_inspection'] = report
    if feature == 'size':
        for observation in report['observations']:
            row = next((r for r in rows if r['id'] == observation['id']), None)
            if row:
                if v5_enabled() and observation['completeness'] == 'unknown':
                    row.setdefault('visual_observations', []).append(observation)
                    continue
                row['observed_completeness'].update(status=observation['completeness'],
                                                     visual_evidence=observation)
                if v5_enabled():
                    parsed.setdefault('_v5_measurement_observations', {})[str(row['id'])] = {
                        'measurement_version': measurement_version(row),
                        'observed_completeness': deepcopy(row['observed_completeness'])}
        for choice in report['metric_choices']:
            category = retrieval_category(choice.get('category', ''))
            specs = size_conditions(parsed, category)
            spec = next((s for s in specs if s['word'] == choice.get('word') and
                         s['original_quote'].casefold() in choice['source_quote'].casefold() and
                         (s['metric'] in ('oriented_volume_m3', 'oriented_footprint_m2', 'body_plane_area_m2') or
                          (v7_enabled() and category == 'book'))), None)
            if spec:
                parsed.setdefault('_v4_metric_choices', {})['|'.join(('target', category, spec['word']))] = choice
                parsed['_v4_invalidated_conditions'] = [q for q in parsed.get('_v4_invalidated_conditions', [])
                                                        if q != spec['original_quote']]
        scope = [o for o in candidates if o['bbox_id'] in diagnosis['candidate_ids']]
        evidence, constraints, decision = size_candidate_evidence(parsed, scope, rows,
            comparison_pool_ids=details['comparison_pool_ids'], objects=objects, checks=diagnosis.get('checks', []))
        # Joint reference/height hypotheses keep their independent binding requirements.
        if details.get('joint_size_diagnosis') is None:
            details.update(candidate_evidence=evidence, size_constraints=constraints, size_decision=decision)
            result['candidate_evidence'] = evidence
    else:
        verified = []
        for row in relation_rows:
            item = next((r for r in report['relations'] if r['candidate_id'] == row['candidate_id'] and
                         r['source_quote'].casefold() == row['source_quote'].casefold()), None)
            geometry = row['geometry_evidence']
            anchors_visible = all(view['visible_instance_pixels'].get(str(oid), 0) >= 20
                                  for oid in geometry.get('anchor_ids', []))
            state = (item['status'] if item and item['status'] == geometry['status'] and
                     geometry.get('anchor_ids') and anchors_visible else 'unknown')
            binding_proofs = []
            if v5_enabled('scope') and geometry.get('anchor_hypotheses'):
                states = []
                for binding in geometry['anchor_hypotheses']:
                    anchor = binding['anchor_ids'][0] if binding.get('anchor_ids') else None
                    proof = next((p for p in report['relations'] if p.get('anchor_id') == anchor
                        and p['candidate_id'] == row['candidate_id'] and
                        p['source_quote'].casefold() == row['source_quote'].casefold()), None)
                    states.append(binding['status'] if proof and proof['status'] == binding['status']
                                  and view['visible_instance_pixels'].get(str(anchor), 0) >= 20 else 'unknown')
                    binding_proofs.append({'anchor_id': anchor, 'status': states[-1],
                                           'geometry_evidence': binding, 'visual_evidence': proof})
                state = 'pass' if 'pass' in states else 'fail' if all(s == 'fail' for s in states) else 'unknown'
                if geometry.get('reference_binding', {}).get('status') == 'unknown':
                    state = 'unknown'
            reference_bound = bool(geometry.get('anchor_ids')) and geometry.get('reference_binding', {}).get('status') != 'unknown'
            verified.append({**row, 'status': state, 'visual_evidence': item,
                             **({'reference_bindings': binding_proofs} if binding_proofs else {}),
                             'measurement_source': view['source'], 'reference_role': geometry.get('anchor_ids'),
                             'role_binding_status': 'bound' if reference_bound else 'unknown',
                             'comparison_scope_status': 'bound' if reference_bound else 'unknown',
                             'comparison_scope_ids': [o['bbox_id'] for o in candidates], 'target_role': 'target',
                             'explicit_fail_criterion': 'bound box separation and visible relation both contradict quote' if state == 'fail' else None})
        keys = {(r['candidate_id'], r['source_quote']) for r in verified}
        if v5_enabled():
            previous_rows = {(r['candidate_id'], r['source_quote']): r for r in parsed.get('_v4_relation_evidence', [])}
            for index, row in enumerate(verified):
                previous = previous_rows.get((row['candidate_id'], row['source_quote']))
                if (row['status'] == 'unknown' and previous and previous['status'] in ('pass', 'fail')
                        and previous.get('geometry_evidence') == row['geometry_evidence']):
                    verified[index] = {**previous, 'additional_visual_observations':
                        previous.get('additional_visual_observations', []) + [row.get('visual_evidence')]}
        cache = [r for r in parsed.get('_v4_relation_evidence', []) if (r['candidate_id'], r['source_quote']) not in keys]
        parsed['_v4_relation_evidence'] = cache + verified
        details['relation_tasks'] = cache + verified
        evidence = []
        for oid in diagnosis['candidate_ids']:
            proofs = [r for r in cache + verified if r['candidate_id'] == oid]
            if not proofs:
                continue
            decisive = next((r for r in proofs if r['status'] == 'fail'), proofs[0])
            state = 'fail' if any(r['status'] == 'fail' for r in proofs) else 'pass' if all(r['status'] == 'pass' for r in proofs) else 'unknown'
            evidence.append({**decisive, 'status': state, 'relation_evidence': proofs})
        # Existing definite predicates continue to govern; new unknowns cannot erase them.
        by_id = {r['candidate_id']: r for r in result['candidate_evidence']}
        for row in evidence:
            previous = by_id.get(row['candidate_id'], {})
            if previous.get('status') != 'fail':
                by_id[row['candidate_id']] = row
        result['candidate_evidence'] = list(by_id.values())
        details['candidate_evidence'] = result['candidate_evidence']


def apply_v4_relation_resolution(diagnosis, parsed):
    """Clear only the explicitly covered quoted relation gap, never other blockers."""
    rows = parsed.get('_v4_relation_evidence', [])
    live = set(diagnosis.get('candidate_ids', []))
    completed = {r['source_quote'] for r in rows if live and all(any(
        p['candidate_id'] == oid and p['source_quote'] == r['source_quote'] and p['status'] == 'pass'
        for p in rows) for oid in live)}
    missing = diagnosis.get('missing', [])
    diagnosis['missing'] = [s for s in missing if not any(s in
        ('unsupported relation: ' + q, 'unhandled condition: ' + q) for q in completed)]
    near_quotes = [r['phrase'] for r in parsed.get('Relations', []) if r.get('relation') == 'near']
    if near_quotes and all(q in completed for q in near_quotes):
        diagnosis['missing'] = [s for s in diagnosis['missing'] if s != 'near has no calibrated predicate; distances are evidence only']
    if rows and not diagnosis['missing'] and len(live) == 1 and not diagnosis.get('pending_size_conditions'):
        diagnosis['status'] = 'resolved'


def execute_evidence(action, query, parsed, diagnosis, candidates, geometry_objects,
                     xyz, rgb, ids, out, index, calls):
    """Run one bounded evidence task; candidate changes belong to the controller."""
    dimension = action['dimension']
    before = list(diagnosis.get('candidate_ids', []))
    details = compute_observation(dimension, query, parsed, diagnosis, candidates, geometry_objects,
                                   xyz, ids)
    result = {
        'tool': 'evidence', 'dimension': dimension, 'task': action['reason'],
        'scope_ids': before, 'candidate_evidence': details.get('candidate_evidence', []),
        'details': details, 'images': [], 'checks': [], 'missing': [],
        'internal_steps': [details], 'operation_status': 'information_only',
    }
    if enabled() and dimension in ('size', 'position', 'distance'):
        execute_v4_inspection(action, query, parsed, diagnosis, candidates, geometry_objects,
                              xyz, rgb, ids, out, index, calls, result)
    if dimension == 'appearance':
        view = render_action(action, geometry_objects, xyz, rgb, ids, out, index)
        result['internal_steps'].append(view)
        result['images'].append(view['image'])
        requirements = _appearance_review_requirements(parsed, diagnosis, view['focus_ids'], geometry_objects)
        result['review_requirements'] = requirements
        visible_requirements = [item for item in requirements
                                if item['id'] not in view.get('unmarked_focus_ids', [])]
        visible_ids = [item['id'] for item in visible_requirements]
        provenance = {'image': view['image'], 'source_type': 'colored_point_cloud',
                      'description': view.get('provenance', 'CPU colored point-cloud inspection; not a real RGB crop'),
                      'context': view.get('context', action.get('context')),
                      'azimuth_deg': view.get('azimuth_deg', action.get('azimuth_deg')),
                      'elevation_deg': view.get('elevation_deg', action.get('elevation_deg')),
                      'language_frame_changed': False}
        groups = [visible_requirements]
        if len(set(visible_ids)) != len(visible_ids):
            # One row per ID is ambiguous when that instance has multiple roles.
            keys = list(dict.fromkeys((item['category'], item['role'])
                                      for item in visible_requirements))
            groups = [[item for item in visible_requirements
                       if (item['category'], item['role']) == key] for key in keys]
        for group in groups:
            output_schema = schema_kwargs('appearance_checks', APPEARANCE)
            call = {'stage': 'active_evidence_visual_check', 'dimension': dimension,
                'backend': api_provider(), 'role': 'visual_verifier',
                'step': index, 'image': view['image'], 'focus_ids': view['focus_ids'],
                'review_requirements': group,
                'response_format': {'type': 'json_object'}}
            if output_schema:
                call['response_format'] = {'type': 'json_schema', 'json_schema': output_schema['response_schema']}
            result['internal_steps'].append({'tool': 'visual_check', 'call': call})
            if not group:
                call.update(skipped=True, reason='no marked instance has a visual review requirement')
                continue
            calls.append(call)
            try:
                data = chat([
                    {'role': 'system', 'content':
                        'Verify only the supplied target candidates and bound reference requirements in the supplied image. '
                        'Return JSON {"checks":[{"id":1,"category":"supplied category",'
                        '"role":"supplied role","category_verdict":"yes|no|uncertain",'
                        '"verdict":"yes|no|uncertain",'
                        '"visibility":"visible|occluded|low_detail|unknown",'
                        '"reason":"category evidence","attribute_checks":['
                        '{"attribute":"exact supplied attribute","verdict":"yes|no|uncertain",'
                        '"visibility":"visible|occluded|low_detail|unknown",'
                        '"reason":"visible evidence"}]}]}. category_verdict and top-level verdict check category only; '
                        'a failed color does not make the annotated object a different category. '
                        'Check EACH per-instance supplied attribute separately, including every modifier. '
                        + ('For not-color, yes means visibly incompatible with that color, no means visibly that color; '
                           'uncertain or missing visibility never establishes not-color. ' if v7_enabled() else '') +
                        'Interpret fragments in their source_quotes; a color on a stripe or panel is not a whole-body color. '
                        'A bare color does not require every part or a predominant fraction to share that color. '
                        'Do not add purity, uniformity, exact shade, or whole-object coverage absent from the quotes. '
                        'Use ordinary surface-color matching: yes for a directly supported described color, '
                        'no for a clear incompatible visible color, and uncertain for actually ambiguous or obscured evidence. '
                        'A modifier also present in an independent source attribute must satisfy that independent condition. '
                        'The complete quoted qualified property remains mandatory; component yes alone cannot prove it. '
                        'A reference attribute belongs only to its listed reference instance, never to the target. '
                        'Return one check for EACH supplied (id, category, role) requirement, even when uncertain. '
                        'Red numbers and white label boxes are annotations, not object surface colors. '
                        'Visibility means evidence sufficiency for the specific category or property, not photographic realism. '
                        'Use visible for an unambiguous sampled surface color; point-cloud rendering alone does not imply low_detail. '
                        'Background gaps between point samples are not foreground occluders; assess missing samples for the requested property. '
                        'Use low_detail when the requested property cannot actually be distinguished, '
                        'and use uncertain for every non-visible check. Materials, texture and qualified parts need their own detail; '
                        'visible color alone does not establish them. '
                        'Isolated tiles use independent cameras and scales; do not compare physical sizes '
                        'or spatial relations from tile positions or displayed object heights. '
                        'For tall, short, small, large and other physical-scale conditions, isolated tiles '
                        'require uncertain; both positive and negative comparisons lack a shared scale. '
                        'Copy its category and role exactly; never merge requirements with the same ID. '
                        'Do not infer missing surface details from the category. '
                        'Compare physical height, not elevation above the floor or screen position. '
                        'Skinny is contextual visual slenderness, not a fixed horizontal box ratio. '
                        'Use no only for visible contradiction; '
                        'use uncertain for occlusion or insufficient detail. Do not judge spatial '
                        'relations, exclude unchecked alternatives, or invent IDs. '
                        'Only supplied attribute quotes are conditions; do not import other conditions.'},
                    {'role': 'user', 'content': [image_part(view['image']),
                        {'type': 'text', 'text': json.dumps({
                            'review_requirements': group,
                            'render': view}, ensure_ascii=False)}]},
                ], max_tokens=min(4096, 400 + 180 * len(group) +
                                  140 * sum(len(item['attributes']) for item in group)),
                   json_mode=True, **output_schema)
                call.update(response=data['_content'], usage=data.get('usage'),
                            finish_reason=data.get('_finish_reason'), **data['_baseline'])
                if data.get('_finish_reason') != 'stop':
                    raise QwenAPIError('appearance verification did not complete normally')
                try:
                    payload = json.loads(data['_content'])
                except json.JSONDecodeError:
                    payload = parse_json_object(data['_content'])
                if isinstance(payload, list):
                    payload = {'checks': payload}
                if not isinstance(payload, dict):
                    raise ValueError('visual_check response must be an object or checks array')
                result['checks'].extend(_check_payload(payload,
                    [item['id'] for item in group], geometry_objects, parsed, group, provenance))
            except (ValueError, OSError) as exc:
                call['error'] = str(exc)
                if isinstance(exc, QwenAPIError) and getattr(exc, 'diagnostic', None):
                    call['api_diagnostic'] = dict(exc.diagnostic)
                    call['failure_kind'] = exc.diagnostic.get('failure_kind', 'response_parse')
                    raise
                result['missing'].append('appearance verification failed: ' + str(exc))
        reported = {(check['id'], check['category'], check['role']) for check in result['checks']}
        for need in requirements:
            oid = need['id']
            if (oid, need['category'], need['role']) not in reported:
                visibility = 'unmarked' if oid in view.get('unmarked_focus_ids', []) else 'unknown'
                fallback = {
                    'id': oid, 'category': need['category'], 'role': need['role'],
                    'verdict': 'uncertain', 'reason': 'unmarked in image or not verified',
                    'visibility': visibility, 'source': 'active rendered view',
                    'inspection_status': 'uninspected',
                    'render_provenance': provenance,
                }
                result['checks'].append(normalize_visual_check(
                    fallback, need['attributes'], source_phrases=need.get('source_attributes')))
        scope = [obj for obj in candidates if obj['bbox_id'] in before]
        comparison_pool_ids, comparison_scope_bound, scope_unknown_ids = _target_size_scope(
            parsed, diagnosis, candidates)
        comparison_scope = (candidates if comparison_pool_ids is None else
                            [obj for obj in candidates
                             if obj['bbox_id'] in set(comparison_pool_ids)])
        _, _, comparison = size_candidate_evidence(
            parsed, scope, point_cloud_measurements(comparison_scope, xyz, ids),
            comparison_pool_ids=comparison_pool_ids,
            comparison_scope_bound=comparison_scope_bound, objects=geometry_objects,
            checks=diagnosis.get('checks', []))
        if scope_unknown_ids:
            comparison['scope_unknown_ids'] = scope_unknown_ids
        result['details']['size_comparison'] = comparison
        for check in result['checks']:
            check['image'] = view['image']
        result['candidate_evidence'] = [
            {'candidate_id': check['id'], 'status': {'yes': 'pass', 'no': 'fail',
             'uncertain': 'unknown'}[check['verdict']], 'reason': check['reason'],
             'image': view['image'], 'visibility': check['visibility'],
             'inspection_status': ('inspected' if check['inspection_status'] == 'inspected'
                 and all(item['inspection_status'] == 'inspected'
                         for item in check.get('attribute_checks', [])) else 'uninspected'),
             'attribute_evidence': check.get('attribute_checks', []),
             'render_provenance': provenance,
             'source_quote': next((quote for item in requirements
                                   if item.get('id') == check['id'] and item.get('role') == 'target'
                                   for quote in item.get('source_attributes', [])),
                                  parsed.get('source_query', query)),
             'target_role': 'target', 'role_binding_status': 'bound',
             'comparison_scope_ids': [check['id']],
             'comparison_scope_status': 'bound',
             'measurement_source': 'active rendered visual check',
             'explicit_fail_criterion': (check.get('reason')
                                         if check['verdict'] == 'no' else None)}
             for check in result['checks'] if check['role'] == 'target']
        result['reference_evidence'] = [check for check in result['checks'] if check['role'] != 'target']
    # Follow-up focus must use the same exclusion contract as the actual update.
    # A raw visual verdict is not itself an allowed removal.
    generation = parsed.get('candidate_generation', {})
    fallback_unbound = (generation.get('source') == 'broad_fallback' and
                        generation.get('status') != 'bound')
    result['candidate_evidence'] = [candidate_removal_gate(
        item, target_role='target', reference_role=item.get('reference_role'),
        role_binding_status='unknown' if fallback_unbound else item.get('role_binding_status', 'unknown'),
        comparison_scope_status='unknown' if fallback_unbound else item.get('comparison_scope_status', 'unknown'))
        for item in result['candidate_evidence']]
    records = result['candidate_evidence']
    checked = [item['candidate_id'] for item in records if item.get('status') != 'not_applicable'
               and item.get('inspection_status') != 'uninspected']
    unknown = [item['candidate_id'] for item in records if item.get('status') == 'unknown']
    uninspected = [oid for oid in before if oid not in checked]
    result['coverage'] = {'checked_ids': checked, 'unknown_ids': unknown,
                          'uninspected_ids': uninspected}
    if checked or (dimension == 'appearance' and records):
        result['operation_status'] = 'partial' if unknown or uninspected or result['missing'] else 'evaluated'
    if uninspected:
        result['missing'].append('candidates without a verdict in this operation: ' + str(uninspected))
    if unknown:
        result['missing'].append('candidate evidence remains unknown: ' + str(unknown))
    live_targets = [oid for oid in before if not any(
        item['candidate_id'] == oid and item.get('removal_allowed')
        for item in records)]
    result['next_evidence'] = ([{'action': 'evidence', 'dimension': 'appearance',
                                'focus_ids': uninspected[:6],
                                'azimuth_deg': action['azimuth_deg'],
                                'elevation_deg': action['elevation_deg'],
                                'context': action['context'],
                                'reason': 'inspect remaining target candidates'}]
                               if dimension == 'appearance' and uninspected else [])
    if dimension == 'appearance' and unknown:
        result['next_evidence'].append({
            'action': 'evidence', 'dimension': 'appearance', 'focus_ids': unknown[:6],
            'azimuth_deg': (float(action['azimuth_deg']) + 90.) % 360.,
            'elevation_deg': action['elevation_deg'], 'context': action['context'],
            'reason': 'use a targeted alternate view for occluded, low-detail, or uncertain attributes'})
    if dimension == 'appearance':
        reference_checks = result.get('reference_evidence', [])
        checked_references = {check['id'] for check in reference_checks
                              if check['inspection_status'] == 'inspected'
                              and all(item['inspection_status'] == 'inspected'
                                      for item in check.get('attribute_checks', []))}
        known_ids = {obj['bbox_id'] for obj in geometry_objects}
        needed_references = {need['id'] for need in diagnosis.get('visual_needs', [])
                             if need.get('role') != 'target' and type(need.get('id')) is int
                             and need['id'] in known_ids}
        unknown_references = sorted({check['id'] for check in reference_checks
                                     if check['verdict'] == 'uncertain'})
        uninspected_references = sorted((needed_references - checked_references) | {
            check['id'] for check in reference_checks
            if check['inspection_status'] == 'uninspected' or any(
                item['inspection_status'] == 'uninspected'
                for item in check.get('attribute_checks', []))})
        result['coverage'].update(reference_checked_ids=sorted(checked_references),
                                  reference_unknown_ids=unknown_references,
                                  reference_uninspected_ids=uninspected_references)
        result['attribute_coverage'] = [
            {'id': check['id'], 'role': check['role'], 'category': check['category'],
             **attribute, 'image': check['image'],
             'render_provenance': check['render_provenance']}
            for check in result['checks'] for attribute in check.get('attribute_checks', [])]
        result['unknown_attributes'] = sorted({item['attribute']
            for item in result['attribute_coverage'] if item.get('verdict') == 'uncertain'})
        pending_references = sorted(set(unknown_references + uninspected_references))
        if pending_references:
            result['operation_status'] = 'partial'
            result['missing'].append('reference appearance remains unknown or uninspected: ' + str(pending_references))
            if live_targets:
                result['next_evidence'].append({
                    'action': 'evidence', 'dimension': 'appearance',
                    'focus_ids': list(dict.fromkeys(live_targets[:1] + pending_references))[:6],
                    'azimuth_deg': ((float(action['azimuth_deg']) + 90.) % 360.
                                    if unknown_references else action['azimuth_deg']),
                    'elevation_deg': action['elevation_deg'], 'context': action['context'],
                    'reason': 'inspect bound reference attributes in a separate focused group'})
    if enabled() and dimension in ('size', 'position', 'distance') and before:
        details = result['details']
        if details.get('visual_inspection', {}).get('status') == 'unknown':
            result['next_evidence'].append({'action': 'evidence', 'dimension': dimension,
                'focus_ids': before[:6], 'inspect': True, 'azimuth_deg': 180, 'elevation_deg': 25,
                'reason': 'inspect unresolved observed geometry from one targeted alternate common view'})
        if dimension == 'size' and any(c.get('scope_status') != 'bound' for c in details.get('size_constraints', [])):
            result['next_evidence'].append({'action': 'evidence', 'dimension': 'position',
                'focus_ids': before[:4], 'inspect': True,
                'reason': 'bind the quoted comparison subset relation before ranking size'})
    return result


def evidence_state(diagnosis):
    """Expose current audit needs without calling inspected dimensions solved."""
    return {
        'status': diagnosis.get('status'),
        'remaining_candidate_ids': list(diagnosis.get('candidate_ids', [])),
        'missing': list(diagnosis.get('missing', [])),
        'pending_size_conditions': deepcopy(diagnosis.get('pending_size_conditions', [])),
        'visual_needs': deepcopy(diagnosis.get('visual_needs', [])),
    }


def singleton_prediction(predicted_id, remaining, objects):
    """At normal completion, select one surviving real target without certification."""
    if predicted_id is not None or len(remaining) != 1:
        return predicted_id, False
    oid = remaining[0]
    if type(oid) is int and any(obj['bbox_id'] == oid for obj in objects):
        return oid, True
    return None, False


def active_ground(query, parsed, objects, geometry_objects, candidates, xyz, rgb, ids,
                  out, calls, trace, limit, trusted_categories, resume=None):
    backend = react_backend()
    room_center = (xyz.min(0) + xyz.max(0)) / 2
    diagnosis = diagnose(parsed, geometry_objects, candidates, room_center,
                         trusted_categories=trusted_categories,
                         apply_size_constraints=False, query=query)
    initial_candidates = sorted(obj['bbox_id'] for obj in candidates)
    # Size comparisons are scoped once at category binding time.  Other
    # evidence dimensions may remove candidates later, but must not redefine
    # what "largest" or "tallest" meant in the query.
    diagnosis['size_comparison_pool_ids'] = initial_candidates[:]
    category_candidates = initial_candidates[:]
    parsed.setdefault('_v4_budget', {'size': 0, 'relations': 0, 'review': 0})
    inventory = evidence_inventory(query, parsed) if enabled('inventory') else None
    deletions = deletion_events(initial_candidates, diagnosis['candidate_ids'], 'initial',
        diagnosis.get('candidate_removal_audit', []), deepcopy(diagnosis), 0) if enabled() else []
    for event in deletions:
        event['before_state'] = {'status': 'category_bound_before_initial_pruning',
                                 'candidate_ids': initial_candidates[:], 'missing': [], 'hypotheses': []}
    restored_ids, invalidated_dimensions = set(), set()
    backtrack_used = False
    reviews = []
    tentative_id, tentative_source = None, None
    observation = geometry_observation(diagnosis, category_candidates)
    history, images, used = [], [], set()
    checks, used_dimensions, computed_filters = [], set(), []
    controller_feedback = None
    trace.update(
        policy='category first; model selects the next evidence dimension; no fixed order',
        interface='evidence_by_dimension', evidence_budget=limit,
        decision_backend=backend, planner_input='text_only', visual_backend=api_provider(),
        internal_limits={'renders_per_evidence': 6,
                         'visual_call_policy': 'one per category/role group for shared IDs; no retries'},
        internal_usage={'compute': 0, 'render': 0, 'visual_check': 0},
        tool_budget=limit, category_candidates=category_candidates,
        initial_diagnosis=deepcopy(diagnosis), steps=history,
        v4_settings=v4_settings(), evidence_inventory=inventory, deletion_provenance=deletions,
        backtracks=[], final_reviews=reviews,
    )
    target_table = [{'bbox_id': obj['bbox_id'], 'category': obj['target']}
                    for obj in candidates]
    target_ids = set(category_candidates)
    references = [{'bbox_id': obj['bbox_id'], 'category': obj['target']}
                  for obj in geometry_objects if obj['bbox_id'] not in target_ids]
    if v6_enabled():
        from query_graph import local_graph
        graph = local_graph(parsed, geometry_objects, candidates, diagnosis)
        relevant_ids = {n['bbox_id'] for n in graph['nodes']}
        references = [r for r in references if r['bbox_id'] in relevant_ids]
        trace['query_structure'] = parsed['query_structure']
        trace['query_graph'] = graph
    attempts = 0
    tool_count = 0
    max_attempts = max(limit * 2 + 3, limit + 3)
    if enabled('backtrack'):
        max_attempts += 3
    original_limit = limit
    trace.update(action_attempt_budget=max_attempts, action_attempts=0,
                 successful_tools=0, rejected_actions=0)
    data = {}

    current_evidence = {'measurements': [], 'size_constraints': [], 'observations': [], 'relations': []}
    if resume is not None and v5_enabled():
        diagnosis = resume['diagnosis']
        trace.update(resume['active_trace'])
        history = trace['steps']
        deletions, reviews = trace['deletion_provenance'], trace['final_reviews']
        current_evidence = resume['current_evidence']
        runtime = resume['runtime']
        checks, computed_filters = runtime['checks'], runtime['computed_filters']
        used = {tuple(signature) for signature in runtime['used']}
        used_dimensions = set(runtime['used_dimensions'])
        attempts, tool_count, limit = runtime['attempts'], runtime['tool_count'], runtime['limit']
        backtrack_used = bool(trace['backtracks'])
        restored_ids = set(runtime['restored_ids'])
        invalidated_dimensions = set(runtime['invalidated_dimensions'])
        images = resume['images']
        data = runtime['data']
        observation = runtime.get('observation', observation)
        controller_feedback = runtime.get('controller_feedback')
        tentative_id, tentative_source = runtime['tentative_id'], runtime['tentative_source']

    def checkpoint(stage, action=None):
        if not v5_enabled():
            return
        from v5_runtime import save_checkpoint
        save_checkpoint(out / 'pending_inference.json', {
            'stage': stage, 'action': action, 'checkpoint_identity': trace.get('checkpoint_identity'),
            'parsed_query': parsed, 'diagnosis': diagnosis, 'active_trace': trace,
            'current_evidence': current_evidence, 'calls': calls, 'images': images,
            'runtime': {'checks': checks, 'computed_filters': computed_filters, 'used': list(used),
                'used_dimensions': list(used_dimensions), 'attempts': attempts, 'tool_count': tool_count,
                'limit': limit, 'restored_ids': list(restored_ids), 'invalidated_dimensions': list(invalidated_dimensions),
                'data': data, 'tentative_id': tentative_id, 'tentative_source': tentative_source,
                'observation': observation, 'controller_feedback': controller_feedback}})

    def recover(problem):
        nonlocal diagnosis, observation, backtrack_used, limit, controller_feedback
        affected = backtrack_events(deletions, problem, backtrack_used)
        if not enabled('backtrack') or not affected or attempts >= max_attempts - 1:
            return False
        backtrack_used = True
        if v5_enabled():
            (out / 'pending_review.json').unlink(missing_ok=True)
        before = deepcopy(diagnosis)
        restored = {e['candidate_id'] for e in affected}
        dimensions = {e['dimension'] for e in affected}
        for event in affected:
            event['invalidated'] = True
        restored_ids.update(restored)
        invalidated_dimensions.update(dimensions)
        quotes = {q for e in affected for q in e['source_quotes'] if q}
        parsed['_v4_invalidated_conditions'] = sorted(set(parsed.get('_v4_invalidated_conditions', [])) | quotes)
        if v5_enabled():
            parsed['_size_scope_bindings'] = {k: v for k, v in parsed.get('_size_scope_bindings', {}).items()
                if not any(q in (json.loads(k)[4] or '') or q in (json.loads(k)[5] or '') for q in quotes)
                and not (k in parsed.get('_v4_binding_changes', {}) and
                         parsed['_v4_binding_changes'][k].get('source_quote') in quotes)}
            parsed['_v4_metric_choices'] = {k: v for k, v in parsed.get('_v4_metric_choices', {}).items()
                if v.get('source_quote') not in quotes}
            parsed['_v5_visual_cache'] = {k: v for k, v in parsed.get('_v5_visual_cache', {}).items()
                if not any(c.get('original_quote') in quotes or set(c.get('scope_relation_quotes', [])) & quotes
                    for c in json.loads(k).get('constraints', []))
                and not any(r.get('candidate_id') in restored and r.get('source_quote') in quotes
                    for r in json.loads(k).get('relations', []))}
        else:
            parsed['_size_scope_bindings'] = {}
            parsed['_v4_metric_choices'] = {}
        if not v5_enabled():
            parsed['_v4_binding_changes'] = {}
        parsed['_v4_relax_once'] = False
        if 'appearance' in dimensions:
            if v5_enabled():
                for check in checks:
                    if check.get('id') not in restored:
                        continue
                    for attribute in check.get('attribute_checks', []):
                        if set(attribute.get('source_quotes', [])) & quotes or attribute.get('attribute') in quotes:
                            attribute.update(verdict='uncertain', reason='condition invalidated by supported backtrack')
                    verdicts = [check.get('category_verdict', 'uncertain')] + [
                        a.get('verdict', 'uncertain') for a in check.get('attribute_checks', [])]
                    check['verdict'] = 'no' if 'no' in verdicts else 'yes' if all(v == 'yes' for v in verdicts) else 'uncertain'
            else:
                checks[:] = [c for c in checks if c.get('id') not in restored]
        parsed['_v4_relation_evidence'] = [r for r in parsed.get('_v4_relation_evidence', [])
                                          if r.get('source_quote') not in quotes or
                                          (v5_enabled() and r.get('candidate_id') not in restored)]
        if v5_enabled():
            for computed in computed_filters:
                affected_specs = [c for c in computed.get('size_constraints', [])
                    if c.get('original_quote') in quotes or set(c.get('scope_relation_quotes', [])) & quotes]
                dependent_quotes = {c['original_quote'] for c in affected_specs}
                computed['size_constraints'] = [c for c in computed.get('size_constraints', []) if c not in affected_specs]
                for proof in computed['evidence']:
                    attrs = proof.get('attribute_evidence')
                    if attrs is not None:
                        proof['attribute_evidence'] = [e for e in attrs if e.get('original_quote') not in quotes | dependent_quotes]
                        states = [e.get('status') for e in proof['attribute_evidence']]
                        proof['status'] = 'fail' if 'fail' in states else 'pass' if states and all(s == 'pass' for s in states) else 'unknown'
                    if proof.get('source_quote') in quotes | dependent_quotes:
                        proof.update(status='unknown', removal_allowed=False, role_binding_status='unknown')
            current_evidence['size_constraints'] = [c for c in current_evidence['size_constraints']
                if c.get('original_quote') not in quotes and not set(c.get('scope_relation_quotes', [])) & quotes]
            current_evidence['relations'] = [r for r in current_evidence['relations']
                if r.get('source_quote') not in quotes or r.get('candidate_id') not in restored]
        else:
            computed_filters[:] = [f for f in computed_filters if f['dimension'] not in dimensions and f['dimension'] != 'size']
        diagnosis = diagnose(parsed, geometry_objects, candidates, room_center, checks,
            trusted_categories=trusted_categories, apply_size_constraints=False, query=query)
        diagnosis['size_comparison_pool_ids'] = initial_candidates[:]
        # Recomputing the audit must not revive unrelated past exclusions.
        permitted = set(before['candidate_ids']) | restored
        diagnosis['candidate_ids'] = sorted((set(diagnosis['candidate_ids']) | restored) & permitted)
        for computed in computed_filters:
            apply_computed_evidence(diagnosis, computed['evidence'], computed['dimension'], record=False,
                                    size_constraints=computed.get('size_constraints', []) if v5_enabled() else ())
        diagnosis['missing'].append('backtracked deletion requires fresh evidence: ' + ', '.join(map(str, sorted(restored))))
        diagnosis['status'] = 'missing'
        limit = original_limit + 1  # one declared reserved operation; never unbounded replenishment
        trace['tool_budget'] = limit
        event = {'problem': problem, 'restored_ids': sorted(restored),
                 'invalidated_event_ids': [e['event_id'] for e in affected],
                 'invalidated_dimensions': sorted(dimensions if v5_enabled() else dimensions | {'size'}),
                 'before': before, 'after': deepcopy(diagnosis), 'opportunity_used': 1}
        trace['backtracks'].append(event)
        used.clear()
        observation = {'tool': 'backtrack', **event}
        controller_feedback = 'One supported backtrack restored affected alternatives. Obtain fresh evidence; no second backtrack is available.'
        history.append({'observation': observation, 'remaining_candidate_ids': diagnosis['candidate_ids'][:]})
        return True

    def finish_result(action, stop_reason, oid=None):
        nonlocal tentative_id, tentative_source
        remaining = diagnosis.get('candidate_ids', [])
        status = diagnosis.get('status') if oid is not None else 'missing'
        if not remaining:
            status = ('missing' if diagnosis.get('status') == 'missing'
                      and diagnosis.get('missing') else 'conflict')
        elif oid is not None and len(remaining) > 1:
            status = 'missing'
        elif oid is not None and diagnosis.get('pending_size_conditions'):
            status = 'missing'
        prediction_source = 'model' if oid is not None else 'abstained'
        oid, fallback = singleton_prediction(oid, remaining, objects)
        if fallback:
            prediction_source = 'singleton_fallback'
        if v7_enabled():
            from v7_evidence import supported_choice
            choice = supported_choice(query, parsed, geometry_objects, candidates, diagnosis, checks)
            trace['v7_decision'] = choice
            # A planner's unsupported budget-end ID cannot bypass the evidence gate.
            if len(remaining) > 1 or oid in choice['contradicted_ids']:
                oid = choice['selected_id']
                prediction_source = 'v7_supported_tentative' if oid is not None else 'abstained'
                fallback = True
                status = 'missing'
        elif oid is None and tentative_id in remaining:
            oid, prediction_source, fallback = tentative_id, tentative_source, True
        if v74_enabled() and oid is None:
            from v74_grounding import ground_scene
            checkpoint('scene_grounding', {'excluded_ids': choice['contradicted_ids']})
            scene_result = ground_scene(query, parsed, geometry_objects, candidates, xyz, rgb, ids,
                out, calls, diagnosis, trace, excluded_ids=choice['contradicted_ids'])
            if scene_result is not None:
                scene_result['images'] = list(dict.fromkeys(images + scene_result['images']))
                return scene_result
        if v5_enabled() and oid is not None:
            from v5_runtime import save_checkpoint
            save_checkpoint(out / 'pending_review.json', {'query': query, 'predicted_id': oid,
                'implementation_version': v4_settings()['version'], 'prediction_source': prediction_source,
                'accepted': not fallback and status == 'resolved', 'decision_status': status,
                'diagnosis': diagnosis, 'active_trace': trace, 'parsed_query': parsed,
                'current_evidence': current_evidence, 'calls': calls, 'images': images,
                'checkpoint_identity': trace.get('checkpoint_identity'), 'response': data.get('_content', ''),
                'runtime': {'checks': checks, 'computed_filters': computed_filters, 'used': list(used),
                    'used_dimensions': list(used_dimensions), 'attempts': attempts, 'tool_count': tool_count,
                    'limit': limit, 'restored_ids': list(restored_ids), 'invalidated_dimensions': list(invalidated_dimensions),
                    'data': data, 'tentative_id': tentative_id, 'tentative_source': tentative_source},
                'action': action, 'stop_reason': stop_reason})
        review = {'status': 'not_applicable', 'reason': 'no prediction'}
        if oid is not None and not enabled('final_review'):
            review = {'status': 'disabled', 'reason': 'final-review ablation'}
        if oid is not None and enabled('final_review'):
            budget = parsed['_v4_budget']
            review_limit = v4_settings()['final_review_calls_limit']
            if v5_enabled('call_economy'):
                review_limit = min(review_limit, 1 + int(backtrack_used))
            if budget['review'] < review_limit:
                references_to_review = sorted({ref for h in diagnosis.get('hypotheses', [])
                                               for ref in h.get('references', {}).values() if type(ref) is int})
                alternatives = [e['candidate_id'] for e in reversed(deletions) if e['candidate_id'] != oid]
                alternatives += [i for i in remaining if i != oid]
                focus = list(dict.fromkeys([oid] + alternatives[:3] + references_to_review[:2] + alternatives))[:6]
                view = boxed_view(geometry_objects, focus, xyz, rgb, ids, out,
                                  f'v4_final_review_{budget["review"] + 1}',
                                  current_evidence['measurements'] if v5_enabled() else ())
                images.append(view['image'])
                budget['review'] += 1
                trace['internal_usage']['render'] += 1
                trace['internal_usage']['visual_check'] += 1
                payload = {'selected_id': oid, 'prediction_source': prediction_source,
                    'missing': diagnosis.get('missing', []), 'evidence_inventory': inventory, 'deletions': deletions}
                if v5_enabled():
                    payload['measurements'] = [m for m in current_evidence['measurements'] if m['id'] in focus]
                    payload['size_constraints'] = [{**c, 'candidate_evidence': [e for e in c['candidate_evidence']
                        if e['candidate_id'] in focus]} for c in current_evidence['size_constraints']]
                    payload['relations'] = [r for r in current_evidence.get('relations', []) if r['candidate_id'] in focus]
                    payload['deletions'] = [{k: event.get(k) for k in
                        ('event_id', 'candidate_id', 'dimension', 'source_quotes', 'invalidated', 'evidence')}
                        for event in deletions if event['candidate_id'] in focus]
                try:
                    review = visual_report('final_prediction_review', query, view, payload, calls, chat_fn=chat)
                except QwenAPIError as exc:
                    if not v5_enabled():
                        raise
                    review = {'status': 'review_failed', 'reason': str(exc), 'supported': False,
                              'api_diagnostic': exc.diagnostic, 'findings': []}
                    budget['review'] -= 1  # A transport failure has not completed this review opportunity.
                    from v5_runtime import save_checkpoint, sample_recovery
                    pending = json.loads((out / 'pending_review.json').read_text())
                    pending.update(calls=calls, parsed_query=parsed, recovery_state=sample_recovery.get())
                    save_checkpoint(out / 'pending_review.json', pending)
                review.update(predicted_id=oid, prediction_source=prediction_source,
                              before_accepted=not fallback and status == 'resolved')
                reviews.append(review)
                findings = review.get('findings', [])
                if v5_enabled():
                    for finding in findings:
                        if finding['finding'] == 'confirms_exclusion':
                            event = next(e for e in deletions if e['event_id'] == finding['event_id'])
                            event.setdefault('confirmations', []).append(finding)
                    if review.get('selected_contradicted'):
                        contradicted_quotes = {f['source_quote'] for f in findings
                                              if f['finding'] == 'contradicts_selected'}
                        parsed['_v4_invalidated_conditions'] = sorted(set(parsed.get('_v4_invalidated_conditions', [])) |
                                                                    contradicted_quotes)
                        for constraint in current_evidence['size_constraints']:
                            if constraint.get('original_quote') not in contradicted_quotes:
                                continue
                            constraint.update(status='unknown', selected_ids=[], tentative_selected_ids=[],
                                reason='selected-target condition contradicted by final review')
                            for proof in constraint.get('candidate_evidence', []):
                                if proof['candidate_id'] == oid:
                                    proof.update(status='unknown', invalidated_by='final_review')
                        diagnosis['missing'].append('final review refuted a quoted selected-target condition')
                        diagnosis['status'] = status = 'missing'
                if v7_enabled() and review.get('selected_contradicted'):
                    oid = None
                    prediction_source = 'v7_review_contradiction'
                    fallback = True
                if (review.get('restore_supported') if v5_enabled() else review['supported']):
                    problem = {'supported': True,
                        'candidate_ids': sorted({i for c in review['contradictions'] for i in c['candidate_ids']}),
                        'source_quotes': [c['source_quote'] for c in review['contradictions']],
                        'evidence': review['contradictions'], 'trigger': 'final_review'}
                    problem['findings'] = findings
                    if recover(problem):
                        tentative_id, tentative_source = None, None
                        return None
                    diagnosis['missing'].append('final review found an unresolved concrete contradiction')
                    diagnosis['status'] = status = 'missing'
            else:
                review = {'status': 'unknown', 'reason': 'final-review budget exhausted'}
        history.append({'action': action, 'remaining_candidate_ids': list(remaining),
                        'evidence_state': evidence_state(diagnosis)})
        trace.update(stop_reason=stop_reason, final_candidate_ids=list(remaining),
                     final_checks=checks, final_evidence_state=evidence_state(diagnosis))
        obj = next((obj for obj in objects if obj['bbox_id'] == oid), None)
        final_data = data.get('_target_decision') or data
        return {
            'query': query, 'predicted_id': oid,
            'implementation_version': v4_settings()['version'],
            'prediction_source': prediction_source,
            'bbox_3d': obj['bbox_3d'] if obj else None,
            'response': final_data.get('_content', ''), 'usage': final_data.get('usage'),
            'model': final_data.get('_baseline', {}).get('model'),
            'latency_s': final_data.get('_baseline', {}).get('latency_s'),
            'finish_reason': final_data.get('_finish_reason'), 'images': images,
            'diagnosis': diagnosis, 'decision_status': status,
            'accepted': not fallback and oid is not None and status == 'resolved',
            'final_review': review, 'backtrack_count': len(trace['backtracks']),
            'internal_budget_usage': dict(parsed['_v4_budget']),
            'v4_api_calls': sum(c.get('stage') in ('active_size_visual_check', 'active_relations_visual_check',
                                                  'final_prediction_review') for c in calls),
            'v4_latency_s': sum(c.get('latency_s', 0) or 0 for c in calls if c.get('stage') in
                                ('active_size_visual_check', 'active_relations_visual_check', 'final_prediction_review')),
            'review_triggered': any(r.get('restore_supported') if v5_enabled() else r.get('supported') for r in reviews),
            'current_evidence': current_evidence if v5_enabled() else None,
            'decision_reasons': [action['reason']],
            'diagnosis_policy': 'active_model_selected_evidence', 'active_trace': trace,
        }

    if v74_enabled() and (resume is None or resume.get('stage') == 'scene_grounding'):
        from v74_grounding import ground_scene, needs_scene_reasoning
        if needs_scene_reasoning(diagnosis) or (resume and resume.get('stage') == 'scene_grounding'):
            excluded = (resume.get('action') or {}).get('excluded_ids', []) if resume else []
            checkpoint('scene_grounding', {'excluded_ids': excluded})
            result = ground_scene(query, parsed, geometry_objects, candidates, xyz, rgb, ids,
                                  out, calls, diagnosis, trace, excluded_ids=excluded)
            if result is not None:
                return result
    if resume is not None and v5_enabled() and resume.get('stage', 'review') == 'review':
        result = finish_result(resume['action'], resume['stop_reason'], resume['predicted_id'])
        if result is not None:
            return result
    resume_action = resume.get('action') if resume and resume.get('stage') == 'evidence' else None
    while attempts < max_attempts:
        remaining = diagnosis.get('candidate_ids') or []
        if resume_action is not None:
            action, resume_action = resume_action, None
        else:
            checkpoint('planner')
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
                'available_dimensions': list(DIMENSIONS),
                'evidence_capabilities': EVIDENCE_CAPABILITIES,
                'tools_remaining': max(0, limit - tool_count),
                'evidence_inventory': inventory, 'backtrack_available': enabled('backtrack') and not backtrack_used,
                'deletion_events': deletions if v5_enabled() else [],
                'reserved_operations': 1 if enabled('backtrack') and not backtrack_used else 0,
                'internal_visual_budget': dict(parsed['_v4_budget']),
            }
            instructions = INSTRUCTIONS
            if v7_enabled():
                from v7_evidence import supported_choice
                state['v7_preview'] = supported_choice(query, parsed, geometry_objects, candidates, diagnosis, checks)
                instructions += ('\nV7 preview is tentative box/context evidence, not completed predicates. '
                    'Use it to schedule useful checks. Negative color attributes are executed as stated; '
                    'unknown visibility never proves absence. Multiple references agreeing on a preference '
                    'support a tentative answer. Inspect contradicted or unknown conditions when informative. '
                    'Prefer an unchecked useful dimension over a third attempt at an unchanged unknown.')
            if v6_enabled():
                from query_graph import local_graph
                state['query_graph'] = local_graph(parsed, geometry_objects, candidates, diagnosis)
                instructions += ('\nUse query_structure and query_graph to inspect source quotes, modifier roles '
                    'and every possible reference binding. The graph is context, not new evidence. '
                    'Unknown or exclusion conditions are unresolved obligations; never execute them as '
                    'positive predicates or count them as solved. Size comparison pools remain frozen.')
            if enabled():
                instructions = instructions.replace('Size selects its own source-bound pools. Supply only action, dimension and reason.',
                    'Size selects its own source-bound pools; optional inspection parameters are described below.')
                instructions += ('\nUse evidence_inventory to resolve quoted dependencies; do not execute a fixed schedule. '
                    'Size, position and distance support optional inspect=true, focus_ids (1-6 known IDs including a target), '
                    'azimuth_deg and elevation_deg for one common-camera internal inspection. '
                    'Request another focused view or reference-binding evidence when useful. '
                    'Metric choices based on structure remain tentative; never infer hidden geometry. '
                    'Backtrack opportunity and internal visual budgets are shared and bounded.')
                state['evidence_capabilities'] = {**EVIDENCE_CAPABILITIES,
                    'size': 'Source-bound metric quality, completeness risk, appropriate proxy interpretation; optional common-camera VLM inspection, tentative outputs preserve unknown constraints.',
                    'position': 'Existing geometry and quoted support/attachment/corner tasks; uniquely bound anchors and visible confirmation required.',
                    'distance': 'Nearest/farthest as before; near uses an explicit box support band plus visible confirmation, with an unknown intermediate band. Near never means nearest.'}
            if v7_enabled():
                instructions = instructions.replace(
                    'Unknown or exclusion conditions are unresolved obligations; never execute them as positive predicates or count them as solved.',
                    'Source-bound negative colors have their own appearance checks; unsupported negations remain unresolved. Unknown is never evidence of absence.')
            if controller_feedback:
                instructions += '\nController correction: ' + controller_feedback
            if v73_enabled():
                instructions += ('\nPlanner field contract: appearance always renders, so set inspect=null. '
                    'For size/position/distance, context=null; internal inspection uses a fixed comparison panel. '
                    'For viewpoint, set inspect, focus_ids, azimuth_deg, elevation_deg and context to null. '
                    'For evidence set predicted_id=null. For finish/stop set dimension and all inspection fields to null.')
            if tool_count >= limit:
                instructions += '\nThe tool budget is exhausted. Return action=finish or action=stop now.'
            instructions += ('\nOUTPUT FORMAT (STRICT): Respond with ONLY one raw JSON object. '
                'No markdown fences, no prose, no text outside the JSON. '
                'Start with { and end with }. '
                'Example: {"action":"evidence","dimension":"size","reason":"measure the bound comparison pools"}')
            state = text_state(state)
            context_audit = None
            if v5_enabled():
                from react_decision import planner_state
                state, context_audit = planner_state(state, instruction_bytes=len(instructions.encode()))
            call = {'stage': 'active_decision', 'step': attempts + 1,
                    'backend': backend, 'role': 'planner',
                    'response_format': {'type': 'json_object'}}
            calls.append(call)
            if context_audit:
                call['context_audit'] = context_audit
            output_schema = schema_kwargs('planner_action', planner_schema(
                [obj['bbox_id'] for obj in geometry_objects], remaining or category_candidates))
            if output_schema and backend != 'jev':
                call['response_format'] = {'type': 'json_schema', 'json_schema': output_schema['response_schema']}
            trace['action_attempts'] += 1
            try:
                if backend == 'jev':
                    data = jev_action(state, instructions, call, calls)
                else:
                    data = chat([{'role': 'system', 'content': instructions},
                                 {'role': 'user', 'content': [{'type': 'text', 'text':
                                     json.dumps(state, ensure_ascii=False, separators=(',', ':'))}]}], max_tokens=900, json_mode=True,
                                **output_schema)
                call.update(response=data['_content'], usage=data.get('usage'),
                            finish_reason=data.get('_finish_reason'), **data['_baseline'])
                if data.get('_finish_reason') == 'length':
                    call['error'] = 'active decision was truncated; no action was executed'
                    observation = {'tool': 'planner', 'error': call['error'],
                                   'no_evidence_acquired': True}
                    history.append({'observation': observation,
                                    'remaining_candidate_ids': list(remaining)})
                    controller_feedback = (
                        'The previous response was truncated and acquired no evidence. '
                        'Return one complete JSON action with reason within 40 words.'
                    )
                    trace['rejected_actions'] += 1
                    attempts += 1
                    continue
                if data.get('_finish_reason') != 'stop':
                    raise QwenAPIError('active decision did not complete normally')
                try:
                    action = parse_json_object(data['_content'])
                    if output_schema and backend != 'jev':
                        action = {key: value for key, value in action.items() if value is not None}
                    if not isinstance(action.get('reason'), str) or not action['reason'].strip():
                        raise QwenAPIError('active decision requires a brief evidence-based reason')
                except ValueError as exc:
                    call['error'] = str(exc)
                    observation = {'tool': 'planner', 'error': call['error'],
                                   'no_evidence_acquired': True}
                    history.append({'observation': observation,
                                    'remaining_candidate_ids': list(remaining)})
                    controller_feedback = (
                        'The previous response was not a valid JSON action and acquired no evidence. '
                        'Return one complete JSON action with reason within 40 words; use stop if unresolved.'
                    )
                    trace['rejected_actions'] += 1
                    attempts += 1
                    continue
            except (ValueError, OSError) as exc:
                call['error'] = str(exc)
                if getattr(exc, 'diagnostic', None):
                    call['api_diagnostic'] = dict(exc.diagnostic)
                raise
        if v7_enabled() and tool_count < limit:
            # Keep the existing budget; prevent repeated unknown geometry from
            # consuming the final operation while a quoted size is unchecked.
            has_size = any(size_conditions(parsed, name) for name in parsed.get('Target', []))
            repeats = sum(e.get('action', {}).get('dimension') == action.get('dimension') for e in history)
            if has_size and 'size' not in used_dimensions and (action.get('action') in ('stop', 'finish') or repeats >= 2):
                trace.setdefault('v7_schedule_corrections', []).append({'original_action': deepcopy(action),
                    'reason': 'unchecked quoted size before stop or third repeated dimension', 'tool_count': tool_count})
                action = {'action': 'evidence', 'dimension': 'size',
                          'reason': 'Measure the unchecked quoted size before stopping or repeating unknown evidence'}
        if action.get('action') == 'stop':
            pending_keys = {(item.get('attribute'), item.get('role', 'target'))
                            for item in diagnosis.get('pending_size_conditions', [])}
            unexecuted_size = (
                pending_keys and remaining and 'size' not in used_dimensions
                and tool_count < limit and any(
                    (item.get('word', item.get('attribute')), item.get('role', 'target')) in pending_keys
                    and item.get('scope_status') == 'bound'
                    and len(item.get('comparison_pool_ids', [])) >= 2
                    and item.get('interpretation') != 'descriptive_attribute'
                    for item in diagnosis.get('geometry', {}).get('size_constraints', [])))
            if unexecuted_size:
                observation = {'tool': 'stop', 'no_evidence_acquired': True,
                    'error': 'A bound size comparison is pending and size has not been executed. '
                             'Request {"action":"evidence","dimension":"size",'
                             '"reason":"verify pending size conditions"} before stopping.'}
                history.append({'action': action, 'observation': observation,
                                'remaining_candidate_ids': list(remaining)})
                controller_feedback = observation['error']
                trace['rejected_actions'] += 1
                attempts += 1
                continue
            if v4_settings()['dynamic_threshold'] and not parsed.get('_v4_threshold_used') and tool_count < limit:
                last_size = next((f for f in reversed(computed_filters) if f['dimension'] == 'size'), None)
                constraints = (last_size or {}).get('size_constraints', [])
                if constraints and all(c.get('scope_status') == 'bound' and not c.get('unknown_measurement_ids') and
                    not c.get('metric_ambiguity') and c.get('required_margin') is not None and
                    c['metric'] not in ('height_m', 'tabletop_height_m') for c in constraints):
                    parsed.update(_v4_threshold_used=True, _v4_relax_once=True)
                    details = compute_observation('size', query, parsed, diagnosis, candidates, geometry_objects, xyz, ids)
                    tool_count += 1
                    trace['internal_usage']['compute'] += 1
                    trace['successful_tools'] = tool_count
                    proposed = details['size_decision'].get('tentative_selected_ids', [])
                    if len(proposed) == 1:
                        tentative_id, tentative_source = proposed[0], 'dynamic_threshold_tentative'
                    history.append({'observation': details, 'trigger': 'bounded_threshold_after_stalled_stop'})
            result = finish_result(action, 'model_stop')
            if result is not None:
                return result
            attempts += 1
            continue
        if action.get('action') == 'finish':
            oid = action.get('predicted_id')
            allowed = set(remaining) or set(category_candidates)
            valid_id = type(oid) is int and oid in allowed
            hypotheses = diagnosis.get('hypotheses', [])
            shared_target = (bool(hypotheses) and
                             all(set(item.get('candidate_ids', [])) == {oid}
                                 for item in hypotheses) and
                             diagnosis.get('stable_candidate_ids') == [oid])
            special_support = candidate_hypothesis_support(diagnosis, oid)
            other_two_resolved = (
                special_support is not None and
                special_support.get('status') == 'pass' and
                remaining == [oid] and
                bool(special_support.get('applicable_hypothesis_indices')) and
                not parsed.get('diagnostic_missing') and
                not any(item.get('status') == 'fail' and not item.get('removal_allowed')
                        for item in diagnosis.get('candidate_removal_audit', [])) and
                not diagnosis.get('pending_size_conditions') and
                not diagnosis.get('visual_needs'))
            resolved = (diagnosis.get('status') == 'resolved' and remaining == [oid] and
                        shared_target and not diagnosis.get('pending_size_conditions'))
            resolved = resolved or other_two_resolved
            if valid_id and (resolved or tool_count >= limit):
                if other_two_resolved:
                    # The generic diagnosis retains alternative self-reference
                    # hypotheses; the narrow "other two" policy has already
                    # proved every applicable hypothesis for this candidate.
                    diagnosis['status'] = 'resolved'
                result = finish_result(action, 'budget_final_choice' if tool_count >= limit else 'model_finish', oid)
                if result is not None:
                    return result
                attempts += 1
                continue
            error = ('finish requires an ID from the remaining candidates; use stop if unresolved'
                     if not valid_id else
                     'Premature finish: the evidence has not resolved a unique target. '
                     'A yes for one candidate does not exclude others. Compare unchecked alternatives '
                     'or inspect missing conditions; use stop if no useful evidence can be acquired.')
            if valid_id and tool_count < limit and observation.get('next_evidence'):
                error += (' There are ' + str(limit - tool_count) +
                          ' evidence operations remaining. The last observation suggests: ' +
                          json.dumps(observation['next_evidence'], ensure_ascii=False) +
                          '. Consider these available checks before concluding that no useful evidence remains.')
            observation = {'tool': 'finish', 'no_evidence_acquired': True,
                           'error': error}
            history.append({'action': action, 'observation': observation,
                            'remaining_candidate_ids': list(remaining)})
            trace['rejected_actions'] += 1
            controller_feedback = observation['error']
            if diagnosis.get('pending_size_conditions') and 'size' not in used_dimensions:
                controller_feedback += (' Size has not been executed. Request '
                    '{"action":"evidence","dimension":"size","reason":"verify pending size conditions"}.')
            attempts += 1
            continue
        if tool_count >= limit:
            history.append({'action': action, 'observation': {
                'tool': action.get('action'), 'error': 'tool budget exhausted',
                'no_evidence_acquired': True}, 'remaining_candidate_ids': list(remaining)})
            trace['rejected_actions'] += 1
            result = finish_result(
                {'action': 'stop', 'reason': 'tool budget exhausted without a valid final decision'},
                'tool_budget_exhausted')
            if result is not None:
                return result
            attempts += 1
            continue
        try:
            request = evidence_request(action, remaining, geometry_objects)
        except ValueError as exc:
            observation = {'tool': 'evidence', 'error': str(exc), 'no_evidence_acquired': True}
            history.append({'action': action, 'observation': observation,
                            'remaining_candidate_ids': list(remaining)})
            controller_feedback = str(exc)
            if action.get('dimension') == 'size':
                controller_feedback += (' Valid size request: '
                    '{"action":"evidence","dimension":"size","reason":"measure source-bound pools"}.')
            trace['rejected_actions'] += 1
            attempts += 1
            continue
        signature = (
            json.dumps({k: v for k, v in request.items() if k != 'reason'}, sort_keys=True),
            json.dumps({'state': evidence_state(diagnosis), 'checks': list({
                            (check['id'], check['category']): {
                                key: check[key] for key in ('id', 'category', 'verdict', 'reason')
                                if key in check}
                            for check in checks}.values()),
                        'hypotheses': diagnosis.get('hypotheses', []),
                        'geometry': diagnosis.get('geometry', {})}, sort_keys=True),
        )
        if signature in used:
            history.append({
                'action': action,
                'observation': {
                    'tool': action.get('action'),
                    'error': 'identical evidence request on unchanged evidence; choose another task or stop',
                    'no_evidence_acquired': True,
                },
                'remaining_candidate_ids': diagnosis.get('candidate_ids', []),
            })
            controller_feedback = (
                'The previous action was a duplicate and acquired no evidence. '
                'Do not repeat it on unchanged evidence. Choose another evidence dimension, '
                'or change the appearance focus/view, or stop if no useful task remains.'
            )
            trace['rejected_actions'] += 1
            attempts += 1
            continue
        checkpoint('evidence', action)
        entry = {'action': request}
        if v73_enabled() and request != action:
            entry['requested_action'] = deepcopy(action)
        before = list(diagnosis.get('candidate_ids', []))
        before_state = deepcopy(diagnosis)
        try:
            dimension = request['dimension']
            observation = execute_evidence(
                request, query, parsed, diagnosis, candidates, geometry_objects,
                xyz, rgb, ids, out, tool_count + 1, calls)
            if v5_enabled():
                details = observation.get('details', {})
                if dimension == 'size':
                    if v7_enabled():
                        parsed['_v7_measurement_rows'] = details.get('comparison_metrics', [])
                    current_evidence['measurements'] = details.get('comparison_metrics', [])
                    current_evidence['size_constraints'] = details.get('size_constraints', [])
                if details.get('relation_tasks'):
                    current_evidence['relations'] = details['relation_tasks']
                if details.get('visual_inspection'):
                    current_evidence['observations'].append({'dimension': dimension,
                        'report': details['visual_inspection']})
            images.extend(observation['images'])
            for step in observation['internal_steps']:
                if not step.get('call', {}).get('skipped'):
                    trace['internal_usage'][step['tool']] += (step.get('views_rendered', 1)
                                                             if step['tool'] == 'render' else 1)
            used_dimensions.add(dimension)
            if observation['checks'] or (enabled('relations') and observation.get('details', {}).get('relation_tasks')):
                checks.extend(observation['checks'])
                diagnosis = diagnose(parsed, geometry_objects, candidates, room_center, checks,
                                     trusted_categories=trusted_categories,
                                     apply_size_constraints=False, query=query)
                diagnosis['size_comparison_pool_ids'] = initial_candidates[:]
                if restored_ids:
                    diagnosis['candidate_ids'] = sorted((set(diagnosis['candidate_ids']) | restored_ids) &
                                                        (set(before) | restored_ids))
                for computed in computed_filters:
                    if computed.get('size_measurements') and computed.get('joint_size'):
                        diagnosis = diagnose(parsed, geometry_objects, candidates, room_center, checks,
                            trusted_categories=trusted_categories, apply_size_constraints=True, query=query,
                            size_measurements=computed['size_measurements'])
                        diagnosis['size_comparison_pool_ids'] = initial_candidates[:]
                        if v5_enabled():
                            current_evidence['size_constraints'] = diagnosis['geometry'].get('size_constraints', [])
                    elif computed.get('size_measurements'):
                        scope_ids = set(diagnosis['candidate_ids'])
                        evidence, constraints, _ = size_candidate_evidence(parsed,
                            [o for o in candidates if o['bbox_id'] in scope_ids],
                            computed['size_measurements'], comparison_pool_ids=initial_candidates,
                            objects=geometry_objects, checks=checks)
                        diagnosis = apply_size_observation(diagnosis,
                            {'candidate_evidence': evidence, 'size_constraints': constraints}, record=False)
                        if v5_enabled():
                            computed['evidence'], computed['size_constraints'] = evidence, constraints
                            current_evidence['size_constraints'] = constraints
                    else:
                        apply_computed_evidence(diagnosis, computed['evidence'], computed['dimension'],
                                                record=False,
                                                size_constraints=computed.get('size_constraints', []))
            if restored_ids and 'initial' in invalidated_dimensions:
                diagnosis['missing'] = list(dict.fromkeys(diagnosis['missing'] +
                    ['initial predicate invalidated; restored candidates need new evidence']))
                diagnosis['status'] = 'missing'
            if dimension in ('position', 'distance', 'size'):
                candidate_evidence = observation['candidate_evidence']
                size_constraints = observation['details'].get('size_constraints', [])
                if dimension == 'size':
                    diagnosis = apply_size_observation(diagnosis, observation['details'])
                else:
                    apply_computed_evidence(diagnosis, candidate_evidence, dimension,
                                            size_constraints=size_constraints)
                computed_filters.append({'dimension': dimension, 'evidence': candidate_evidence,
                                         'size_constraints': size_constraints,
                                         'joint_size': observation['details'].get('joint_size_diagnosis') is not None,
                                         'size_measurements': observation['details'].get('comparison_metrics')})
            observation['candidate_update'] = {
                'before': before,
                'removed_ids': [oid for oid in before if oid not in diagnosis['candidate_ids']],
                'after': list(diagnosis['candidate_ids']),
            }
            if enabled():
                deletions.extend(deletion_events(before, diagnosis['candidate_ids'], dimension,
                    observation['candidate_evidence'], before_state, len(deletions)))
                apply_v4_relation_resolution(diagnosis, parsed)
                proposed = observation.get('details', {}).get('size_decision', {}).get('tentative_selected_ids', [])
                if dimension == 'size':
                    tentative_id, tentative_source = None, None
                if len(proposed) == 1 and proposed[0] in diagnosis['candidate_ids']:
                    tentative_id, tentative_source = proposed[0], 'size_proxy_tentative'
                if tentative_id not in diagnosis['candidate_ids']:
                    tentative_id, tentative_source = None, None
            observation['candidate_ids'] = list(diagnosis['candidate_ids'])
            observation['evidence_state'] = evidence_state(diagnosis)
            suggested_actions = []
            live_ids = diagnosis['candidate_ids']
            reference_ids = {need.get('id') for need in diagnosis.get('visual_needs', [])
                             if need.get('role') != 'target'}
            for suggestion in observation.get('next_evidence', []):
                if not live_ids:
                    break
                focus = [oid for oid in suggestion['focus_ids']
                         if oid in live_ids or oid in reference_ids]
                if not set(focus) & set(live_ids):
                    focus = live_ids[:1] + focus[:5]
                suggestion = dict(suggestion, focus_ids=focus)
                suggested_actions.append(evidence_request(suggestion, live_ids, geometry_objects))
            observation['next_evidence'] = suggested_actions
        except (ValueError, OSError) as exc:
            observation = {'tool': action.get('action'), 'error': str(exc),
                           'no_evidence_acquired': True}
            entry.update(observation=observation,
                         remaining_candidate_ids=diagnosis.get('candidate_ids', []))
            history.append(entry)
            trace['diagnosis_after_step'] = diagnosis
            if getattr(exc, 'diagnostic', None):
                raise
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
        suspect_ids = {r['id'] for r in observation.get('details', {}).get('comparison_metrics', [])
                       if r.get('observed_completeness', {}).get('status') == 'suspected_incomplete'}
        affected_size_ids = {event['candidate_id'] for event in deletions if event['dimension'] == 'size'
                             and not event['invalidated'] and any(suspect_ids.intersection(
                                 proof.get('comparison_scope_ids', [])) for proof in event['evidence'])}
        if enabled('backtrack') and not diagnosis['candidate_ids'] and observation['candidate_update']['removed_ids']:
            recover({'supported': True, 'candidate_ids': observation['candidate_update']['removed_ids'],
                     'source_quotes': [], 'trigger': 'empty_candidates',
                     'evidence': observation['candidate_evidence']})
        elif enabled('backtrack') and affected_size_ids:
            recover({'supported': True, 'candidate_ids': sorted(affected_size_ids), 'source_quotes': [],
                     'trigger': 'new_observed_completeness_contradicts_size_deletion',
                     'evidence': [r for r in observation['details']['comparison_metrics'] if r['id'] in suspect_ids]})
        elif enabled('backtrack') and parsed.get('_v4_binding_changes'):
            changes = list(parsed['_v4_binding_changes'].values())
            supported = [c for c in changes if c['status'] == 'bound' and c['old_ids'] != c['new_ids']]
            if supported:
                recover({'supported': True, 'candidate_ids': [],
                         'source_quotes': [c['source_quote'] for c in supported],
                         'trigger': 'comparison_binding_changed', 'evidence': supported})
    return finish_result(
        {'action': 'stop', 'reason': 'action-attempt budget exhausted without a valid final decision'},
        'action_attempt_budget_exhausted')
