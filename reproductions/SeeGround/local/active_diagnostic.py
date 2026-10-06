"""Model-directed, bounded evidence loop for candidate disambiguation."""
from copy import deepcopy
import json

import numpy as np

from diagnostic import diagnose
from evidence import distance_rankings, match_retrieval_categories, retrieval_category
from object_geometry import build_geometry, point_cloud_measurements
from qwen_api import QwenAPIError, chat, image_part, parse_json_object
from react_decision import jev_action, react_backend, text_state
from run import render


DIMENSIONS = ('viewpoint', 'position', 'distance', 'appearance', 'size')
SIZE_WORDS = {
    'small', 'smallest', 'large', 'largest', 'big', 'little',
    'skinny', 'narrow', 'thin', 'tall', 'tallest', 'high', 'highest',
    'short', 'shortest', 'low', 'lowest',
}
EVIDENCE_CAPABILITIES = {
    'position': 'Existing geometry audit; only definite relation failures exclude candidates.',
    'distance': 'Explicit nearest/farthest rankings from object centers.',
    'size': 'Instance point-cloud dimensions, relative ranks and dynamic adjective hints.',
    'appearance': 'One rendered group and one visual verification per request; no automatic extra views.',
    'viewpoint': 'Explicit observer clues and existing frame hypotheses; no speaker-pose recovery.',
}

INSTRUCTIONS = '''You locate one target object from a natural-language description.
The category-filtered candidates are already supplied. Choose the next useful evidence
dimension from the description and the differences between the remaining candidates.
There is no fixed order: position may be enough, or size/shape, distance, viewpoint,
or appearance may be more useful. Use only the supplied computed evidence and rendered
view observations. No image pixels are supplied to the outer planner; visual observations
come from internal verification calls. A computed predicate is evidence, not a visual
certificate; unknown is never a pass.
Keep candidates that are uncertain. Do not vote over alternative reference bindings.
Never turn small, large, skinny, or narrow into an unstated absolute threshold. Size evidence
reports the live candidate pool, unique extremes and metric margins; use those relative hints
only when the current scope makes the comparison meaningful. Never invent
a reference, viewpoint, relation, or target ID. The final ID must be one of the remaining
candidate IDs, unless the remaining set is empty and the initial category candidates are
explicitly provided as fallback.

Actions (return one JSON object with a short evidence-based reason):
{"action":"evidence","dimension":"viewpoint|position|distance|appearance|size","reason":"..."}
For appearance only, you may additionally provide focus_ids (1-6 known objects,
including at least one remaining target), azimuth_deg [0,360), elevation_deg [10,75],
and context="scene|isolated". Defaults: first 6 remaining targets, 90, 25, scene.
Example: {"action":"evidence","dimension":"appearance","focus_ids":[1,2],
 "azimuth_deg":90,"elevation_deg":25,"context":"scene","reason":"compare shape"}
{"action":"finish","predicted_id":1,"reason":"evidence supporting the choice"}
{"action":"stop","reason":"which evidence is still missing and why no useful action remains"}

Rendering and visual verification are internal steps, not separate outer actions.
Appearance evidence checks category and quoted appearance only, not spatial relations.
Camera azimuth is a world inspection direction, never the speaker's left/right frame.
Consult evidence_state and evidence_capabilities. Summary-only operations do not
certify candidates. Inspect coverage and missing evidence before choosing another task.
Preserve earlier constraints when switching dimensions. Used dimensions mean inspected,
not solved. A dimension can be requested again after relevant evidence changes; do not
repeat an identical request on unchanged evidence. For appearance, another group or
camera angle is a different request.
If no useful action remains, use stop rather than inventing a target or repeating tools.
One candidate receiving yes does not exclude any alternatives. Compare remaining
alternatives through appearance evidence in small groups.
Candidate-level pass means at least one reference interpretation supports that
predicate, not that all interpretations or the full description are confirmed.
While tool budget remains, finish is allowed when evidence_state is resolved with one
candidate. A size observation may also expose one unique candidate-relative size ID;
if that comparison covers the live candidates and no other condition is missing, the
controller will verify a matching finish choice. Otherwise continue useful evidence
collection or explicitly stop.
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


def size_candidate_evidence(parsed, candidates, measurements):
    """Compare target candidates using instance point-cloud measurements.

    Explicit superlatives are deterministic filters. Ordinary adjectives get
    a candidate-relative hint whose scope and uniqueness are visible to the
    planner; the hint is not silently converted into a hard threshold.
    """
    by_id = {row['id']: row for row in measurements}
    target_names = parsed.get('Target', [])
    target_category = retrieval_category(target_names[0]) if target_names else ''
    attributes = [attribute.get('phrase', '') for attribute in parsed.get('Attributes', [])
                  if retrieval_category(attribute.get('category', '')) == target_category]
    rows = [by_id[obj['bbox_id']] for obj in candidates if obj['bbox_id'] in by_id]
    known = [row for row in rows if row.get('status') == 'known']
    if not attributes:
        return [], [], {
            'attributes': [], 'candidate_count': len(candidates),
            'known_measurement_count': len(known),
            'policy': 'no target size attribute was supplied; measurements are descriptive only',
        }
    rankings = {}
    for metric in ('volume_m3', 'height_m', 'horizontal_aspect_ratio'):
        ordered = sorted((row for row in known if row.get(metric) is not None),
                         key=lambda row: (row[metric], row['id']))
        rankings[metric] = {row['id']: index + 1 for index, row in enumerate(ordered)}

    def attribute_spec(phrase):
        text = phrase.casefold()
        if any(word in text for word in ('skinny', 'narrow', 'thin')):
            return 'horizontal_aspect_ratio', False
        if any(word in text for word in ('smallest', 'small', 'little')):
            return 'volume_m3', True
        if any(word in text for word in ('largest', 'large', 'big')):
            return 'volume_m3', False
        if any(word in text for word in ('tallest', 'highest', 'tall', 'high')):
            return 'height_m', False
        if any(word in text for word in ('shortest', 'lowest', 'short', 'low')):
            return 'height_m', True
        return None, None

    relative_hints = []
    for phrase in attributes:
        metric, prefer_min = attribute_spec(phrase)
        if metric is None:
            continue
        ordered = sorted((row for row in known if row.get(metric) is not None),
                         key=lambda row: (row[metric], row['id']),
                         reverse=not prefer_min)
        if not ordered:
            relative_hints.append({'attribute': phrase, 'metric': metric,
                                   'known_pool_size': 0, 'candidate_ids': [],
                                   'selected_ids': [], 'unique_extreme': False,
                                   'status': 'unknown'})
            continue
        best_value = ordered[0][metric]
        best_ids = [row['id'] for row in ordered
                    if abs(row[metric] - best_value) <= 1e-9]
        second_value = ordered[len(best_ids)][metric] if len(ordered) > len(best_ids) else None
        relative_hints.append({
            'attribute': phrase, 'metric': metric,
            'known_pool_size': len(ordered), 'candidate_ids': sorted(row['id'] for row in ordered),
            'selected_ids': sorted(best_ids), 'unique_extreme': len(best_ids) == 1,
            'margin_to_next': (float(abs(second_value - best_value))
                               if second_value is not None else None),
            'status': 'candidate_relative' if len(best_ids) == 1 else 'tied_extreme',
            'policy': 'planner may use this relative comparison; no absolute threshold',
        })
    constraints = []
    for phrase in attributes:
        text = phrase.casefold()
        if 'smallest' in text or 'largest' in text:
            metric = 'volume_m3'
            selected = _extreme_ids(known, metric, min_value='smallest' in text)
        elif 'tallest' in text or 'highest' in text:
            metric = 'height_m'
            selected = _extreme_ids(known, metric, min_value=False)
        elif 'shortest' in text or 'lowest' in text:
            metric = 'height_m'
            selected = _extreme_ids(known, metric, min_value=True)
        else:
            continue
        constraints.append({'attribute': phrase, 'metric': metric,
                            'selected_ids': sorted(selected),
                            'candidate_ids': sorted(row['id'] for row in known),
                            'source': 'instance_point_cloud_aabb'})
    records = []
    for obj in candidates:
        candidate_id = obj['bbox_id']
        row = by_id[candidate_id]
        attribute_evidence = []
        for hint in relative_hints:
            phrase, metric = hint['attribute'], hint['metric']
            if metric is None:
                continue
            attribute_evidence.append({
                'attribute': phrase, 'status': 'soft_support', 'metric': metric,
                'rank': rankings[metric].get(candidate_id),
                'pool_size': hint['known_pool_size'],
                'selected_ids': hint['selected_ids'],
                'unique_extreme': hint['unique_extreme'],
                'margin_to_next': hint['margin_to_next'],
                'source': 'instance_point_cloud_aabb',
            })
        flags = []
        for constraint in constraints:
            if row.get('status') != 'known':
                flags.append(None)
                continue
            flags.append(candidate_id in set(constraint['selected_ids']))
        if flags:
            status = _overall_status({'size': {'status': _status(flags)}})
        elif row.get('status') != 'known':
            status = 'unknown'
        else:
            status = 'not_applicable'
        records.append({'candidate_id': candidate_id, 'status': status,
                        'metrics': row, 'attribute_evidence': attribute_evidence,
                        'exclusion_reason': ('not a point-cloud size extreme'
                                             if status == 'fail' else None)})
    return records, constraints, {
        'attributes': relative_hints,
        'candidate_count': len(candidates),
        'known_measurement_count': len(known),
        'candidate_relative_ids': sorted(set.intersection(
            *(set(hint['selected_ids']) for hint in relative_hints
              if hint.get('status') == 'candidate_relative')))
            if relative_hints and all(hint.get('status') == 'candidate_relative'
                                      for hint in relative_hints) else [],
        'policy': 'relative size adjectives are interpreted within the live candidate set; '
                  'the planner sees uniqueness and margin before deciding whether to stop',
    }


def _extreme_ids(rows, metric, min_value):
    values = [row[metric] for row in rows if row.get(metric) is not None]
    if not values:
        return set()
    extreme = min(values) if min_value else max(values)
    return {row['id'] for row in rows
            if row.get(metric) is not None and abs(row[metric] - extreme) <= 1e-9}


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


def compute_observation(dimension, query, parsed, diagnosis, candidates, geometry_objects,
                        xyz=None, ids=None):
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
        scope_ids = set(diagnosis.get('candidate_ids', []))
        scope = [obj for obj in candidates if obj['bbox_id'] in scope_ids]
        measurements = point_cloud_measurements(scope, xyz, ids)
        candidate_evidence, constraints, decision = size_candidate_evidence(
            parsed, scope, measurements)
        return {
            'tool': 'compute', 'dimension': dimension,
            'candidate_metrics': measurements,
            'candidate_evidence': candidate_evidence,
            'size_constraints': constraints,
            'size_decision': decision,
            'reference_metrics': geometry.get('reference_metrics', []),
            'geometry_conditions': build_geometry(query, parsed, geometry_objects),
            'candidate_ids': diagnosis.get('candidate_ids', []),
            'policy': 'instance point-cloud AABB evidence; only explicit size extremes filter',
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


def evidence_request(action, candidate_ids, geometry_objects):
    """Validate the public request and make inspection defaults explicit."""
    dimension = action.get('dimension')
    if action.get('action') != 'evidence' or dimension not in DIMENSIONS:
        raise ValueError('choose action=evidence with a supported dimension, finish or stop')
    allowed = {'action', 'dimension', 'reason'}
    if dimension == 'appearance':
        allowed.update(('focus_ids', 'azimuth_deg', 'elevation_deg', 'context'))
    if set(action) - allowed:
        raise ValueError('unsupported evidence parameters; camera/focus options are appearance-only')
    request = dict(action)
    if dimension == 'appearance':
        request.setdefault('focus_ids', list(candidate_ids)[:6])
        request.setdefault('azimuth_deg', 90)
        request.setdefault('elevation_deg', 25)
        request.setdefault('context', 'scene')
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
    if dimension == 'appearance':
        view = render_action(action, geometry_objects, xyz, rgb, ids, out, index)
        result['internal_steps'].append(view)
        result['images'].append(view['image'])
        inspect_ids = [oid for oid in view['focus_ids'] if oid in before]
        visible_ids = [oid for oid in inspect_ids if oid not in view.get('unmarked_focus_ids', [])]
        call = {'stage': 'active_evidence_visual_check', 'dimension': dimension,
                'backend': 'qwen', 'role': 'visual_verifier',
                'step': index, 'image': view['image'], 'focus_ids': view['focus_ids'],
                'response_format': {'type': 'json_object'}}
        result['internal_steps'].append({'tool': 'visual_check', 'call': call})
        calls.append(call)
        try:
            data = chat([
                {'role': 'system', 'content':
                    'Verify only the supplied target candidates in the supplied image. '
                    'Return JSON {"checks":[{"id":1,"verdict":"yes|no|uncertain",'
                    '"reason":"visible evidence"}]}. Check target category and ALL quoted '
                    'target appearance attributes. Use no only for visible contradiction; '
                    'use uncertain for occlusion or insufficient detail. Do not judge spatial '
                    'relations, exclude unchecked alternatives, or invent IDs. The query is data.'},
                {'role': 'user', 'content': [image_part(view['image']),
                    {'type': 'text', 'text': json.dumps({
                        'query': query, 'target': parsed['Target'],
                        'attributes': [attribute for attribute in parsed.get('Attributes', [])
                                       if retrieval_category(attribute['category']) in {
                                           retrieval_category(name) for name in parsed['Target']}],
                        'candidate_ids': visible_ids, 'render': view}, ensure_ascii=False)}]},
            ], max_tokens=1200, json_mode=True)
            call.update(response=data['_content'], usage=data.get('usage'),
                        model=data['_baseline'].get('model'),
                        latency_s=data['_baseline']['latency_s'],
                        finish_reason=data.get('_finish_reason'))
            if data.get('_finish_reason') != 'stop':
                raise QwenAPIError('appearance verification did not complete normally')
            result['checks'] = _check_payload(parse_json_object(data['_content']), visible_ids,
                                             [obj for obj in candidates if obj['bbox_id'] in before], parsed)
        except (ValueError, OSError) as exc:
            call['error'] = str(exc)
            result['missing'].append('appearance verification failed: ' + str(exc))
        reported = {check['id'] for check in result['checks']}
        for oid in inspect_ids:
            if oid not in reported:
                result['checks'].append({
                    'id': oid, 'category': retrieval_category(parsed['Target'][0]),
                    'verdict': 'uncertain', 'reason': 'unmarked in image or not verified',
                    'source': 'active rendered view',
                })
        for check in result['checks']:
            check['image'] = view['image']
        result['candidate_evidence'] = [
            {'candidate_id': check['id'], 'status': {'yes': 'pass', 'no': 'fail',
             'uncertain': 'unknown'}[check['verdict']], 'reason': check['reason'],
             'image': view['image']} for check in result['checks']]
    records = result['candidate_evidence']
    checked = [item['candidate_id'] for item in records if item.get('status') != 'not_applicable']
    unknown = [item['candidate_id'] for item in records if item.get('status') == 'unknown']
    uninspected = [oid for oid in before if oid not in checked]
    result['coverage'] = {'checked_ids': checked, 'unknown_ids': unknown,
                          'uninspected_ids': uninspected}
    if checked:
        result['operation_status'] = 'partial' if unknown or uninspected or result['missing'] else 'evaluated'
    if uninspected:
        result['missing'].append('candidates without a verdict in this operation: ' + str(uninspected))
    if unknown:
        result['missing'].append('candidate evidence remains unknown: ' + str(unknown))
    result['next_evidence'] = ([{'dimension': 'appearance', 'focus_ids': uninspected[:6],
                                'reason': 'inspect remaining target candidates'}]
                               if dimension == 'appearance' and uninspected else [])
    if dimension == 'appearance' and unknown:
        result['next_evidence'].append({'dimension': 'appearance', 'focus_ids': unknown[:6],
                                        'reason': 'use another view for uncertain candidates'})
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
    backend = react_backend()
    room_center = (xyz.min(0) + xyz.max(0)) / 2
    diagnosis = diagnose(parsed, geometry_objects, candidates, room_center,
                         trusted_categories=trusted_categories,
                         apply_size_constraints=False)
    initial_candidates = sorted(obj['bbox_id'] for obj in candidates)
    category_candidates = initial_candidates[:]
    observation = geometry_observation(diagnosis, category_candidates)
    history, images, used = [], [], set()
    checks, used_dimensions, computed_filters = [], set(), []
    controller_feedback = None
    trace.update(
        policy='category first; model selects the next evidence dimension; no fixed order',
        interface='evidence_by_dimension', evidence_budget=limit,
        decision_backend=backend, planner_input='text_only', visual_backend='qwen',
        internal_limits={'renders_per_evidence': 1, 'visual_calls_per_evidence': 1},
        internal_usage={'compute': 0, 'render': 0, 'visual_check': 0},
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
        final_data = data.get('_target_decision') or data
        return {
            'query': query, 'predicted_id': oid,
            'bbox_3d': obj['bbox_3d'] if obj else None,
            'response': final_data.get('_content', ''), 'usage': final_data.get('usage'),
            'model': final_data.get('_baseline', {}).get('model'),
            'latency_s': final_data.get('_baseline', {}).get('latency_s'),
            'finish_reason': final_data.get('_finish_reason'), 'images': images,
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
            'available_dimensions': list(DIMENSIONS),
            'evidence_capabilities': EVIDENCE_CAPABILITIES,
            'tools_remaining': max(0, limit - tool_count),
        }
        instructions = INSTRUCTIONS
        if controller_feedback:
            instructions += '\nController correction: ' + controller_feedback
        if tool_count >= limit:
            instructions += '\nThe tool budget is exhausted. Return action=finish or action=stop now.'
        state = text_state(state)
        call = {'stage': 'active_decision', 'step': attempts + 1,
                'backend': backend, 'role': 'planner',
                'response_format': {'type': 'json_object'}}
        calls.append(call)
        trace['action_attempts'] += 1
        try:
            if backend == 'jev':
                data = jev_action(state, instructions, call, calls)
            else:
                data = chat([{'role': 'system', 'content': instructions},
                             {'role': 'user', 'content': [{'type': 'text', 'text':
                                 json.dumps(state, ensure_ascii=False)}]}], max_tokens=900, json_mode=True)
            call.update(response=data['_content'], usage=data.get('usage'),
                        model=data['_baseline'].get('model'),
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
            relative_size_ids = []
            for prior in reversed(history):
                details = (prior.get('observation') or {}).get('details', {})
                if details.get('dimension') == 'size':
                    relative_size_ids = details.get('size_decision', {}).get(
                        'candidate_relative_ids', [])
                    break
            relative_size_resolution = (
                valid_id and relative_size_ids == [oid] and len(remaining) > 1 and
                trusted_categories and
                all(retrieval_category(attribute.get('category', '')) ==
                    retrieval_category(parsed.get('Target', [''])[0])
                    for attribute in parsed.get('Attributes', [])) and
                all(any(word in attribute.get('phrase', '').casefold().split()
                        for word in SIZE_WORDS)
                    for attribute in parsed.get('Attributes', [])) and
                all(item.get('status') == 'known'
                    for prior in reversed(history)
                    if (prior.get('observation') or {}).get('details', {}).get('dimension') == 'size'
                    for item in (prior.get('observation') or {}).get('details', {}).get(
                        'candidate_metrics', [])) and
                not [item for item in diagnosis.get('missing', [])
                     if item != 'category/appearance evidence is uncertain or unreviewed'])
            if relative_size_resolution:
                diagnosis['candidate_ids'] = [oid]
                diagnosis['stable_candidate_ids'] = [oid]
                diagnosis['status'] = 'resolved'
                diagnosis.setdefault('active_computed_evidence', []).append({
                    'dimension': 'size', 'policy': 'planner accepted unique candidate-relative size hint',
                    'selected_id': oid, 'before': remaining,
                })
                remaining = [oid]
                resolved = True
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
        try:
            request = evidence_request(action, remaining, geometry_objects)
        except ValueError as exc:
            observation = {'tool': 'evidence', 'error': str(exc), 'no_evidence_acquired': True}
            history.append({'action': action, 'observation': observation,
                            'remaining_candidate_ids': list(remaining)})
            controller_feedback = str(exc)
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
        entry = {'action': request}
        before = list(diagnosis.get('candidate_ids', []))
        try:
            dimension = request['dimension']
            observation = execute_evidence(
                request, query, parsed, diagnosis, candidates, geometry_objects,
                xyz, rgb, ids, out, tool_count + 1, calls)
            images.extend(observation['images'])
            for step in observation['internal_steps']:
                trace['internal_usage'][step['tool']] += 1
            used_dimensions.add(dimension)
            if observation['checks']:
                checks.extend(observation['checks'])
                diagnosis = diagnose(parsed, geometry_objects, candidates, room_center, checks,
                                     trusted_categories=trusted_categories,
                                     apply_size_constraints=False)
                for computed in computed_filters:
                    apply_computed_evidence(diagnosis, computed['evidence'], computed['dimension'],
                                            record=False)
            if dimension in ('position', 'distance', 'size'):
                candidate_evidence = observation['candidate_evidence']
                apply_computed_evidence(diagnosis, candidate_evidence, dimension)
                computed_filters.append({'dimension': dimension, 'evidence': candidate_evidence})
            observation['candidate_update'] = {
                'before': before,
                'removed_ids': [oid for oid in before if oid not in diagnosis['candidate_ids']],
                'after': list(diagnosis['candidate_ids']),
            }
            observation['candidate_ids'] = list(diagnosis['candidate_ids'])
            observation['evidence_state'] = evidence_state(diagnosis)
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
