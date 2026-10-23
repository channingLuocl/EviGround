"""Common-camera, answer-neutral box inspection and structured VLM reports."""
import itertools
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from run import render, project
from qwen_api import chat, image_part, parse_json_object
from response_schemas import VISUAL_REPORT, schema_kwargs
from size_semantics import evaluate_size_condition
from v4_policy import v5_enabled


def boxed_view(objects, focus_ids, xyz, rgb, ids, out, name, measurements=(), azimuth=90, elevation=30):
    known = {o['bbox_id']: o for o in objects}
    focus = [known[oid] for oid in dict.fromkeys(focus_ids) if oid in known][:6]
    if not focus:
        raise ValueError('boxed inspection requires known candidates')
    corners = {o['bbox_id']: np.asarray([np.asarray(o['bbox_3d'][:3]) +
               np.asarray(sign) * np.asarray(o['bbox_3d'][3:]) / 2
               for sign in itertools.product((-1, 1), repeat=3)]) for o in focus}
    bounds = np.concatenate(list(corners.values()))
    lo, hi = bounds.min(0), bounds.max(0)
    at = (lo + hi) / 2
    span = max(float(np.linalg.norm(hi - lo)), .5)
    a, e = np.deg2rad([azimuth, elevation])
    eye = at + 1.35 * span * np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])
    region = np.all((xyz >= lo - .25 * span) & (xyz <= hi + .25 * span), axis=1)
    path = Path(out) / (name + '.png')
    stats = render(xyz[region], rgb[region], path, eye=eye, at=at, size=1000, point_radius=1)
    # Visibility from the same point z-buffer; a drawn box alone is not a visible object.
    uv, depth = project(xyz[region], eye, at, 1000)
    valid = (depth > .001) & np.all((uv >= 0) & (uv < 1000), axis=1)
    indices = np.flatnonzero(valid)
    pixels = uv[indices].astype(int)
    order = np.argsort(depth[indices], kind='stable')
    _, first = np.unique((pixels[:, 1] * 1000 + pixels[:, 0])[order], return_index=True)
    visible_ids = ids[region][indices[order[first]]]
    counts = {str(o['bbox_id']): int(np.sum(visible_ids == o['bbox_id'])) for o in focus}
    scene = Image.open(path).convert('RGB')
    image = Image.new('RGB', (1000, 1000 + 110 * len(focus) + 35), 'white')
    image.paste(scene, (0, 0))
    draw, font = ImageDraw.Draw(image), ImageFont.load_default(size=18)
    color = (25, 105, 180)
    by_id = {r['id']: r for r in measurements}
    measurement_states = {}
    for index, obj in enumerate(focus):
        oid = obj['bbox_id']
        projected, z = project(corners[oid], eye, at, 1000)
        for i, j in itertools.combinations(range(8), 2):
            if bin(i ^ j).count('1') == 1 and min(z[i], z[j]) > .001:
                draw.line([tuple(projected[i]), tuple(projected[j])], fill=color, width=3)
        center, z = project(np.asarray([obj['bbox_3d'][:3]]), eye, at, 1000)
        if z[0] > .001:
            x, y = center[0]
            draw.text((max(0, min(960, x)), max(0, min(975, y))), str(oid), fill=color, font=font,
                      stroke_width=2, stroke_fill='white')
        row = by_id.get(oid, {})
        y = 1008 + index * 110
        dimensions = ' x '.join(f'{v:.3f}' for v in obj['bbox_3d'][3:])
        draw.text((12, y), f'ID {oid} | {obj["target"]} | AABB dx,dy,dz = {dimensions} m', fill=color, font=font)
        draw.text((12, y + 23), 'Source: supplied instance box; same observation as assigned point cloud; Z up', fill='black', font=font)
        metrics = [f'{key}={row.get(key):.4g}' for key in ('oriented_volume_m3', 'oriented_footprint_m2', 'height_m')
                   if isinstance(row.get(key), (int, float))]
        draw.text((12, y + 46), ' | '.join(metrics) or 'Point metrics unavailable', fill='black', font=font)
        quality = row.get('metric_reliability', {})
        draw.text((12, y + 69), 'Boundary quality: ' + ' | '.join(f'{key}: {quality.get(key, {}).get("status", "unknown")}'
                  for key in ('oriented_volume_m3', 'oriented_footprint_m2', 'height_m')), fill='black', font=font)
        completeness = row.get('observed_completeness', {})
        text = 'Completeness: ' + completeness.get('status', 'unknown')
        from v4_policy import v5_enabled
        if v5_enabled():
            data_state = 'not_supplied' if not row else 'computed' if row.get('status') == 'known' else 'calculation_failed'
            measurement_states[str(oid)] = {'data_status': data_state,
                'reason': row.get('reason', row.get('reliability_reason')),
                'tabletop_status': quality.get('tabletop_height_m', {}).get('status',
                    'not_applicable' if obj['target'].casefold() not in ('table', 'desk') else 'not_supplied')}
            draw.rectangle((0, y + 46, 999, y + 91), fill='white')
            fields = ('oriented_volume_m3', 'oriented_footprint_m2', 'height_m', 'tabletop_height_m', 'body_plane_area_m2')
            labels = {'oriented_volume_m3': 'volume[m3]', 'oriented_footprint_m2': 'footprint[m2]',
                      'height_m': 'height[m]', 'tabletop_height_m': 'tabletop[m]', 'body_plane_area_m2': 'body area[m2]'}
            metrics = [f'{labels[k]}={row[k]:.4g}' for k in fields if isinstance(row.get(k), (float, int))]
            draw.text((12, y + 46), ' | '.join(metrics) or 'Measurements: ' + data_state, fill='black', font=font)
            draw.text((12, y + 69), ' | '.join(f'{labels[k]}: {quality.get(k, {}).get("status", "not_supplied")}'
                for k in fields if k in quality) or 'Quality: ' + data_state, fill='black', font=font)
            if row and 'tabletop_height_m' not in quality:
                text += ' | tabletop: ' + measurement_states[str(oid)]['tabletop_status']
        if completeness.get('surface_only_risk'):
            text += ' | surface-only risk; complete volume unproved'
        draw.text((12, y + 92), text, fill='black', font=font)
    image.save(path)
    return {'image': str(path), 'focus_ids': [o['bbox_id'] for o in focus],
            'visible_instance_pixels': counts, 'eye': stats['eye'], 'at': stats['at'],
            'box_color': list(color), 'source': 'common-camera supplied boxes and observed colored points',
            **({'measurement_data_status': measurement_states} if v5_enabled() else {}),
            'policy': 'IDs only; no target answer/colors; projection is not physical volume; view adds no missing geometry'}


def visual_report(stage, query, view, payload, calls, chat_fn=None):
    output_schema = schema_kwargs('visual_report', VISUAL_REPORT)
    call = {'stage': stage, 'role': 'visual_verifier', 'image': view['image'],
            'focus_ids': view['focus_ids'], 'response_format': {'type': 'json_object'}}
    if output_schema:
        call['response_format'] = {'type': 'json_schema', 'json_schema': output_schema['response_schema']}
    calls.append(call)
    instructions = '''Inspect the original query and the numbered instances in this common-camera point-cloud view.
All boxes have the same color. Box and point cloud are the same scan evidence, not independent confirmations.
Do not infer hidden geometry, speaker direction, or 3D volume from projected area. Low detail or occlusion means unknown.
Return JSON with:
"observations":[{"id":1,"completeness":"supported|suspected_incomplete|unknown","visible_surfaces":["side","bottom"],"reason":"concrete visible evidence"}],
"metric_choices":[{"source_quote":"exact original size quote","category":"supplied category","word":"size word","metric":"oriented_volume_m3|oriented_footprint_m2","reason":"visible structure supporting this proxy"}],
"relations":[{"candidate_id":1,"source_quote":"exact supplied relation quote","status":"pass|fail|unknown","reason":"visible spatial evidence"}],
"contradictions":[{"candidate_ids":[1,2],"source_quote":"exact query constraint","evidence_type":"visible_geometry|measurement","reason":"specific contradiction, not a preference"}],
"status":"consistent|contradiction|unknown", "reason":"what remains unverified".
Complete only fields relevant to the supplied task. A different preferred ID is not a contradiction.
In final review, inspect selected and alternative IDs; review consistency does not certify all description constraints.'''
    from v4_policy import v5_enabled
    if v5_enabled():
        instructions = instructions.replace('oriented_volume_m3|oriented_footprint_m2', 'oriented_volume_m3|oriented_footprint_m2|body_plane_area_m2')
        instructions += '''
For every relation, include anchor_id and evaluate each supplied reference binding separately.
In final review return findings, each with finding=confirms_exclusion|refutes_exclusion|contradicts_selected|unknown,
candidate_id, event_id (null for selected), source_quote, target_role="target", predicate_status=pass|fail|unknown,
evidence_type=visible_geometry|measurement, reason, observation_ids, measurement_keys.
confirms_exclusion means this candidate fails the quoted requirement, supporting the existing deletion.
refutes_exclusion means new evidence shows it passes the exact requirement that caused deletion.
contradicts_selected means the selected candidate fails the requirement. A failed alternative does NOT refute its deletion.
Link visible evidence to observations with condition_checks=[{source_quote,status:pass|fail|unknown,reason}].
Measurement findings must include comparison={key,operator:lt|gt,value,reference_id,reference_key};
The comparison must reproduce a bound, known, non-tentative size_constraints predicate; provide reference_id and reference_key.
Unknown completeness is not a contradiction of a reliable measurement. Body-plane area does not require a bottom view.'''
    from v4_policy import v7_enabled
    if v7_enabled():
        instructions += ('\nV7 box metrics describe supplied observed instance bounds. Scan incompleteness or unknown '
            'alone cannot refute their ordering. Ordinary taller/shorter compares total instance dz; '
            'tabletop elevation is a different predicate. Provisional ordering does not certify missing conditions.')
    from v4_policy import v74_enabled
    if v74_enabled():
        instructions += ('\nKeep output compact: at most two condition checks per instance and at most 20 words '
            'per reason. Do not duplicate observations as findings unless they change the decision.')
    if output_schema:
        instructions += '\nReturn every schema field; use empty arrays for unused fields and null for inapplicable comparison or anchor_id.'
    instructions += ('\nOUTPUT FORMAT (STRICT): Respond with ONLY one raw JSON object. '
        'No markdown fences, no prose, no text outside the JSON. '
        'Start with { and end with }. '
        'Example: {"observations":[],"metric_choices":[],"relations":[],"contradictions":[],"status":"unknown","reason":"nothing verified"}')
    try:
        data = (chat_fn or chat)([{'role': 'system', 'content': instructions},
            {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps(
                {'query': query, 'view': {k: v for k, v in view.items() if k != 'image'}, 'task': payload}, ensure_ascii=False)},
                image_part(view['image'])]}], max_tokens=4800 if v74_enabled() else 2400, json_mode=True,
                **output_schema)
        call.update(response=data['_content'], usage=data.get('usage'), **data.get('_baseline', {}))
        if data.get('_finish_reason') != 'stop':
            raise ValueError('visual report did not complete normally')
        report = parse_json_object(data['_content'])
        result = normalize_report(report, query, view)
        if v5_enabled():
            result['findings'] = validate_findings(report, query, view, payload)
            result['restore_supported'] = any(f['finding'] == 'refutes_exclusion' for f in result['findings'])
            result['selected_contradicted'] = any(f['finding'] == 'contradicts_selected' for f in result['findings'])
            result['supported'] = bool(result['findings'])
            if result['restore_supported'] or result['selected_contradicted']:
                result['status'] = 'contradiction'
        return result
    except (ValueError, OSError) as exc:
        call['error'] = str(exc)
        if getattr(exc, 'diagnostic', None):
            call['api_diagnostic'] = dict(exc.diagnostic)
            raise
        return {'status': 'unknown', 'reason': str(exc), 'contradictions': [], 'observations': [],
                'relations': [], 'metric_choices': [], 'supported': False}


def validate_findings(report, query, view, payload):
    events = {e['event_id']: e for e in payload.get('deletions', [])}
    visible = {i for i in view['focus_ids'] if view['visible_instance_pixels'].get(str(i), 0) >= 20}
    observations = {o.get('id'): o for o in (report.get('observations') or []) if isinstance(o, dict) and o.get('reason')}
    measurements = {r['id']: r for r in payload.get('measurements', [])}
    inventory = payload.get('evidence_inventory') or {}
    result = []
    findings = report.get('findings', [])
    for f in findings if isinstance(findings, list) else []:
        if not isinstance(f, dict):
            continue
        kind, oid, quote = f.get('finding'), f.get('candidate_id'), f.get('source_quote')
        if (kind not in ('confirms_exclusion', 'refutes_exclusion', 'contradicts_selected')
                or type(oid) is not int or oid not in visible or not isinstance(quote, str)
                or not quote.strip() or quote.casefold() not in query.casefold()
                or f.get('target_role') != 'target' or not isinstance(f.get('reason'), str) or not f['reason'].strip()):
            continue
        if f.get('predicate_status') != ('pass' if kind == 'refutes_exclusion' else 'fail'):
            continue
        if kind == 'contradicts_selected':
            requirements = [r for r in inventory.get('items', []) if r.get('target_role')]
            if requirements and not any(quote == r.get('source_quote') for r in requirements):
                continue
            if oid != payload.get('selected_id'):
                continue
        else:
            event = events.get(f.get('event_id'))
            if not event or event['candidate_id'] != oid or event.get('invalidated') or quote not in event['source_quotes']:
                continue
        if f.get('evidence_type') == 'visible_geometry':
            ids = f.get('observation_ids', [])
            if not isinstance(ids, list) or oid not in ids or any(type(i) is not int or i not in observations or i not in visible for i in ids):
                continue
            proofs = observations[oid].get('condition_checks', [])
            if not isinstance(proofs, list) or not any(isinstance(p, dict) and p.get('source_quote') == quote and p.get('status') == f['predicate_status']
                       and isinstance(p.get('reason'), str) and p['reason'].strip() for p in proofs):
                continue
            reference_required = any(r.get('source_quote') == quote and r.get('reference_role')
                                     for r in inventory.get('items', []))
            reference_required = reference_required or any(r.get('candidate_id') == oid and r.get('source_quote') == quote
                and r.get('geometry_evidence', {}).get('anchor_ids') for r in payload.get('relations', []))
            if reference_required:
                bound = [r for r in payload.get('relations', []) if r.get('candidate_id') == oid
                    and r.get('source_quote') == quote and r.get('role_binding_status') == 'bound'
                    and r.get('geometry_evidence', {}).get('reference_binding', {}).get('status') != 'unknown']
                anchors = {anchor for r in bound for anchor in r.get('geometry_evidence', {}).get('anchor_ids', [])}
                witnessed = {p.get('anchor_id') for p in (report.get('relations') or []) if isinstance(p, dict)
                    and p.get('candidate_id') == oid and p.get('source_quote') == quote
                    and p.get('status') == f['predicate_status'] and p.get('reason')
                    and p.get('anchor_id') in ids}
                if not anchors or (not anchors & witnessed if f['predicate_status'] == 'pass' else not anchors <= witnessed):
                    continue
        elif f.get('evidence_type') == 'measurement':
            keys = f.get('measurement_keys', [])
            row = measurements.get(oid, {})
            if row.get('status') != 'known' or not row.get('source') or not isinstance(keys, list) or not keys or any(not isinstance(k, str) or type(row.get(k)) not in (int, float) or
                              not np.isfinite(row[k]) or
                              row.get('metric_reliability', {}).get(k, {}).get('status') != 'usable' for k in keys):
                continue
            comparison = f.get('comparison', {})
            if not isinstance(comparison, dict):
                continue
            key = comparison.get('key')
            if key not in keys:
                continue
            # A numerical comparison must reproduce a currently source-bound
            # predicate; a coincidental number elsewhere in the trace is no proof.
            constraints = [c for c in payload.get('size_constraints', [])
                if c.get('original_quote') == quote and c.get('metric') == key
                and c.get('scope_status') == 'bound' and c.get('status') == 'known'
                and not c.get('tentative_metric')]
            proven = []
            for constraint in constraints:
                hypotheses = constraint.get('scope_hypotheses') or [constraint]
                if any(not all(name in hypothesis for name in ('word', 'direction', 'form', 'interpretation', 'metric_reason'))
                       for hypothesis in hypotheses):
                    continue
                checks = [evaluate_size_condition(hypothesis, list(measurements.values()),
                    hypothesis.get('comparison_pool_ids', []), scope_bound=hypothesis.get('scope_status') == 'bound',
                    candidate_ids=[oid], scope_source=hypothesis.get('scope_source', 'review current evidence'),
                    scope_unknown_ids=hypothesis.get('scope_unknown_ids', [])) for hypothesis in hypotheses]
                if all(any(e['candidate_id'] == oid and e['status'] == f['predicate_status']
                           for e in check['candidate_evidence']) for check in checks):
                    proven.append(constraint)
            if not proven:
                continue
            reference = measurements.get(comparison.get('reference_id'), {})
            reference_key = comparison.get('reference_key')
            threshold = reference.get(reference_key)
            if (reference_key != key or not isinstance(threshold, (int, float))
                    or reference.get('metric_reliability', {}).get(reference_key, {}).get('status') != 'usable'
                    or not reference.get('source') or reference.get('status') != 'known'
                    or not np.isfinite(threshold)
                    or not any(comparison.get('reference_id') in (c.get('reference_ids') or c.get('comparison_pool_ids', []))
                               for c in proven)):
                continue
            expected_operators = {'lt' if (c.get('direction') == 'min') == (f['predicate_status'] == 'pass') else 'gt'
                                  for c in proven if c.get('direction') in ('min', 'max')}
            if comparison.get('operator') not in expected_operators or not (
                row[key] < threshold if comparison['operator'] == 'lt' else row[key] > threshold):
                continue
        else:
            continue
        result.append({**f, 'validated': True})
    return result


def normalize_report(report, query, view):
    allowed = set(view['focus_ids'])
    visible = {oid for oid in allowed if view['visible_instance_pixels'].get(str(oid), 0) >= 20}
    result = {'status': report.get('status') if report.get('status') in
              ('consistent', 'contradiction', 'unknown') else 'unknown',
              'reason': report.get('reason', ''), 'observations': [], 'relations': [],
              'metric_choices': [], 'contradictions': []}
    for field in ('observations', 'relations', 'metric_choices', 'contradictions'):
        rows = report.get(field, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get('reason'), str) or not row['reason'].strip():
                continue
            if field == 'observations':
                if type(row.get('id')) is not int or row['id'] not in visible:
                    continue
                row = dict(row)
                if row.get('completeness') not in ('supported', 'suspected_incomplete', 'unknown'):
                    row['completeness'] = 'unknown'
                if row['completeness'] == 'supported' and not {'side', 'bottom'}.issubset(row.get('visible_surfaces', [])):
                    row['completeness'] = 'unknown'
            else:
                quote = row.get('source_quote')
                if not isinstance(quote, str) or not quote.strip() or quote.casefold() not in query.casefold():
                    continue
                if field == 'contradictions':
                    ids = row.get('candidate_ids')
                    if (not isinstance(ids, list) or not ids or any(type(i) is not int or i not in visible for i in ids)
                            or row.get('evidence_type') not in ('visible_geometry', 'measurement')):
                        continue
                elif field == 'relations':
                    if type(row.get('candidate_id')) is not int or row['candidate_id'] not in visible or row.get('status') not in ('pass', 'fail', 'unknown'):
                        continue
                elif row.get('metric') not in (('oriented_volume_m3', 'oriented_footprint_m2', 'body_plane_area_m2')
                        if v5_enabled() else ('oriented_volume_m3', 'oriented_footprint_m2')) or len(visible) < 2:
                    continue
            result[field].append(row)
    result['supported'] = bool(result['contradictions'])
    if result['status'] == 'contradiction' and not result['supported']:
        result['status'] = 'unknown'
    return result
