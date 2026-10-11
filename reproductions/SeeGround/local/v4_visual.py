"""Common-camera, answer-neutral box inspection and structured VLM reports."""
import itertools
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from run import render, project
from qwen_api import chat, image_part, parse_json_object


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
        if completeness.get('surface_only_risk'):
            text += ' | surface-only risk; complete volume unproved'
        draw.text((12, y + 92), text, fill='black', font=font)
    image.save(path)
    return {'image': str(path), 'focus_ids': [o['bbox_id'] for o in focus],
            'visible_instance_pixels': counts, 'eye': stats['eye'], 'at': stats['at'],
            'box_color': list(color), 'source': 'common-camera supplied boxes and observed colored points',
            'policy': 'IDs only; no target answer/colors; projection is not physical volume; view adds no missing geometry'}


def visual_report(stage, query, view, payload, calls, chat_fn=None):
    call = {'stage': stage, 'role': 'visual_verifier', 'image': view['image'],
            'focus_ids': view['focus_ids'], 'response_format': {'type': 'json_object'}}
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
    try:
        data = (chat_fn or chat)([{'role': 'system', 'content': instructions},
            {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps(
                {'query': query, 'view': {k: v for k, v in view.items() if k != 'image'}, 'task': payload}, ensure_ascii=False)},
                image_part(view['image'])]}], max_tokens=1600, json_mode=True)
        call.update(response=data['_content'], usage=data.get('usage'), **data.get('_baseline', {}))
        if data.get('_finish_reason') != 'stop':
            raise ValueError('visual report did not complete normally')
        report = parse_json_object(data['_content'])
        return normalize_report(report, query, view)
    except (ValueError, OSError) as exc:
        call['error'] = str(exc)
        if getattr(exc, 'diagnostic', None):
            call['api_diagnostic'] = dict(exc.diagnostic)
            raise
        return {'status': 'unknown', 'reason': str(exc), 'contradictions': [], 'observations': [],
                'relations': [], 'metric_choices': [], 'supported': False}


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
                elif row.get('metric') not in ('oriented_volume_m3', 'oriented_footprint_m2') or len(visible) < 2:
                    continue
            result[field].append(row)
    result['supported'] = bool(result['contradictions'])
    if result['status'] == 'contradiction' and not result['supported']:
        result['status'] = 'unknown'
    return result
