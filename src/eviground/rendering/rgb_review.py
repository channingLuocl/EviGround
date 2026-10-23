"""Optional real RGB crops with declared projected-instance provenance."""
import hashlib
import json
from pathlib import Path
from PIL import Image
from eviground.api.qwen_api import chat, image_part, parse_json_object, QwenAPIError
from eviground.api.response_schemas import RGB_REVIEW, schema_kwargs
from eviground.grounding.appearance_evidence import attribute_contexts, normalize_visual_check, required_visual_attributes


def load_crops(scene_dir, objects):
    path = scene_dir/'rgb_crops.json'
    if not path.exists():
        return {}
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict) or manifest.get('scene') != scene_dir.name or not isinstance(manifest.get('crops'), list):
        raise ValueError('RGB manifest must bind the scene and provide crops')
    known = {o['bbox_id'] for o in objects}
    crops = {}
    for entry in manifest['crops']:
        if (not isinstance(entry, dict) or type(entry.get('id')) is not int or entry['id'] not in known
                or entry.get('binding') != 'projected_instance' or not isinstance(entry.get('source_frame'), str)
                or not entry['source_frame'].strip() or not isinstance(entry.get('image'), str)):
            raise ValueError('RGB crop requires a known instance ID, image and projected-instance source frame')
        image = (scene_dir/entry['image']).resolve()
        if not image.is_relative_to(scene_dir.resolve()) or not image.is_file():
            raise ValueError('RGB crop image must exist within its scene directory')
        with Image.open(image) as im:
            if im.format not in ('PNG', 'JPEG', 'WEBP'):
                raise ValueError('RGB crop must be PNG, JPEG or WEBP')
            im.verify()
        crops.setdefault(entry['id'], []).append({**entry, 'image': str(image)})
    return crops


def input_hashes(scene_dir, objects):
    crops = load_crops(scene_dir, objects)
    paths = [scene_dir/'rgb_crops.json'] if (scene_dir/'rgb_crops.json').exists() else []
    paths.extend(Path(crop['image']) for entries in crops.values() for crop in entries)
    return {str(path.relative_to(scene_dir.resolve())): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted({path.resolve() for path in paths})}


def review_need(need, crops, calls):
    oid = need['id']
    source_attributes = need.get('source_attributes', need.get('attributes', []))
    expected = required_visual_attributes(source_attributes)
    result = {**need, 'source_attributes': source_attributes,
              'verdict': 'uncertain', 'reason': 'no real RGB crop available',
              'attribute_checks': [{**context, 'verdict': 'uncertain',
                                   'visibility': 'missing_view',
                                   'reason': 'no real RGB crop available'}
                                  for context in attribute_contexts(expected, source_attributes)],
              'render_provenance': {'source_type': 'real_rgb_projected_instance',
                                    'binding': 'projected_instance', 'images': []}}
    if oid not in crops:
        return result
    # At most two supplied views per check; acquisition and projection are external.
    selected = crops[oid][:2]
    prompt = (f'The crops show instance ID {oid}, bound by upstream 3D-to-frame projection. '
              f'Check ONLY whether this instance belongs to category {need["category"]!r} '
              f'AND satisfies these quoted appearance attributes: {expected!r}. '
              f'Attribute contexts: {attribute_contexts(expected, source_attributes)!r}. '
              'Do not select a grounding target or infer spatial relations. Ignore surrounding objects. '
              'If the instance binding, visibility, scale or attributes cannot be verified, answer uncertain. '
              'Crops do not provide calibrated physical scale; return uncertain for physical-size '
              'conditions such as tall, short, small or large, whether they appear to match or contradict. '
              'Use no only for visible contradicting evidence; yes requires support for ALL supplied conditions. '
              'Return one independent check for every supplied attribute, including visibility. '
              'Interpret fragments in their source_quotes; a color on a stripe or panel is not a whole-body color. '
              'A bare color does not require purity, uniformity, exact shade or whole-object coverage; '
              'do not add such conditions absent from the quotes. Use ordinary surface-color matching: '
              'yes for directly supported color, no for clear incompatible color, '
              'and uncertain for actually ambiguous or obscured evidence. '
              'A modifier also present in an independent source attribute must satisfy that independent condition. '
              'The complete quoted qualified property remains mandatory; component yes alone cannot prove it. '
              'category_verdict checks category only; an attribute contradiction does not change it. '
              'The top-level verdict may summarize all conditions but never replaces attribute_checks. '
              f'Return JSON {{"id":{oid},"category_verdict":"yes|no|uncertain",'
              '"visibility":"visible|occluded|low_detail|unknown",'
              '"verdict":"yes|no|uncertain","reason":"visual evidence",'
              '"attribute_checks":[{"attribute":"supplied attribute",'
              '"verdict":"yes|no|uncertain",'
              '"visibility":"visible|occluded|low_detail|unknown",'
              '"reason":"attribute evidence"}]}}.')
    result['crops'] = selected
    result['render_provenance'] = {
        'source_type': 'real_rgb_projected_instance',
        'binding': 'projected_instance',
        'images': [crop['image'] for crop in selected],
        'source_frames': [crop['source_frame'] for crop in selected],
        'language_frame_changed': False,
    }
    try:
        data = chat([{'role':'user','content':[image_part(c['image']) for c in selected]+[{'type':'text','text':prompt}]}],
                    max_tokens=min(2048, 400 + 140 * len(expected)), json_mode=True,
                    **schema_kwargs('rgb_review', RGB_REVIEW))
        calls.append({'stage':'rgb_review','object_id':oid,'category':need['category'],
                      'response':data['_content'],'usage':data.get('usage'),
                      'latency_s':data['_baseline']['latency_s'],'finish_reason':data.get('_finish_reason'),
                      **data['_baseline']})
        if data.get('_finish_reason') != 'stop':
            raise QwenAPIError('RGB review did not complete normally: ' +
                               str(data.get('_finish_reason')))
        answer = parse_json_object(data['_content'])
        if (type(answer.get('id')) is not int or answer['id'] != oid
                or answer.get('verdict', answer.get('category_verdict')) not in ('yes','no','uncertain')
                or not isinstance(answer.get('reason'), str)):
            raise QwenAPIError('invalid RGB per-instance answer')
        answer['render_provenance'] = result['render_provenance']
        result.update(normalize_visual_check(answer, expected, source_phrases=source_attributes))
    except (ValueError, OSError) as exc:
        result.update(reason='RGB review failed; no conclusion', error=str(exc))
        if not calls or calls[-1].get('stage') != 'rgb_review' or calls[-1].get('object_id') != oid:
            calls.append({'stage':'rgb_review','object_id':oid,'error':str(exc)})
        if isinstance(exc, QwenAPIError) and getattr(exc, 'diagnostic', None):
            calls[-1]['api_diagnostic'] = dict(exc.diagnostic)
            calls[-1]['failure_kind'] = exc.diagnostic.get('failure_kind', 'response_parse')
            raise
    return result


def selective_review(parsed, objects, candidates, room_center, scene_dir, calls, limit, trusted_categories=False):
    from eviground.grounding.diagnostic import diagnose
    crops = load_crops(scene_dir, objects)
    checks, actions, attempted = [], [], set()
    diagnosis = diagnose(parsed, objects, candidates, room_center, trusted_categories=trusted_categories)
    initial = diagnosis['status']
    for _ in range(limit):
        needs = [n for n in diagnosis['visual_needs'] if (n['id'], n['category'], n['role']) not in attempted]
        if diagnosis['status'] == 'resolved' or not needs:
            break
        available = [n for n in needs if n['id'] in crops]
        if not available:
            actions.append({'action':'skip_rgb_review','reason':'no supplied RGB crop for unresolved instances',
                            'ids':sorted({n['id'] for n in needs})})
            break
        need = available[0]
        attempted.add((need['id'], need['category'], need['role']))
        actions.append({'action':'rgb_review','trigger_status':diagnosis['status'], **need})
        checks.append(review_need(need, crops, calls))
        diagnosis = diagnose(parsed, objects, candidates, room_center, checks, trusted_categories=trusted_categories)
    diagnosis.update(initial_status=initial, actions=actions, review_limit=limit,
                     rgb_manifest_present=(scene_dir/'rgb_crops.json').exists(),
                     rgb_binding='user-supplied projected-instance provenance, not independently certified')
    return diagnosis
