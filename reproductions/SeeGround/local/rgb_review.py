"""Optional real RGB crops with declared projected-instance provenance."""
import hashlib
import json
from pathlib import Path
from PIL import Image
from qwen_api import chat, image_part, parse_json_object, QwenAPIError


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
    result = {**need, 'verdict': 'uncertain', 'reason': 'no real RGB crop available'}
    if oid not in crops:
        return result
    # At most two supplied views per check; acquisition and projection are external.
    selected = crops[oid][:2]
    prompt = (f'The crops show instance ID {oid}, bound by upstream 3D-to-frame projection. '
              f'Check ONLY whether this instance belongs to category {need["category"]!r} '
              f'AND satisfies these quoted appearance attributes: {need["attributes"]!r}. '
              'Do not select a grounding target or infer spatial relations. Ignore surrounding objects. '
              'If the instance binding, visibility, scale or attributes cannot be verified, answer uncertain. '
              'Use no only for visible contradicting evidence; yes requires support for ALL supplied conditions. '
              f'Return JSON {{"id":{oid},"verdict":"yes|no|uncertain","reason":"visual evidence"}}.')
    result['crops'] = selected
    try:
        data = chat([{'role':'user','content':[image_part(c['image']) for c in selected]+[{'type':'text','text':prompt}]}],max_tokens=350)
        calls.append({'stage':'rgb_review','object_id':oid,'category':need['category'],
                      'response':data['_content'],'usage':data.get('usage'),
                      'latency_s':data['_baseline']['latency_s'],'finish_reason':data.get('_finish_reason')})
        if data.get('_finish_reason') == 'length':
            raise QwenAPIError('RGB review truncated')
        answer = parse_json_object(data['_content'])
        if (type(answer.get('id')) is not int or answer['id'] != oid or answer.get('verdict') not in ('yes','no','uncertain')
                or not isinstance(answer.get('reason'), str)):
            raise QwenAPIError('invalid RGB per-instance answer')
        result.update(verdict=answer['verdict'], reason=answer['reason'])
    except (ValueError, OSError) as exc:
        result.update(reason='RGB review failed; no conclusion', error=str(exc))
        if not calls or calls[-1].get('stage') != 'rgb_review' or calls[-1].get('object_id') != oid:
            calls.append({'stage':'rgb_review','object_id':oid,'error':str(exc)})
    return result


def selective_review(parsed, objects, candidates, room_center, scene_dir, calls, limit, trusted_categories=False):
    from diagnostic import diagnose
    crops = load_crops(scene_dir, objects)
    checks, actions, attempted = [], [], set()
    diagnosis = diagnose(parsed, objects, candidates, room_center, trusted_categories=trusted_categories)
    initial = diagnosis['status']
    for _ in range(limit):
        needs = [n for n in diagnosis['visual_needs'] if (n['id'], n['category']) not in attempted]
        if diagnosis['status'] == 'resolved' or not needs:
            break
        available = [n for n in needs if n['id'] in crops]
        if not available:
            actions.append({'action':'skip_rgb_review','reason':'no supplied RGB crop for unresolved instances',
                            'ids':sorted({n['id'] for n in needs})})
            break
        need = available[0]
        attempted.add((need['id'], need['category']))
        actions.append({'action':'rgb_review','trigger_status':diagnosis['status'], **need})
        checks.append(review_need(need, crops, calls))
        diagnosis = diagnose(parsed, objects, candidates, room_center, checks, trusted_categories=trusted_categories)
    diagnosis.update(initial_status=initial, actions=actions, review_limit=limit,
                     rgb_manifest_present=(scene_dir/'rgb_crops.json').exists(),
                     rgb_binding='user-supplied projected-instance provenance, not independently certified')
    return diagnosis
