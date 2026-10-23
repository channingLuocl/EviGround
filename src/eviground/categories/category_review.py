"""Independent target-category checks; never changes anchors or camera."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from eviground.api.qwen_api import chat, image_part, parse_json_object, QwenAPIError
from eviground.api.response_schemas import CATEGORY_REVIEW, schema_kwargs
from eviground.rendering.run import render
from eviground.grounding.evidence import retrieval_category


def check_categories(names, candidates, xyz, rgb, ids, out, calls):
    if isinstance(names, str):
        names = [names]
    categories = sorted({retrieval_category(name) for name in names if retrieval_category(name)})
    result = {'checks': [], 'policy': 'soft evidence only; candidates, references and camera unchanged',
              'status': 'completed'}
    if len(categories) != 1:
        return {**result, 'status': 'skipped: requires one target category'}
    category = categories[0]
    for obj in candidates:
        oid = obj['bbox_id']
        points, colors = xyz[ids == oid], rgb[ids == oid]
        check = {'id': oid, 'category': category, 'predicted_label': obj['target'],
                 'verdict': 'uncertain', 'reason': 'no instance points'}
        if not len(points):
            result['checks'].append(check)
            continue
        sheet = Image.new('RGB', (768, 424), 'white')
        ImageDraw.Draw(sheet).text((10, 5), f'Instance ID {oid}: two views of this same instance',
                                  fill='black', font=ImageFont.load_default(size=20))
        center = (points.min(0)+points.max(0))/2
        span = max(float(np.ptp(points, axis=0).max()), .1)
        for view, direction in enumerate(([1., -1., .7], [-1., 1., .7])):
            path = out/f'category_{oid}_view{view}.png'
            render(points, colors, path, eye=center+span*np.asarray(direction), at=center, size=384)
            with Image.open(path) as image:
                sheet.paste(image, (view*384, 40))
        path = out/f'category_{oid}.png'
        sheet.save(path)
        check['image'] = str(path)
        prompt = (f'This image shows ONLY instance ID {oid} from two views. '
                  f'Does this instance visually belong to category "{category}"? '
                  'Inspect shape and appearance. Answer uncertain for incomplete or insufficient evidence. '
                  'Do not infer spatial relations or select a query target. Predicted labels are not supplied. '
                  f'Return JSON {{"id":{oid},"verdict":"yes|no|uncertain","reason":"short visual evidence"}}. '
                  'A no verdict requires visible evidence against the category, not just missing details.')
        try:
            output_schema = schema_kwargs('category_review', CATEGORY_REVIEW)
            data = chat([{'role':'user','content':[image_part(path),{'type':'text','text':prompt}]}],max_tokens=300,
                        **({'json_mode': True, **output_schema} if output_schema else {}))
            calls.append({'stage':'category_review','object_id':oid,'response':data['_content'],
                          'usage':data.get('usage'),'latency_s':data['_baseline']['latency_s'],
                          'finish_reason':data.get('_finish_reason'), **data['_baseline']})
            if data.get('_finish_reason') == 'length':
                raise QwenAPIError('category answer truncated')
            answer = parse_json_object(data['_content'])
            if (type(answer.get('id')) is not int or answer['id'] != oid or
                    answer.get('verdict') not in ('yes','no','uncertain') or
                    not isinstance(answer.get('reason'), str)):
                raise QwenAPIError('invalid per-instance category answer')
            check.update(verdict=answer['verdict'], reason=answer['reason'])
        except QwenAPIError as exc:
            check.update(verdict='uncertain', reason='review failed; no category conclusion', error=str(exc))
            if not calls or calls[-1].get('object_id') != oid:
                calls.append({'stage':'category_review','object_id':oid,'error':str(exc)})
            if getattr(exc, 'diagnostic', None):
                calls[-1]['api_diagnostic'] = dict(exc.diagnostic)
                calls[-1]['failure_kind'] = exc.diagnostic.get('failure_kind', 'response_parse')
                raise
        result['checks'].append(check)
    return result
