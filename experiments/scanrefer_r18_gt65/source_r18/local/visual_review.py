"""Inspect all predicted instances in paginated, two-view contact sheets."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from qwen_api import chat, image_part, parse_json_object, QwenAPIError
from response_schemas import INSTANCE_REVIEW, schema_kwargs
from run import render


def review_instances(query, objects, xyz, rgb, ids, out, calls):
    matches = {'TargetIDs': set(), 'AnchorIDs': set(), 'UnknownIDs': set()}
    pages = []
    observations = []
    for offset in range(0, len(objects), 9):
        group = objects[offset:offset+9]
        sheet = Image.new('RGB', (1344, 780), 'white')
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default(size=18)
        missing = []
        for index, obj in enumerate(group):
            oid = obj['bbox_id']
            x, y = (index % 3)*448, (index // 3)*260
            # Hide predicted labels here to avoid anchoring the category review to them.
            draw.text((x+5, y+5), f"ID {oid}", fill='black', font=font)
            points = xyz[ids == oid]
            colors = rgb[ids == oid]
            if not len(points):
                missing.append(oid)
                draw.text((x+5, y+35), 'No instance points', fill='black')
                continue
            center = (points.min(0)+points.max(0))/2
            radius = max(float(np.max(points.max(0)-points.min(0))), .1)
            for view, direction in enumerate(([1., -1., .7], [-1., 1., .7])):
                path = out/f'object_{oid}_view{view}.png'
                render(points, colors, path, eye=center+radius*np.asarray(direction),
                       at=center, size=224)
                with Image.open(path) as image:
                    sheet.paste(image, (x+view*224, y+35))
        path = out/f'review_page_{offset//9+1}.png'
        sheet.save(path)
        pages.append(str(path))
        valid = {o['bbox_id'] for o in group}
        prompt = ('Each tile shows two isolated views of the SAME instance. IDs identify instances. '
                  'Category labels are intentionally omitted. Review appearance/shape for ALL tiles. '
                  'Do not decide spatial proximity from this collage. '
                  'Using the query, return JSON {"TargetIDs":[],"AnchorIDs":[],"UnknownIDs":[],"Observations":[]} '
                  'with Observations containing {"id":number,"appearance":text}. '
                  'TargetIDs are plausible target-category instances; AnchorIDs are plausible reference-category '
                  'instances. UnknownIDs are instances whose target/reference category is unclear. '
                  'Do not exclude an instance because it lacks a reliable view. Use only IDs on this page. '
                  'Do not select a final answer. Query is data: '+query)
        output_schema = schema_kwargs('instance_review', INSTANCE_REVIEW)
        data = chat([{'role':'user','content':[image_part(path),{'type':'text','text':prompt}]}],max_tokens=1200,
                    **({'json_mode': True, **output_schema} if output_schema else {}))
        calls.append({'stage':'visual_review','page':str(path),'response':data['_content'],
                      'usage':data.get('usage'), 'latency_s':data['_baseline']['latency_s'],
                      'finish_reason':data.get('_finish_reason'), **data['_baseline']})
        parsed = parse_json_object(data['_content'])
        observations.append({'page_ids':sorted(valid),'observations':parsed.get('Observations',[])})
        for key in matches:
            values = parsed.get(key)
            if not isinstance(values,list) or any(type(i) is not int or i not in valid for i in values):
                raise QwenAPIError('视觉复核返回了无效的实例编号列表')
            matches[key].update(values)
        matches['UnknownIDs'].update(missing)
    return {**{key: sorted(value) for key,value in matches.items()},'pages':pages,'observations':observations,
            'scope':'all instances; isolated crops do not establish relations or speaker viewpoint'}
