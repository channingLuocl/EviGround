"""One-off interface audit: sequential requests, no retries or scene inference."""
import json
import time
from pathlib import Path
from unittest.mock import patch

import requests
from PIL import Image

from qwen_api import _settings, _redact, api_request_slot, image_part, request_options
from real_scene import parse_query


OUTPUT = Path(__file__).parent
base, model, key = _settings()
records = []


def check(name, messages, response_format, tokens=512, expected=None):
    payload = {'model': model, 'messages': messages,
               **request_options(tokens, None, False)}
    if response_format is not None:
        payload['response_format'] = response_format
    record = {'name': name, 'request': _redact(payload, limit=None)}
    started = time.perf_counter()
    try:
        with api_request_slot():
            response = requests.post(base + '/chat/completions',
                headers={'Authorization': 'Bearer ' + key}, json=payload,
                timeout=(15, 120))
        record.update(http_status=response.status_code,
                      request_id=response.headers.get('x-request-id'),
                      response_body=_redact(response.text, limit=None))
        try:
            data = response.json()
            choice = data['choices'][0]
            content = choice['message'].get('content')
            record.update(content=content, finish_reason=choice.get('finish_reason'),
                          usage=data.get('usage'))
            try:
                obj = json.loads(content)
                record['valid_json_object'] = isinstance(obj, dict)
                if expected is not None:
                    record['expected_fields_match'] = expected(obj)
            except (ValueError, TypeError):
                record['valid_json_object'] = False
        except (ValueError, KeyError, IndexError, TypeError):
            record['valid_completion_response'] = False
    except requests.RequestException as exc:
        record['transport_error'] = type(exc).__name__
    record['latency_s'] = round(time.perf_counter() - started, 3)
    records.append(_redact(record, limit=None))
    (OUTPUT / 'report.json').write_text(json.dumps({'endpoint': base, 'model': model,
        'requests': records}, ensure_ascii=False, indent=2))
    print(json.dumps({k: record.get(k) for k in ('name', 'http_status',
        'valid_json_object', 'expected_fields_match', 'finish_reason',
        'transport_error', 'latency_s')}), flush=True)


json_mode = {'type': 'json_object'}
schema_mode = {'type': 'json_schema', 'json_schema': {'name': 'relay_probe',
    'strict': True, 'schema': {'type': 'object', 'properties': {
        'ok': {'type': 'boolean', 'enum': [True]},
        'marker': {'type': 'string', 'enum': ['seeground_json_probe']}},
        'required': ['ok', 'marker'], 'additionalProperties': False}}}
schema_matches = lambda obj: obj == {'ok': True, 'marker': 'seeground_json_probe'}

check('simple_json', [{'role': 'user', 'content':
    'Return only the JSON object {"ok":true,"kind":"text"}.'}], json_mode,
    expected=lambda obj: obj == {'ok': True, 'kind': 'text'})
conflict = [{'role': 'user', 'content':
    'This is a JSON mode interface test. Reply with exactly the plain text '
    'PROBE_COMPLETE. Do not use JSON, quotes, braces, or code fences.'}]
check('plain_text_control', conflict, None)
check('json_mode_enforcement', conflict, json_mode)
check('strict_schema_enforcement', conflict, schema_mode, expected=schema_matches)
check('invalid_format_validation', [{'role': 'user', 'content':
    'Return only JSON {"ok":true}.'}], {'type': 'invalid_seeground_probe'})

class CapturedRequest(Exception):
    pass

for query in ('The largest picture in the room.', 'the smaller chair'):
    captured = {}
    def capture(messages, **options):
        captured.update(messages=messages, options=options)
        raise CapturedRequest()
    try:
        with patch('real_scene.chat', side_effect=capture):
            parse_query(query, mode='active_diagnostic')
    except CapturedRequest:
        pass
    check('production_parser_' + query, captured['messages'], json_mode,
          tokens=captured['options']['max_tokens'],
          expected=lambda obj: isinstance(obj, dict) and all(
              isinstance(obj.get(k), list) for k in ('Target', 'Anchor',
                  'Relations', 'Attributes', 'Unhandled')) and
              isinstance(obj.get('Viewpoint'), str) and isinstance(obj.get('Observer'), dict))

image = OUTPUT / 'red.png'
Image.new('RGB', (64, 64), 'red').save(image)
check('image_json', [{'role': 'user', 'content': [image_part(image),
    {'type': 'text', 'text': 'Inspect the image. Return only the JSON object '
     '{"ok":true,"kind":"image","dominant_color":"red"} if it is red.'}]}],
    json_mode, expected=lambda obj: obj == {'ok': True, 'kind': 'image', 'dominant_color': 'red'})
