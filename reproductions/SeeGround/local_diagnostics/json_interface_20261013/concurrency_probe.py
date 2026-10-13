"""Repeat a saved production parser request at concurrency two, without retries."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import requests

from qwen_api import _redact, _settings

directory = Path(__file__).parent
report = json.loads((directory / 'report.json').read_text())
payload = next(r['request'] for r in report['requests']
               if r['name'] == 'production_parser_The largest picture in the room.')
base, model, key = _settings()
payload['model'] = model


def check(index):
    started = time.perf_counter()
    record = {'index': index}
    try:
        response = requests.post(base + '/chat/completions',
            headers={'Authorization': 'Bearer ' + key}, json=payload, timeout=(15, 120))
        record.update(http_status=response.status_code,
                      response_body=_redact(response.text, limit=None))
        try:
            choice = response.json()['choices'][0]
            content = choice['message']['content']
            record.update(content=content, finish_reason=choice.get('finish_reason'))
            obj = json.loads(content)
            record['valid_json_object'] = isinstance(obj, dict)
            record['expected_fields_match'] = isinstance(obj, dict) and all(
                isinstance(obj.get(k), list) for k in ('Target', 'Anchor',
                    'Relations', 'Attributes', 'Unhandled')) and isinstance(
                obj.get('Observer'), dict) and isinstance(obj.get('Viewpoint'), str)
        except (ValueError, KeyError, IndexError, TypeError):
            record['valid_json_object'] = False
    except requests.RequestException as exc:
        record['transport_error'] = type(exc).__name__
    record['latency_s'] = round(time.perf_counter() - started, 3)
    return record


with ThreadPoolExecutor(max_workers=2) as pool:
    records = list(pool.map(check, range(1, 5)))
(directory / 'concurrency_report.json').write_text(json.dumps(_redact({
    'endpoint': base, 'model': model, 'concurrency': 2, 'request': payload,
    'requests': records}, limit=None), ensure_ascii=False, indent=2))
for record in records:
    print(json.dumps({k: v for k, v in record.items()
                      if k not in ('content', 'response_body')}))
