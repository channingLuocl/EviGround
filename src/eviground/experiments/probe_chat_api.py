"""User-run text JSON + image JSON API probe, without scene inference or prediction."""
import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image

from eviground.api.qwen_api import QwenAPIError, _redact, chat, image_part, parse_json_object
from eviground.api.response_schemas import TEXT, obj, schema_kwargs


def probe(output, image=None):
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a new probe output directory')
    output.mkdir(parents=True)
    record = {'status': 'running', 'calls': []}
    stage = 'text_json'
    try:
        with tempfile.TemporaryDirectory() as td:
            if image is None:
                image = Path(td) / 'red.png'
                Image.new('RGB', (64, 64), 'red').save(image)
            messages = [
                ('text_json', [{'role': 'user', 'content':
                    'Return one JSON object {"ok":true,"kind":"text"}.'}]),
                ('image_json', [{'role': 'user', 'content': [image_part(image),
                    {'type': 'text', 'text': 'Inspect this image. Return one JSON object '
                     'with keys ok (boolean), kind ("image"), dominant_color (string).'}]}]),
            ]
            for stage, request in messages:
                fields = {'ok': {'type': 'boolean', 'enum': [True]},
                          'kind': {'type': 'string', 'enum': [stage.split('_')[0]]}}
                if stage == 'image_json':
                    fields['dominant_color'] = TEXT
                data = chat(request, max_tokens=1024, json_mode=True,
                            **schema_kwargs(stage, obj(fields)))
                call = {'stage': stage, **data['_baseline'],
                        'response': data['_content'], 'finish_reason': data.get('_finish_reason'),
                        'usage': data.get('usage')}
                record['calls'].append(call)
                try:
                    answer = parse_json_object(data['_content'])
                    if (data.get('_finish_reason') != 'stop' or answer.get('ok') is not True or
                            answer.get('kind') != stage.split('_')[0] or
                            (stage == 'image_json' and not isinstance(answer.get('dominant_color'), str))):
                        raise ValueError('Probe JSON response did not satisfy the requested protocol')
                except ValueError as exc:
                    raise QwenAPIError(str(exc), diagnostic={
                        'failure_kind': 'response_parse', 'stage': 'response_parse',
                        'http_status': data['_baseline'].get('http_status'),
                        'request_id': data['_baseline'].get('request_id'),
                        'model': data['_baseline']['model'],
                        'request_summary': data['_baseline']['request_summary'],
                        'response_body': _redact(data['_content']),
                    }) from exc
        record['status'] = 'passed'
    except (ValueError, OSError) as exc:
        record.update(status='failed', error=str(exc), failure_stage=stage)
        diagnostic = getattr(exc, 'diagnostic', {})
        record['error_diagnostic'] = diagnostic
        record['calls'].append({'stage': stage, 'error': str(exc), 'api_diagnostic': diagnostic})
        raise
    finally:
        (output / 'probe.json').write_text(json.dumps(_redact(record, limit=None),
                                                     ensure_ascii=False, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', type=Path, help='Default: temporary 64x64 red test image')
    args = parser.parse_args()
    try:
        probe(args.output, args.image)
    except (ValueError, OSError) as exc:
        raise SystemExit(_redact(str(exc))) from None
    print('Text JSON and image JSON passed: ' + str(args.output / 'probe.json'))


if __name__ == '__main__':
    main()
