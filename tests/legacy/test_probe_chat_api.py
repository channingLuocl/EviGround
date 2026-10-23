import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eviground.experiments.probe_chat_api import probe
from eviground.api.qwen_api import QwenAPIError


def response(kind):
    return {'_content': json.dumps({'ok': True, 'kind': kind, 'dominant_color': 'red'}),
            '_finish_reason': 'stop', '_baseline': {'model': 'mock', 'provider': 'openai',
                'request_summary': {'contains_images': kind == 'image'}, 'http_status': 200}}


class ProbeTests(unittest.TestCase):
    def test_text_and_image_probe_records_two_json_calls_without_prediction(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'probe'
            with patch('eviground.experiments.probe_chat_api.chat', side_effect=[response('text'), response('image')]) as api:
                record = probe(out)
            self.assertEqual(api.call_count, 2)
            self.assertEqual(record['status'], 'passed')
            self.assertEqual([c['stage'] for c in record['calls']], ['text_json', 'image_json'])
            self.assertFalse((out / 'prediction.json').exists())
            self.assertNotIn('base64', (out / 'probe.json').read_text())

    def test_failed_image_probe_saves_diagnostic_without_retry_or_prediction(self):
        diagnostic = {'failure_kind': 'http_4xx', 'http_status': 400,
                      'response_body': 'invalid image', 'request_id': 'image-400'}
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'probe'
            with patch('eviground.experiments.probe_chat_api.chat', side_effect=[response('text'),
                       QwenAPIError('HTTP 400', diagnostic=diagnostic)]) as api:
                with self.assertRaises(QwenAPIError):
                    probe(out)
            self.assertEqual(api.call_count, 2)
            record = json.loads((out / 'probe.json').read_text())
            self.assertEqual(record['status'], 'failed')
            self.assertEqual(record['failure_stage'], 'image_json')
            self.assertEqual(record['error_diagnostic']['request_id'], 'image-400')
            self.assertFalse((out / 'prediction.json').exists())

    def test_application_json_parse_error_is_recorded_as_failure(self):
        bad = response('text'); bad['_content'] = 'invalid JSON'
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'probe'
            with patch('eviground.experiments.probe_chat_api.chat', return_value=bad) as api:
                with self.assertRaises(QwenAPIError):
                    probe(out)
            self.assertEqual(api.call_count, 1)
            record = json.loads((out / 'probe.json').read_text())
            self.assertEqual(record['error_diagnostic']['failure_kind'], 'response_parse')
            self.assertEqual(record['failure_stage'], 'text_json')
