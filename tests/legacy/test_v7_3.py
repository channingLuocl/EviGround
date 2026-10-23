"""Parameter-contract regressions and explicit historical GT reuse; no API calls."""
import copy
from contextlib import redirect_stderr
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from seeground.grounding.active_diagnostic import evidence_request
from seeground.api.response_schemas import planner_schema
from seeground.experiments.run_size_validation import V73_LEGACY_GT_HASHES, report, reuse_results, run_case
from seeground.grounding.v4_policy import settings


class V73Tests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'SEEGROUND_V4': '1', 'SEEGROUND_V7': '1',
            'SEEGROUND_V7_2': '0', 'SEEGROUND_V7_3': '1'})
        env.start(); self.addCleanup(env.stop)
        self.objects = [{'bbox_id': i} for i in (0, 1, 3, 4, 12, 13, 24, 25)]

    def request(self, **fields):
        return evidence_request({'action': 'evidence', 'reason': 'inspect candidates', **fields},
                                [0, 3, 12], self.objects)

    def test_recorded_parameter_conflicts_are_normalized_without_mutating_input(self):
        actions = [
            {'action': 'evidence', 'reason': 'inspect', 'dimension': 'appearance',
             'inspect': True, 'context': 'scene', 'focus_ids': [0, 1, 24, 25, 12, 13]},
            {'action': 'evidence', 'reason': 'inspect', 'dimension': 'position',
             'inspect': True, 'context': 'scene', 'focus_ids': [3, 4, 12]}]
        for action in actions:
            with self.subTest(dimension=action['dimension']):
                original = copy.deepcopy(action)
                result = evidence_request(action, [0, 3, 12], self.objects)
                self.assertEqual(action, original)
                removed = 'inspect' if action['dimension'] == 'appearance' else 'context'
                self.assertNotIn(removed, result)
                self.assertEqual(result['focus_ids'], action['focus_ids'])
                with patch.dict(os.environ, {'SEEGROUND_V7_3': '0'}):
                    with self.assertRaisesRegex(ValueError, 'unsupported evidence parameters'):
                        evidence_request(action, [0, 3, 12], self.objects)

    def test_all_geometry_dimensions_and_nullable_schema_fields(self):
        for dimension in ('size', 'position', 'distance'):
            result = self.request(dimension=dimension, context='isolated', inspect=True,
                                  focus_ids=[3, 4], predicted_id=None)
            self.assertNotIn('context', result)
            self.assertNotIn('predicted_id', result)
            self.assertTrue(result['inspect'])
        self.assertEqual(self.request(dimension='viewpoint', context=None, inspect=None)['dimension'], 'viewpoint')
        viewpoint = self.request(dimension='viewpoint', inspect=True, context='scene',
                                 focus_ids=[4], azimuth_deg=90, elevation_deg=25)
        self.assertEqual(set(viewpoint), {'action', 'dimension', 'reason'})
        self.assertEqual(settings()['version'], 'relative-size-v7.3')
        schema = planner_schema([0, 3, 12], [0, 3])
        self.assertFalse(schema['additionalProperties'])
        self.assertIn('Only for appearance', schema['properties']['context']['description'])

    def test_normalization_keeps_validation_for_invalid_requests(self):
        invalid = [dict(dimension='appearance', inspect='true'),
                   dict(dimension='position', context='invented'),
                   dict(dimension='appearance', focus_ids=[999]),
                   dict(dimension='size', focus_ids=[4]),
                   dict(dimension='appearance', focus_ids=[0, 0]),
                   dict(dimension='position', azimuth_deg=360),
                   dict(dimension='appearance', elevation_deg=76),
                   dict(dimension='appearance', fabricated=True),
                   dict(dimension='viewpoint', inspect='true'),
                   dict(dimension='viewpoint', context='invented'),
                   dict(dimension='viewpoint', focus_ids=[999]),
                   dict(dimension='viewpoint', azimuth_deg=360)]
        for fields in invalid:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.request(**fields)

    def fixture(self, root):
        row = {'csv_row': 10, 'scene': 'scene0010_00', 'query': 'the chair', 'target_id': 13}
        source = root / 'source'; source.mkdir()
        (source / 'q00010').mkdir()
        saved = {**row, 'predicted_id': None, 'correct': False,
                 'implementation_version': 'relative-size-v7.1'}
        (source / 'q00010/evaluation.json').write_text(json.dumps(saved))
        previous = {'selection': {'rows': [row], 'csv_sha256': 'same-csv'},
            'object_table': 'gt', 'v4': {'version': 'relative-size-v7.1', 'v7': True},
            'code_sha256': {**V73_LEGACY_GT_HASHES, 'qwen_api.py': 'same-code'},
            'models': {'SEEGROUND_MODEL': 'same-model'},
            'input_sha256': {'scene0010_00': {'scene.npz': 'same-scene'}},
            'query_syntax': {'implementation_version': 'relative-size-v7.1'}, 'api_config': {}}
        (source / 'manifest.json').write_text(json.dumps(previous))
        current = copy.deepcopy(previous)
        current['v4']['version'] = 'relative-size-v7.3'
        current['query_syntax']['implementation_version'] = 'relative-size-v7.3'
        current['code_sha256'].update({name: 'new-v73-code' for name in V73_LEGACY_GT_HASHES})
        return source, row, saved, previous, current

    def test_legacy_reuse_requires_opt_in_preserves_null_and_records_mixed_versions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source, row, saved, _, current = self.fixture(root)
            output = root / 'output'; output.mkdir()
            original = (source / 'q00010/evaluation.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'different inference'):
                reuse_results(source, output, [row], current)
            with redirect_stderr(StringIO()):
                reuse_results(source, output, [row], current, True)
                reuse_results(source, output, [row], current, True)
            imported = json.loads((output / 'q00010/evaluation.json').read_text())
            for key, value in saved.items():
                self.assertEqual(imported[key], value)
            self.assertEqual((source / 'q00010/evaluation.json').read_bytes(), original)
            audit = json.loads((output / 'reuse_verification.json').read_text())
            self.assertFalse(audit['compatible_settings'])
            self.assertEqual(audit['reuse_mode'], 'previous_version')
            summary = report(output, [imported, {**row, 'csv_row': 11, 'correct': True,
                             'implementation_version': 'relative-size-v7.3'}], 2)
            self.assertTrue(summary['mixed_implementation_versions'])

    def test_legacy_reuse_rejects_unverified_changes_and_predicted_categories(self):
        for field in ('model', 'input', 'hash', 'table', 'version'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as td:
                root = Path(td); source, row, _, previous, current = self.fixture(root)
                output = root / 'output'; output.mkdir()
                if field == 'model':
                    current['models']['SEEGROUND_MODEL'] = 'changed'
                elif field == 'input':
                    current['input_sha256']['scene0010_00']['scene.npz'] = 'changed'
                elif field == 'hash':
                    previous['code_sha256']['active_diagnostic.py'] = 'unverified'
                    (source / 'manifest.json').write_text(json.dumps(previous))
                elif field == 'table':
                    current['object_table'] = 'pred'
                else:
                    current['v4']['version'] = 'relative-size-v7.2'
                with self.assertRaises(ValueError):
                    reuse_results(source, output, [row], current, True)
                self.assertFalse((output / 'q00010/evaluation.json').exists())

    def test_failed_worker_still_records_v73_version(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            row = {'csv_row': 10, 'scene': 'scene0010_00', 'query': 'the chair', 'target_id': 13}
            with patch('seeground.experiments.run_size_validation.subprocess.run', side_effect=OSError('offline failure')):
                result = run_case(row, root, root, 3)
            self.assertTrue(result['inference_error'])
            self.assertEqual(result['implementation_version'], 'relative-size-v7.3')


if __name__ == '__main__':
    unittest.main()
