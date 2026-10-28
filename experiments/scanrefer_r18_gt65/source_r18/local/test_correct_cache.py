"""Ever-correct answers survive inference changes and are never rerun."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from run_size_validation import (apply_correct_cache, correct_cache_path,
    load_correct_cache, main, remember_correct_result, run_case)


class CorrectCacheTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'SEEGROUND_V4': '1', 'SEEGROUND_V7': '1',
            'SEEGROUND_V7_2': '0', 'SEEGROUND_V7_3': '1', 'SEEGROUND_SYNTAX_BACKEND': 'none',
            'SEEGROUND_API_PROVIDER': 'openai', 'SEEGROUND_REACT_BACKEND': 'openai',
            'SEEGROUND_MODEL': 'offline', 'SEEGROUND_BASE_URL': 'https://example.invalid/v1',
            'SEEGROUND_API_KEY': 'offline', 'SEEGROUND_API_SLOT_DIR': ''})
        env.start(); self.addCleanup(env.stop)

    def row(self, index=10):
        return {'csv_row': index, 'scene': 'scene0010_00', 'query': 'the chair',
                'target_id': 0, 'stimulus_candidate_ids': [0, 5]}

    def selection(self):
        return {'csv_sha256': 'test-csv', 'rows': [self.row(), self.row(11)]}

    def seed(self, root, name='old', oid=0, table='gt', row=None, csv_hash='test-csv', nested=False):
        row = self.row() if row is None else row
        run = root / name; run.mkdir()
        (run / 'manifest.json').write_text(json.dumps({'selection': {
            'csv_sha256': csv_hash, 'rows': [row]}, 'object_table': table,
            'v4': {'version': name}, 'code_sha256': {'active_diagnostic.py': name}}))
        directory = run / f"q{row['csv_row']:05d}"
        if nested:
            directory /= 'null_retries/attempt_001'
        directory.mkdir(parents=True)
        result = {**row, 'predicted_id': oid, 'correct': oid == row['target_id'],
                  'object_table': table, 'implementation_version': name}
        (directory / 'evaluation.json').write_text(json.dumps(result))
        return run, result

    def test_success_imports_across_versions_and_group_metadata_with_zero_id(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old, _ = self.seed(root, row={**self.row(), 'stage': 'size'})
            original = (old / 'q00010/evaluation.json').read_bytes()
            cache = load_correct_cache(root, self.selection(), 'gt')
            output = root / 'new'; output.mkdir()
            self.assertEqual(apply_correct_cache(cache, output, self.selection()['rows'])['imported_rows'], [10])
            saved = json.loads((output / 'q00010/evaluation.json').read_text())
            self.assertEqual(saved['predicted_id'], 0)
            self.assertEqual(saved['implementation_version'], 'old')
            self.assertEqual((old / 'q00010/evaluation.json').read_bytes(), original)
            with patch('run_size_validation.subprocess.run') as process:
                run_case(self.row(), root, output, 3, retry_false=True)
                run_case(self.row(), root, output, 3, retry_errors=True)
            process.assert_not_called()

    def test_wrong_null_predicted_table_and_other_csv_are_not_imported(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.seed(root, name='wrong', oid=5)
            self.seed(root, name='null', oid=None)
            self.seed(root, name='pred', table='pred')
            self.seed(root, name='other_csv', csv_hash='another-csv')
            unrelated = root / 'resource_manifest'; unrelated.mkdir()
            (unrelated / 'manifest.json').write_text('{"selection":"file/path.json"}')
            self.assertEqual(load_correct_cache(root, self.selection(), 'gt')['results'], {})

    def test_historical_correct_attempt_is_found_even_if_latest_attempt_failed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); old, _ = self.seed(root, nested=True)
            (old / 'q00010/evaluation.json').write_text(json.dumps(
                {**self.row(), 'predicted_id': None, 'correct': False}))
            self.assertIn('10', load_correct_cache(root, self.selection(), 'gt')['results'])

    def test_wrong_current_attempt_is_archived_and_success_stays_in_cache(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self.seed(root)
            output, wrong = self.seed(root, name='new', oid=None)
            cache = load_correct_cache(root, self.selection(), 'gt')
            audit = apply_correct_cache(cache, output, self.selection()['rows'])
            self.assertEqual(audit['replaced_rows'], [10])
            previous = output / 'q00010/superseded_by_correct_cache/evaluation.json'
            self.assertEqual(json.loads(previous.read_text()), wrong)
            cache_before = correct_cache_path(root, 'gt').read_bytes()
            self.assertFalse(remember_correct_result(cache, root, self.row(), wrong, output))
            self.assertEqual(correct_cache_path(root, 'gt').read_bytes(), cache_before)
            self.assertEqual(apply_correct_cache(cache, output, self.selection()['rows'])['replaced_rows'], [])

    def test_new_success_is_saved_immediately_and_survives_source_removal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); cache = load_correct_cache(root, self.selection(), 'gt')
            result = {**self.row(), 'predicted_id': 0, 'correct': True, 'implementation_version': 'new'}
            self.assertTrue(remember_correct_result(cache, root, self.row(), result, root / 'removed_run'))
            self.assertIn('10', json.loads(correct_cache_path(root, 'gt').read_text())['results'])
            self.assertIn('10', load_correct_cache(root, self.selection(), 'gt')['results'])

    def test_changed_query_or_wrong_target_does_not_reuse_correct_flag(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self.seed(root, row={**self.row(), 'query': 'another chair'})
            run, _ = self.seed(root, name='incorrect_flag', oid=5)
            path = run / 'q00010/evaluation.json'; data = json.loads(path.read_text())
            data['correct'] = True; path.write_text(json.dumps(data))
            self.assertEqual(load_correct_cache(root, self.selection(), 'gt')['results'], {})

    def test_two_runs_with_stale_cache_cannot_erase_each_others_success(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = load_correct_cache(root, self.selection(), 'gt')
            second = load_correct_cache(root, self.selection(), 'gt')
            for cache, index in ((first, 10), (second, 11)):
                row = self.row(index)
                remember_correct_result(cache, root, row,
                    {**row, 'correct': True, 'predicted_id': 0}, root / str(index))
            self.assertEqual(set(json.loads(correct_cache_path(root, 'gt').read_text())['results']),
                             {'10', '11'})

    def test_new_run_uses_correct_cache_without_model_calls_then_saves_new_success(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); scenes = root / 'scenes'; scenes.mkdir()
            self.seed(scenes)
            scene = scenes / 'scene0010_00'; scene.mkdir()
            for name in ('scene.npz', 'scene_instances.json', 'provenance.json'):
                (scene / name).write_text('{}')
            (scene / 'objects_gt.json').write_text('[{"bbox_id":0},{"bbox_id":5}]')
            selection = root / 'selection.json'; selection.write_text(json.dumps(self.selection()))
            output = scenes / 'new'
            argv = ['batch', '--manifest', str(selection), '--scene-root', str(scenes),
                    '--output', str(output), '--gt', '--workers', '1']
            def worker(command, **kwargs):
                self.assertEqual(command[command.index('--csv-row') + 1], '11')
                directory = Path(command[command.index('--output') + 1])
                (directory / 'prediction.json').write_text(json.dumps({'query': 'the chair',
                    'object_table': 'gt', 'mode': 'active_diagnostic', 'predicted_id': 0,
                    'implementation_version': 'relative-size-v7.3'}))
                return SimpleNamespace(returncode=0)
            def call(flags):
                with patch('sys.argv', argv + flags), patch('run_size_validation.ROOT', root), \
                        patch('run_size_validation.subprocess.run', side_effect=worker) as process, \
                        redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    main()
                return process.call_count
            self.assertEqual(call(['--prepare-only']), 0)
            self.assertEqual(call([]), 1)
            self.assertEqual(call(['--retry-false']), 0)
            cache = json.loads(correct_cache_path(scenes, 'gt').read_text())
            self.assertEqual(set(cache['results']), {'10', '11'})
            summary = json.loads((output / 'results.json').read_text())['summary']
            self.assertEqual(summary['correct'], 2)
            self.assertEqual(summary['evaluation_policy'], 'cumulative_ever_correct')


if __name__ == '__main__':
    unittest.main()
