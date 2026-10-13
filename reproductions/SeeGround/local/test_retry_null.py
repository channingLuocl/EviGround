"""Manual null reruns keep earlier attempts and never rerun existing answers."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from run_size_validation import main, run_case


class RetryNullTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'SEEGROUND_V4': '1', 'SEEGROUND_V5': '1',
            'SEEGROUND_V6': '0', 'SEEGROUND_V62': '0', 'SEEGROUND_API_PROVIDER': 'qwen',
            'SEEGROUND_REACT_BACKEND': 'qwen', 'SEEGROUND_MODEL': 'offline-mock',
            'SEEGROUND_BASE_URL': 'https://example.invalid/v1', 'SEEGROUND_API_KEY': 'offline-only',
            'SEEGROUND_API_SLOT_DIR': '', 'SEEGROUND_SYNTAX_BACKEND': 'none'})
        env.start()
        self.addCleanup(env.stop)

    def row(self, index=10):
        return {'csv_row': index, 'scene': 'scene0010_00', 'query': 'the chair',
                'target_id': 13, 'stimulus_candidate_ids': [13, 5]}

    def prediction(self, oid):
        return {'query': 'the chair', 'mode': 'active_diagnostic', 'object_table': 'gt',
                'predicted_id': oid, 'api_attempts': 1}

    def seed(self, root, oid=None, **extra):
        directory = root / 'q00010'
        directory.mkdir()
        (directory / 'prediction.json').write_text(json.dumps(self.prediction(oid)))
        (directory / 'evaluation.json').write_text(json.dumps({**self.row(),
            'predicted_id': oid, 'correct': oid == 13, **extra}))
        (directory / 'console.log').write_text('original log\n')
        return directory

    def worker(self, oid):
        def complete(command, **kwargs):
            self.assertNotIn('--target-id', command)
            directory = Path(command[command.index('--output') + 1])
            self.assertFalse((directory / 'prediction.json').exists())
            self.assertFalse((directory / 'pending_inference.json').exists())
            (directory / 'prediction.json').write_text(json.dumps(self.prediction(oid)))
            return SimpleNamespace(returncode=0)
        return complete

    def test_normal_abstention_gets_new_attempt_and_previous_files_are_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); original = self.seed(root)
            with patch('run_size_validation.subprocess.run', side_effect=self.worker(13)) as worker:
                result = run_case(self.row(), root, root, 3, retry_null=True)
            self.assertEqual(worker.call_count, 1)
            self.assertEqual(result['predicted_id'], 13)
            self.assertTrue(result['correct'])
            self.assertEqual(json.loads((original / 'prediction.json').read_text())['predicted_id'], None)
            self.assertEqual((original / 'console.log').read_text(), 'original log\n')
            self.assertEqual(json.loads((original / 'evaluation.json').read_text()), result)
            self.assertTrue((original / 'null_retries/attempt_001/previous_evaluation.json').exists())
            self.assertIn('null_retries/attempt_001', result['prediction_path'])

    def test_exhausted_timeout_gets_fresh_attempt_without_touching_old_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); original = self.seed(root, error='timeout', recovery_budget_exhausted=True)
            (original / 'prediction.json').unlink()
            (original / 'pending_inference.json').write_text('old exhausted checkpoint')
            (original / 'api_recovery.json').write_text('old exhausted budget')
            with patch('run_size_validation.subprocess.run', side_effect=self.worker(13)):
                result = run_case(self.row(), root, root, 3, retry_null=True)
            self.assertEqual(result['predicted_id'], 13)
            self.assertEqual((original / 'pending_inference.json').read_text(), 'old exhausted checkpoint')

    def test_existing_wrong_or_review_failed_answer_is_not_retried(self):
        for oid in (0, 5, 13):
            with self.subTest(oid=oid), tempfile.TemporaryDirectory() as td:
                root = Path(td); self.seed(root, oid, review_failed=True, error='review failure')
                with patch('run_size_validation.subprocess.run') as worker:
                    result = run_case(self.row(), root, root, 3, retry_null=True)
                worker.assert_not_called()
                self.assertEqual(result['predicted_id'], oid)

    def test_repeated_null_attempts_are_kept_and_worker_failure_updates_current_result(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); original = self.seed(root)
            with patch('run_size_validation.subprocess.run', side_effect=self.worker(None)):
                first = run_case(self.row(), root, root, 3, retry_null=True)
            with patch('run_size_validation.subprocess.run', return_value=SimpleNamespace(returncode=1)):
                second = run_case(self.row(), root, root, 3, retry_null=True)
            self.assertTrue(second['inference_error'])
            self.assertEqual(json.loads((original / 'null_retries/attempt_001/evaluation.json').read_text()), first)
            self.assertEqual(json.loads((original / 'null_retries/attempt_002/previous_evaluation.json').read_text()), first)
            self.assertEqual(json.loads((original / 'evaluation.json').read_text()), second)

    def fixture(self, root):
        scene = root / 'scenes/scene0010_00'
        scene.mkdir(parents=True)
        for name in ('scene.npz', 'scene_instances.json', 'provenance.json'):
            (scene / name).write_text('{}')
        (scene / 'objects_gt.json').write_text('[{"bbox_id":13},{"bbox_id":5}]')
        selection = root / 'selection.json'
        selection.write_text(json.dumps({'rows': [self.row(10), self.row(11), self.row(12)], 'csv_sha256': 'offline'}))
        return ['run_size_validation.py', '--manifest', str(selection), '--scene-root', str(root / 'scenes'),
                '--output', str(root / 'output'), '--workers', '1', '--limit', '2']

    def call_main(self, root, argv, worker):
        with patch('sys.argv', argv), patch('run_size_validation.ROOT', root), \
                patch('run_size_validation.subprocess.run', side_effect=worker), \
                redirect_stdout(StringIO()) as stdout, redirect_stderr(StringIO()):
            main()
        return stdout.getvalue()

    def test_main_retries_only_completed_nulls_preserves_summary_and_accepts_runner_only_change(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); argv = self.fixture(root)
            def initial(command, **kwargs):
                directory = Path(command[command.index('--output') + 1])
                oid = 13 if directory.name == 'q00010' else None
                (directory / 'prediction.json').write_text(json.dumps(self.prediction(oid)))
                return SimpleNamespace(returncode=0)
            self.call_main(root, argv, initial)
            output = root / 'output'
            manifest = output / 'manifest.json'
            saved = json.loads(manifest.read_text())
            saved['code_sha256']['run_size_validation.py'] = 'previous batch orchestrator hash'
            manifest.write_text(json.dumps(saved))
            original_manifest = manifest.read_text()
            # Remove --limit so the third, never-run row is visible but skipped.
            retry_argv = argv[:-2] + ['--retry-null']
            seen = []
            def retry(command, **kwargs):
                seen.append(command[command.index('--csv-row') + 1])
                return self.worker(13)(command, **kwargs)
            stdout = self.call_main(root, retry_argv, retry)
            self.assertEqual(seen, ['11'])
            self.assertEqual(len(stdout.splitlines()), 1)
            self.assertEqual(manifest.read_text(), original_manifest)
            result = json.loads((output / 'results.json').read_text())
            self.assertEqual(result['summary']['total'], 2)
            self.assertEqual(result['summary']['correct'], 2)
            self.assertFalse((output / 'q00012').exists())
            self.assertEqual(len(list((output / 'null_retry_runs').glob('*/manifest.json'))), 1)
            # A second retry with no nulls sends no worker requests.
            with patch('run_size_validation.subprocess.run') as worker, patch('sys.argv', retry_argv), \
                    patch('run_size_validation.ROOT', root), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                main()
            worker.assert_not_called()

    def test_main_rejects_changed_inference_code_before_any_retry(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); argv = self.fixture(root)
            self.call_main(root, argv, self.worker(None))
            manifest = root / 'output/manifest.json'
            saved = json.loads(manifest.read_text()); saved['code_sha256']['qwen_api.py'] = 'different inference code'
            manifest.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, 'different code/settings'):
                self.call_main(root, argv + ['--retry-null'], self.worker(13))


if __name__ == '__main__':
    unittest.main()
