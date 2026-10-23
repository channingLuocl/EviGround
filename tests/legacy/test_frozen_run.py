"""A frozen experiment cannot be changed through batch, retry or inference entry points."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from seeground.grounding.real_scene import run_query
from seeground.api.qwen_api import api_request_slot
from seeground.experiments.run_size_validation import main, run_case
from seeground.evaluation.run_storage import ensure_run_writable
from seeground.evaluation.validation_cache import apply_correct_cache
from seeground.evaluation.validation_reporting import report
from seeground.evaluation.validation_reuse import reuse_results


class FrozenRunTests(unittest.TestCase):
    def fixture(self, root):
        run = root / 'v7_4_nr3d_val_full_gt'
        run.mkdir()
        (run / 'FROZEN.json').write_text('{"archive_id":"v7.4"}')
        row = {'csv_row': 10, 'scene': 'scene0010_00', 'query': 'the chair', 'target_id': 0}
        directory = run / 'q00010'
        directory.mkdir()
        (directory / 'evaluation.json').write_text(json.dumps(
            {**row, 'predicted_id': None, 'correct': False}))
        (run / 'results.json').write_text('{"correct":5435,"total":7805}')
        return run, row

    def contents(self, root):
        return {p.relative_to(root).as_posix(): p.read_bytes()
                for p in root.rglob('*') if p.is_file()}

    def test_nested_retry_and_symlink_cannot_bypass_freeze(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run, _ = self.fixture(root)
            alias = root / 'alias'
            alias.symlink_to(run, target_is_directory=True)
            for path in (run, run / 'q00010/null_retries/attempt_001', alias / 'q00010'):
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'Frozen run'):
                    ensure_run_writable(path)
            ensure_run_writable(root / 'new_run')

    def test_cli_stops_before_preparing_inputs_importing_cache_or_calling_worker(self):
        with tempfile.TemporaryDirectory() as td:
            run, _ = self.fixture(Path(td))
            before = self.contents(run)
            for flags in ([], ['--prepare-only'], ['--retry-false'], ['--worker']):
                with self.subTest(flags=flags), patch('sys.argv', [
                    'batch', '--manifest', 'does-not-exist.json', '--output', str(run), *flags]), \
                        patch('seeground.experiments.run_size_validation.load_correct_cache') as cache, \
                        patch('seeground.experiments.run_size_validation.subprocess.run') as process, \
                        self.assertRaisesRegex(ValueError, 'Frozen run'):
                    main()
                cache.assert_not_called()
                process.assert_not_called()
            self.assertEqual(self.contents(run), before)

    def test_retry_report_and_imports_preserve_every_existing_byte(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run, row = self.fixture(root)
            before = self.contents(run)
            actions = [lambda flag=flag: run_case(row, root, run, 3, **{flag: True})
                       for flag in ('retry_false', 'retry_null', 'retry_errors')]
            actions += [lambda: report(run, [], 7805),
                        lambda: apply_correct_cache({'results': {}}, run, []),
                        lambda: reuse_results(root / 'missing_source', run, [], {})]
            for action in actions:
                with self.assertRaisesRegex(ValueError, 'Frozen run'):
                    action()
            self.assertEqual(self.contents(run), before)

    def test_direct_inference_stops_before_loading_scene_or_writing_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run, _ = self.fixture(root)
            before = self.contents(run)
            with patch('seeground.grounding.real_scene.load_objects') as loader, \
                    self.assertRaisesRegex(ValueError, 'Frozen run'):
                run_query(root / 'missing_scene', 'the chair', run / 'q00010')
            loader.assert_not_called()
            self.assertEqual(self.contents(run), before)

    def test_request_slot_directory_cannot_write_inside_frozen_run(self):
        with tempfile.TemporaryDirectory() as td:
            run, _ = self.fixture(Path(td))
            before = self.contents(run)
            with patch.dict('os.environ', {'SEEGROUND_API_SLOT_DIR': str(run / '.api_slots')}), \
                    self.assertRaisesRegex(ValueError, 'Frozen run'):
                with api_request_slot():
                    self.fail('A frozen slot directory was accepted')
            self.assertFalse((run / '.api_slots').exists())
            self.assertEqual(self.contents(run), before)


if __name__ == '__main__':
    unittest.main()
