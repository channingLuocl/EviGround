"""Offline checks for the all-false selection, cache, merge, and launch guard."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import retry_all_false_gpt61 as entry
from retry_state import success


class AllFalseGPT61Tests(unittest.TestCase):
    def test_all_failure_types_are_selected_without_prior_answer_reuse(self):
        rows = [
            {'csv_row': 4, 'predicted_id': 1, 'iou_3d': .9},
            {'csv_row': 1, 'predicted_id': 2, 'iou_3d': .1},
            {'csv_row': 3, 'predicted_id': None, 'iou_3d': 0},
            {'csv_row': 2, 'predicted_id': None, 'iou_3d': 0, 'error': 'api'},
        ]
        self.assertEqual(entry.selected_ids(rows), [1, 2, 3])
        self.assertTrue(success(rows[0]))

    def test_current_completed_baseline_has_exactly_3377_false(self):
        source = entry.read(entry.SOURCE/'results.json')['results']
        completion = entry.read(entry.SOURCE/'completion.json')
        self.assertEqual(entry.digest(entry.SOURCE/'results.json'), completion['results_sha256'])
        self.assertEqual(len(source), 9508)
        self.assertEqual(sum(map(success, source)), 6131)
        self.assertEqual(len(entry.selected_ids(source)), 3377)

    def test_one_round_replaces_every_outcome_and_preserves_successes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            round_dir = output/'round001'
            baseline = [{'csv_row': 0, 'predicted_id': 9, 'iou_3d': 1},
                        {'csv_row': 1, 'predicted_id': 2, 'iou_3d': 0},
                        {'csv_row': 2, 'predicted_id': None, 'iou_3d': 0}]
            replacements = [{'csv_row': 1, 'predicted_id': None, 'iou_3d': 0, 'error': 'new error'},
                            {'csv_row': 2, 'predicted_id': 3, 'iou_3d': 1}]
            def fake_run(command):
                self.assertIn('retry_all_false_gpt61_progress.py', command[2])
                self.assertEqual(command[command.index('--manifest')+1], str(output/'selection.json'))
                self.assertNotIn('--target-id', command)
                self.assertNotIn('--prior-answer', command)
                entry.save(round_dir/'results.json', {'results': replacements})
                return SimpleNamespace(returncode=0)
            with (patch.object(entry, 'OUTPUT', output), patch.object(entry, 'ROUND', round_dir),
                  patch.object(entry.subprocess, 'run', side_effect=fake_run) as worker):
                entry.run_visual({'selected_ids': [1, 2]}, baseline)
                worker.assert_called_once()
            final = entry.read(output/'results.json')['results']
            self.assertEqual(final[0], baseline[0])
            self.assertEqual(final[1:], replacements)
            self.assertEqual(entry.read(output/'completion.json')['selected_failed'], 2)

    def test_incomplete_round_does_not_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            round_dir = output/'round001'
            def fake_run(command):
                entry.save(round_dir/'results.json', {'results': [{'csv_row': 1}]})
                return SimpleNamespace(returncode=0)
            with (patch.object(entry, 'OUTPUT', output), patch.object(entry, 'ROUND', round_dir),
                  patch.object(entry.subprocess, 'run', side_effect=fake_run)):
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    entry.run_visual({'selected_ids': [1, 2]},
                        [{'csv_row': 1}, {'csv_row': 2}])
                self.assertFalse((output/'completion.json').exists())

    def test_interrupted_unevaluated_case_is_not_silently_restarted(self):
        with tempfile.TemporaryDirectory() as directory:
            round_dir = Path(directory)
            (round_dir/'q00001').mkdir()
            with patch.object(entry, 'ROUND', round_dir), patch.object(entry.subprocess, 'run') as launch:
                with self.assertRaisesRegex(ValueError, 'Interrupted trajectory'):
                    entry.run_visual({'selected_ids': [1]}, [])
                launch.assert_not_called()

    def test_preflight_and_completed_run_do_not_launch_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            entry.save(output/'manifest.json', {'rows': []})
            entry.save(output/'results.json', {'results': []})
            entry.save(output/'completion.json', {'results_sha256': entry.digest(output/'results.json')})
            settings = {'api': {}, 'models': {}, 'policy': {}, 'query_syntax': {}}
            protocol = {'selected_ids': [1], 'baseline_correct25': 6131, 'runtime_settings': settings}
            for action in ('preflight', 'run'):
                with (patch.object(entry, 'OUTPUT', output),
                      patch.object(entry, 'verify', return_value=({}, output/'manifest.json')),
                      patch.object(entry, 'checked', return_value=(protocol, [])),
                      patch.object(entry, 'validate_new_settings', return_value=settings),
                      patch.object(entry, 'validate'), patch.object(entry, 'run_visual') as launch,
                      patch.object(entry, 'report_visual'),
                      patch('sys.argv', ['retry_all_false_gpt61.py', action]),
                      redirect_stdout(io.StringIO())):
                    entry.main()
                    launch.assert_not_called()

    def test_credential_is_not_embedded_in_entry_or_rules(self):
        key = dict(line.split('=', 1) for line in entry.CREDENTIAL.read_text().splitlines())['SEEGROUND_API_KEY']
        for name in entry.ENTRIES:
            self.assertNotIn(key, (entry.ROOT/name).read_text())
        self.assertEqual(entry.CREDENTIAL.stat().st_mode & 0o077, 0)


if __name__ == '__main__':
    unittest.main()
