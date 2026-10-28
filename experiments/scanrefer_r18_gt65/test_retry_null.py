"""Offline checks: no API requests or real dataset writes."""
from pathlib import Path
import copy
import io
import tempfile
import unittest

from retry_state import save
from retry_null import check_inference_settings, merge, null_ids, snapshot
from retry_null_progress import CumulativeProgress


class NullRetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = [dict(csv_row=i, scene='synthetic', query='synthetic',
                          target_id=i, unique=i == 0) for i in range(4)]
        self.results = [{**r, 'predicted_id': p, 'iou_3d': iou, 'error': error}
                        for r, p, iou, error in zip(self.rows, (1, 2, None, None),
                                                   (.8, 0, 0, 0), (None, None, None, 'API'))]

    def test_null_selects_errors_and_abstentions_but_not_wrong_nonnull(self):
        self.assertEqual(null_ids(self.results), [2, 3])

    def settings(self):
        return {'api': {'api_slot_dir': '/old/.api_slots', 'api_concurrency': 8,
                        'request_options': {'enable_thinking': True}},
                'policy': {'budget': 3}, 'query_syntax': {'backend': 'spacy'},
                'models': {'SEEGROUND_MODEL': 'qwen3.7-plus'}}

    def test_output_lock_directory_can_differ(self):
        old = self.settings()
        new = copy.deepcopy(old)
        new['api']['api_slot_dir'] = '/new/round002/.api_slots'
        check_inference_settings(old, new)

    def test_concurrency_thinking_and_model_changes_are_rejected(self):
        old = self.settings()
        for section, key, value in [('api', 'api_concurrency', 4),
                                    ('api', 'request_options', {'enable_thinking': False}),
                                    ('models', 'SEEGROUND_MODEL', 'different-model')]:
            with self.subTest(section=section, key=key):
                new = copy.deepcopy(old)
                new[section][key] = value
                with self.assertRaisesRegex(ValueError, section):
                    check_inference_settings(old, new)

    def test_merge_preserves_every_unselected_record(self):
        changed = [{**self.results[2], 'predicted_id': 2, 'iou_3d': 1}, self.results[3]]
        merged = merge(self.results, changed, [2, 3])
        self.assertEqual(merged[:2], self.results[:2])
        self.assertEqual(merged[2:], changed)
        with self.assertRaises(ValueError):
            merge(self.results, changed[:1], [2, 3])
        with self.assertRaises(ValueError):
            merge(self.results, changed + changed[:1], [2, 3])

    def test_cumulative_display_includes_previous_true_false_errors_and_abstentions(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        progress = CumulativeProgress(2, stdout=stdout, stderr=stderr, baseline=self.results)
        self.assertEqual(progress.stats()['total'], 4)
        self.assertIn('Acc@0.25 1/4 = 25.00%', stderr.getvalue())
        progress.update({**self.results[2], 'predicted_id': 2, 'iou_3d': .3})
        stats = progress.stats()
        self.assertEqual((stats['total'], stats['correct25'], stats['correct50']), (4, 2, 1))
        self.assertEqual(stats['errors'], 1)
        self.assertIn('Acc@0.25 2/4 = 50.00%', stderr.getvalue())
        self.assertIn('Retry [', stderr.getvalue())
        self.assertIn('1/2', stderr.getvalue())

    def test_resumed_cached_retry_replaces_baseline_once_and_uses_iou(self):
        repaired = {**self.results[2], 'predicted_id': 2, 'iou_3d': .3, 'correct': False}
        progress = CumulativeProgress(2, stdout=io.StringIO(), stderr=io.StringIO(),
                                      initial_results=[repaired], baseline=self.results)
        self.assertEqual(progress.stats()['correct25'], 2)
        self.assertEqual(progress.stats()['total'], 4)
        with self.assertRaises(ValueError):
            progress.update(repaired)

    def test_interrupted_false_snapshot_uses_latest_completed_and_original_unrun(self):
        archive = self.root / 'retry_archive/round001'
        archive.mkdir(parents=True)
        save(archive / 'results.json', {'results': self.results})
        save(self.root / 'results.json', {'results': self.results[:2]})
        save(self.root / 'retry_pending.json', {'round': 1, 'archive': str(archive), 'failed_ids': [1, 2, 3]})
        for row in self.results[1:]:
            directory = archive / f"q{row['csv_row']:05d}"
            directory.mkdir()
            save(directory / 'evaluation.json', row)
        for row in (self.results[0], {**self.results[2], 'predicted_id': 2, 'iou_3d': 1}):
            directory = self.root / f"q{row['csv_row']:05d}"
            directory.mkdir()
            save(directory / 'evaluation.json', row)
        observed, origin = snapshot(self.root, self.rows)
        self.assertEqual(len(observed), 4)
        self.assertEqual(null_ids(observed), [3])
        self.assertEqual(observed[1], self.results[1])
        self.assertEqual(origin['previous_rounds_used'], 1)
        self.assertTrue((archive / 'q00001/evaluation.json').exists())


if __name__ == '__main__':
    unittest.main()
