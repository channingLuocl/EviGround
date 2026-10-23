import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from eviground.experiments.run_size_validation import evaluate, report, run_case, verify_source


class SizeValidationTests(unittest.TestCase):
    def row(self):
        return {'csv_row': 10, 'scene': 'scene0010_00', 'query': 'the tall shelf',
                'target_id': 13, 'stimulus_candidate_ids': [13, 5],
                'stage': 'simple_relative', 'size_form': 'adjective'}

    def test_worker_gets_no_answer_and_failed_process_is_persisted_and_retryable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch('eviground.experiments.run_size_validation.subprocess.run', return_value=Mock(returncode=1)) as proc:
                first = run_case(self.row(), root, root, 3)
                command = proc.call_args.args[0]
                self.assertNotIn('--target-id', command)
                self.assertNotIn('13', command)
                self.assertNotIn('5', command)
                self.assertEqual(run_case(self.row(), root, root, 3), first)
                self.assertEqual(proc.call_count, 1)
                run_case(self.row(), root, root, 3, retry_errors=True)
                self.assertEqual(proc.call_count, 2)
            self.assertFalse(first['correct'])
            summary = report(root, [first], 300)
            self.assertEqual(summary['errors'], 1)
            self.assertEqual(summary['accuracy'], 0.)

    def test_accepted_wrong_is_counted_and_cache_query_is_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction = root / 'prediction.json'
            data = {'query': self.row()['query'], 'mode': 'active_diagnostic',
                    'object_table': 'gt', 'predicted_id': 5, 'accepted': True,
                    'decision_status': 'resolved'}
            prediction.write_text(json.dumps(data))
            result = evaluate(self.row(), prediction, root / 'console.log')
            summary = report(root, [result], 300)
            self.assertEqual(summary['accepted_wrong'], 1)
            self.assertEqual(summary['accepted_accuracy'], 0.)
            data['query'] = 'another query'
            prediction.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
                evaluate(self.row(), prediction, root / 'console.log')

    def test_source_hash_check_can_use_manifest_when_csv_is_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'nr3d_official.csv'
            manifest = {'csv_sha256': 'expected'}
            self.assertEqual(verify_source(manifest, source), {
                'path': str(source), 'expected_sha256': 'expected', 'status': 'missing'})

    def test_source_hash_check_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'nr3d_official.csv'
            source.write_text('row\n')
            with self.assertRaisesRegex(ValueError, 'hash does not match'):
                verify_source(manifest={'csv_sha256': 'wrong'}, source=source)

    def test_source_hash_check_records_matching_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'nr3d_official.csv'
            source.write_text('row\n')
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(verify_source({'csv_sha256': expected}, source), {
                'path': str(source), 'expected_sha256': expected,
                'actual_sha256': expected, 'status': 'verified'})


if __name__ == '__main__':
    unittest.main()
