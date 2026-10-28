"""Isolated synthetic checks of evaluation and archival; never model predictions."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from contextlib import redirect_stdout
import io
from retry_state import prepare, finalize, save, success
from scanrefer_metrics import report, accounting


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='gt65-synthetic-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / 'selection.json'
        self.rows = [dict(csv_row=i, scene='synthetic_test', query='SYNTHETIC TEST',
                          target_id=i, unique=i==0) for i in range(3)]
        save(self.manifest, {'dataset': 'scanrefer', 'rows': self.rows})
        self.results = [{**r, 'predicted_id': r['target_id'], 'iou_3d': iou,
                         'correct': False, 'error': None} for r, iou in zip(self.rows, (.3, .1, 0))]
        for r in self.results:
            q = self.root / f"q{r['csv_row']:05d}"
            q.mkdir()
            save(q / 'evaluation.json', r)
        save(self.root / 'results.json', {'summary': {}, 'results': self.results})

    def test_retry_selects_iou_not_instance_id_and_resumes(self):
        with redirect_stdout(io.StringIO()):
            prepare(self.root, self.manifest, max_rounds=1)
            prepare(self.root, self.manifest, max_rounds=1)
        state = json.loads((self.root / 'retry_pending.json').read_text())
        self.assertEqual(state['failed_ids'], [1, 2])
        self.assertTrue((self.root / 'q00000/evaluation.json').exists())
        self.assertTrue((Path(state['archive']) / 'q00001/evaluation.json').exists())
        for i in (1, 2):
            self.results[i]['iou_3d'] = .6 if i == 1 else 0
            q = self.root / f'q{i:05d}'
            q.mkdir()
            save(q / 'evaluation.json', self.results[i])
        save(self.root / 'results.json', {'summary': {}, 'results': self.results})
        with redirect_stdout(io.StringIO()):
            finalize(self.root, self.manifest)
        h = json.loads((self.root / 'retry_history.json').read_text())
        self.assertEqual(h['rounds'][0]['newly_correct'], 1)
        with self.assertRaises(ValueError):
            prepare(self.root, self.manifest, max_rounds=1)

    def test_incomplete_round_cannot_retry(self):
        save(self.root / 'results.json', {'summary': {}, 'results': self.results[:2]})
        with self.assertRaises(ValueError):
            prepare(self.root, self.manifest, max_rounds=1)

    def test_errors_abstentions_and_pending_stay_in_denominator(self):
        self.results[0]['error'] = 'synthetic API error'
        self.results[1]['predicted_id'] = None
        self.results[1]['iou_3d'] = .8
        save(self.root / 'results.json', {'summary': {}, 'results': self.results[:2]})
        args = SimpleNamespace(manifest=self.manifest, results=self.root/'results.json',
                               proposals='gt', output=self.root/'metrics.json')
        with redirect_stdout(io.StringIO()):
            report(args)
        m = json.loads(args.output.read_text())
        self.assertEqual(m['overall']['total'], 3)
        self.assertEqual(m['overall']['correct25'], 0)
        self.assertEqual(m['overall']['errors'], 1)
        self.assertEqual(m['overall']['abstentions'], 1)
        self.assertEqual(m['pending'], 1)
        self.assertFalse(success(self.results[0]))
        self.assertFalse(success(self.results[1]))

    def test_durable_ledger_counts_interruption_and_avoids_replay_double_count(self):
        q = self.root / 'q00000'
        save(q / 'run.json', {'calls': [{'usage': {'total_tokens': 999}}]})
        events = [{'event': 'start', 'attempt_id': 'a'},
                  {'event': 'end', 'attempt_id': 'a', 'usage': {'total_tokens': 7}},
                  {'event': 'start', 'attempt_id': 'interrupted'}]
        (q / 'api_attempts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events))
        result = accounting(self.root)
        self.assertEqual(result['api_attempts'], 2)
        self.assertEqual(result['total_tokens'], 7)
        self.assertEqual(result['attempts_without_token_usage'], 1)


if __name__ == '__main__':
    unittest.main()
