"""Offline regressions for narrow v5 attachment coverage; saved r3 geometry fixtures."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import query_syntax
from active_diagnostic import size_candidate_evidence
from diagnostic import size_conditions

PROJECT = Path(__file__).resolve().parents[1]
SAVED = PROJECT / 'local_outputs/v5_ablation_scope_r3'
frozen_spec = importlib.util.spec_from_file_location('scope_attachment_frozen_r3',
    PROJECT / 'local_outputs/v5_scope_attachment_fixtures/query_syntax_frozen_r3.py')
frozen = importlib.util.module_from_spec(frozen_spec)
frozen_spec.loader.exec_module(frozen)


class V5ScopeAttachmentTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {'SEEGROUND_V4': '1',
            'SEEGROUND_V5': '1', 'SEEGROUND_V5_SCOPE': '1',
            'SEEGROUND_V5_MEASUREMENTS': '0', 'SEEGROUND_V5_CALL_ECONOMY': '0',
            'SEEGROUND_API_PROVIDER': 'qwen', 'SEEGROUND_REACT_BACKEND': 'qwen'})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        for name in ('requests.sessions.Session.request', 'subprocess.Popen'):
            blocker = patch(name, side_effect=AssertionError('offline external call blocked'))
            blocker.start()
            self.addCleanup(blocker.stop)

    def parsed(self, query, category, word, proposed=True):
        return {'source_query': query, 'Target': [category], 'Anchor': [],
            'Attributes': [{'category': category, 'role': 'target', 'phrase': word}],
            'Relations': [], 'Observer': {}, 'Unhandled': [],
            'query_syntax': (query_syntax if proposed else frozen).analyze_query(query)}

    def assert_ambiguous(self, query, category, word, quote):
        parsed = self.parsed(query, category, word)
        hint = query_syntax.comparison_hint(parsed, word, category)
        self.assertIsNotNone(hint)
        self.assertEqual(hint['attachment'], 'ambiguous')
        self.assertIn(quote, [m['text'] for m in hint['modifiers']])
        for modifier in hint['modifiers']:
            self.assertEqual(query[modifier['start']:modifier['end']], modifier['text'])
        condition = size_conditions(parsed, category)[0]
        self.assertEqual([h['kind'] for h in condition['scope_hypotheses']],
                         ['full_category', 'relation_subset'])
        self.assertLessEqual(len(condition['scope_hypotheses']), 2)
        return parsed

    def test_original_queries_retain_unbound_relation_alternative(self):
        cases = [(3299, 'box', 'small', 'next to the spotted chair'),
                 (19371, 'book', 'small', 'on the table'),
                 (23329, 'mirror', 'larger', 'next to the door'),
                 (27408, 'trash can', 'small', 'by the door')]
        for row, category, word, quote in cases:
            with self.subTest(row=row):
                run = json.loads((SAVED / f'q{row:05d}/run.json').read_text())
                parsed = deepcopy(run['parsed_query'])
                parsed = {key: value for key, value in parsed.items() if not key.startswith('_')}
                baseline = deepcopy(parsed)
                baseline['query_syntax'] = frozen.analyze_query(run['query'])
                self.assertFalse(size_conditions(baseline, category)[0].get('scope_hypotheses'))
                proposed = self.assert_ambiguous(run['query'], category, word, quote)
                parsed['query_syntax'] = proposed['query_syntax']
                size_step = next(step for step in run['active_diagnostic']['steps']
                                 if step.get('action', {}).get('dimension') == 'size')
                metrics = [item['metrics'] for item in size_step['observation']['candidate_evidence']]
                objects = [{'bbox_id': m['id'], 'target': category} for m in metrics]
                old_records, _, _ = size_candidate_evidence(baseline, objects, metrics,
                    comparison_scope_bound=True, objects=objects)
                records, constraints, _ = size_candidate_evidence(parsed, objects, metrics,
                    comparison_scope_bound=True, objects=objects)
                self.assertTrue(any(record['status'] == 'fail' for record in old_records))
                self.assertEqual({record['status'] for record in records}, {'unknown'})
                self.assertEqual(constraints[0]['status'], 'unknown')
                self.assertEqual(constraints[0]['selected_ids'], [])
                self.assertEqual(len(constraints[0]['scope_hypotheses']), 2)

    def test_paraphrases_and_adverb_levels(self):
        for query, category, word, quote in [
            ('the small box directly next to the chair', 'box', 'small', 'directly next to the chair'),
            ('the small box right next to the chair', 'box', 'small', 'right next to the chair'),
            ('Find the larger mirror next to the door.', 'mirror', 'larger', 'next to the door'),
            ('Pick the small book on the desk.', 'book', 'small', 'on the desk'),
            ('the smaller trash bin by the cabinet', 'trash bin', 'smaller', 'by the cabinet')]:
            with self.subTest(query=query):
                self.assert_ambiguous(query, category, word, quote)

    def test_independent_clauses_and_unrelated_verbs_do_not_gain_subset(self):
        for query, category, word in [
            ('This small mirror is alongside the white cabinet.', 'mirror', 'small'),
            ('The small book is on the table.', 'book', 'small'),
            ('Read the small book on the table.', 'book', 'small'),
            ('The small box.', 'box', 'small')]:
            with self.subTest(query=query):
                parsed = self.parsed(query, category, word)
                self.assertFalse(size_conditions(parsed, category)[0].get('scope_hypotheses'))

    def test_imperative_action_and_explicit_time_pps_are_not_comparison_members(self):
        for query, category, word in [
            ('Find the small book by searching the shelf.', 'book', 'small'),
            ('Choose the smaller mirror on Friday.', 'mirror', 'smaller'),
            ('Pick the small book by inspecting the desk.', 'book', 'small'),
            ('Select the larger mirror on Monday morning.', 'mirror', 'larger'),
            ('Pick the smaller can by noon.', 'can', 'smaller')]:
            with self.subTest(query=query):
                parsed = self.parsed(query, category, word)
                hint = query_syntax.comparison_hint(parsed, word, category)
                self.assertEqual(hint['attachment'], 'independent')
                self.assertEqual(hint['modifiers'], [])
                self.assertFalse(size_conditions(parsed, category)[0].get('scope_hypotheses'))
        self.assert_ambiguous('Find the small book by the shelf.', 'book', 'small', 'by the shelf')
        self.assert_ambiguous('Pick the small book on the desk.', 'book', 'small', 'on the desk')

    def test_explicit_counted_set_remains_one_restrictive_interpretation(self):
        query = ('There are two chairs facing the square white objects. '
                 'The target object is the taller chair to the left.')
        parsed = self.parsed(query, 'chair', 'taller')
        condition = size_conditions(parsed, 'chair')[0]
        self.assertEqual(condition['required_member_count'], 2)
        self.assertEqual(condition['scope_kind'], 'relation_subset')
        self.assertFalse(condition.get('scope_hypotheses'))

    def test_closed_v5_scope_preserves_every_saved_v4_syntax_output(self):
        manifest = json.loads((SAVED / 'manifest.json').read_text())
        for row in manifest['selection']['rows']:
            query = row['query']
            with self.subTest(row=row['csv_row']):
                self.assertEqual(query_syntax._analyze(query, False), frozen._analyze(query, False))
        with patch.dict(os.environ, {'SEEGROUND_V5_SCOPE': '0'}):
            self.assertEqual(query_syntax.analyze_query('small trash can by the door.'),
                             frozen.analyze_query('small trash can by the door.'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
