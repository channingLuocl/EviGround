"""Offline full-container sharing and repeated planner regressions."""
from eviground.paths import ROOT
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import eviground.grounding.react_decision as PROPOSED
import tests.legacy.test_v5_context_protocol as prior_tests
from tests.legacy.test_v5_context_protocol import expand_protocol

OUT = ROOT / 'local_outputs/v5_context_protocol_fixtures'
spec = importlib.util.spec_from_file_location('frozen_r4_before_all_containers',
    OUT / 'react_decision_frozen_r4.py')
FROZEN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FROZEN)


class ContextAllContainersTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {'EVIGROUND_V5': '1', 'EVIGROUND_V4': '1',
            'EVIGROUND_CONTEXT_LIMIT': '32768', 'EVIGROUND_REACT_BACKEND': 'qwen',
            'EVIGROUND_API_PROVIDER': 'qwen'})
        environment.start()
        self.addCleanup(environment.stop)

    def test_new_r4_real_failure_fits_same_budget_without_obligation_loss(self):
        saved = json.loads((OUT / 'context_r4_q05902_failed_state.json').read_text())
        saved = saved['5902']
        instruction_bytes = len(saved['instructions'].encode())
        with self.assertRaisesRegex(ValueError, 'required planner obligations'):
            FROZEN.planner_state(saved['state'], instruction_bytes=instruction_bytes)
        encoded, audit = PROPOSED.planner_state(saved['state'], instruction_bytes=instruction_bytes)
        with patch.dict(os.environ, {'EVIGROUND_CONTEXT_LIMIT': '1000000000'}):
            baseline, _ = FROZEN.planner_state(saved['state'], tighter=audit['tighter'],
                                              instruction_bytes=instruction_bytes)
        self.assertEqual(expand_protocol(encoded), baseline)
        self.assertEqual(audit['context_limit'], 32768)
        self.assertLessEqual(audit['input_units'], 30720)
        self.assertEqual(audit['instruction_bytes'], instruction_bytes)
        self.assertEqual(encoded['parsed_query']['condition_records'], baseline['parsed_query']['condition_records'])
        self.assertEqual(encoded['evidence_state']['missing'], baseline['evidence_state']['missing'])
        self.assertEqual(encoded['observation']['next_evidence'], baseline['observation']['next_evidence'])
        self.assertEqual(PROPOSED.evidence_tasks(encoded), PROPOSED.evidence_tasks(baseline))
        self.assertEqual(expand_protocol(PROPOSED.jev_state(encoded)), PROPOSED.jev_state(baseline))

    def test_three_r4_exact_failures_fit_or_explicitly_fail_without_obligation_loss(self):
        states = json.loads((OUT / 'context_r4_three_failed_states.json').read_text())
        self.assertEqual(set(states), {'5902', '22872', '30296'})
        for row, saved in states.items():
            with self.subTest(csv_row=row):
                instruction_bytes = len(saved['instructions'].encode())
                with self.assertRaisesRegex(ValueError, 'required planner obligations'):
                    FROZEN.planner_state(saved['state'], instruction_bytes=instruction_bytes)
                if row == '22872':
                    with self.assertRaisesRegex(ValueError, 'required planner obligations'):
                        PROPOSED.planner_state(saved['state'], instruction_bytes=instruction_bytes)
                    with patch.dict(os.environ, {'EVIGROUND_CONTEXT_LIMIT': '1000000000'}):
                        baseline, _ = FROZEN.planner_state(saved['state'], tighter=True,
                                                          instruction_bytes=instruction_bytes)
                    encoded, _ = PROPOSED.compact_repeated_evidence(baseline)
                    required = len(json.dumps(encoded, ensure_ascii=False, separators=(',', ':')).encode()) + instruction_bytes + 64
                    self.assertGreater(required, 30720)
                else:
                    encoded, audit = PROPOSED.planner_state(saved['state'], instruction_bytes=instruction_bytes)
                    self.assertLessEqual(audit['input_units'], 30720)
                    with patch.dict(os.environ, {'EVIGROUND_CONTEXT_LIMIT': '1000000000'}):
                        baseline, _ = FROZEN.planner_state(saved['state'], tighter=audit['tighter'],
                                                          instruction_bytes=instruction_bytes)
                self.assertEqual(expand_protocol(encoded), baseline)
                self.assertEqual(encoded['parsed_query']['condition_records'], baseline['parsed_query']['condition_records'])
                self.assertEqual(encoded['evidence_state']['missing'], baseline['evidence_state']['missing'])
                self.assertEqual(PROPOSED.evidence_tasks(encoded), PROPOSED.evidence_tasks(baseline))
                self.assertEqual(expand_protocol(PROPOSED.jev_state(encoded)), PROPOSED.jev_state(baseline))
                with patch.dict(os.environ, {'EVIGROUND_CONTEXT_LIMIT': '1000000000'}):
                    second, _ = PROPOSED.planner_state(encoded, tighter=True, instruction_bytes=instruction_bytes)
                expected = {**baseline, 'history': baseline['history'][-1:]}
                self.assertEqual(expand_protocol(second), expected)

    def test_cross_container_proofs_share_and_preserve_every_failure_and_unknown(self):
        source = prior_tests.ContextProtocolTests().synthetic_state()
        proof = {'status': 'unknown', 'candidate_ids': [12, 19], 'anchor_ids': [5, 7],
            'required_member_count': 2, 'unknown_member_ids': [19], 'target_role': 'target',
            'reference_role': 'support', 'source_quote': 'the small chair near the two doors',
            'reason': 'one quoted member and one support reference remain unresolved'}
        source['parsed_query']['_v4_relation_evidence'] = [deepcopy(proof)] * 4
        source['evidence_inventory'] = {'items': [deepcopy(proof)] * 4}
        source['deletion_events'].append({'event_id': 9, 'candidate_id': 12,
            'proofs': [deepcopy(proof)] * 4})
        encoded, _ = PROPOSED.compact_repeated_evidence(source)
        self.assertEqual(expand_protocol(encoded), source)
        self.assertIn('shared_evidence_ref', json.dumps(encoded['parsed_query']['_v4_relation_evidence']))
        self.assertEqual(expand_protocol(PROPOSED.jev_state(encoded)), PROPOSED.jev_state(source))
        self.assertEqual(encoded['parsed_query']['condition_records'], source['parsed_query']['condition_records'])
        self.assertEqual(encoded['evidence_state']['missing'], source['evidence_state']['missing'])

    def test_repeated_distance_rankings_keep_jev_rankings_and_ignored(self):
        rankings = {'rankings': [{'relation': 'nearest', 'anchor_id': 5,
            'best_ids': [12], 'ordered_targets': [12, 19]}],
            'ignored': [{'relation': 'near', 'reason': 'near is not a nearest ranking'}]}
        source = {'observation': {'details': {'distance_rankings': deepcopy(rankings)}},
            'auxiliary': deepcopy(rankings)}
        encoded, _ = PROPOSED.compact_repeated_evidence(source)
        self.assertEqual(encoded['observation']['details']['distance_rankings'], rankings)
        self.assertEqual(expand_protocol(encoded), source)
        self.assertEqual(expand_protocol(PROPOSED.jev_state(encoded)), PROPOSED.jev_state(source))

    def test_shared_reference_in_table_cell_roundtrips_recursively(self):
        proof = {'status': 'fail', 'source_quote': 'near the door',
            'reason': 'visible separation under every compatible anchor ' * 4}
        source = {'observation': {'candidate_evidence': [
            {'candidate_id': 2, 'proof': deepcopy(proof)},
            {'candidate_id': 3, 'proof': {'status': 'unknown', 'reason': 'anchor unresolved'}},
            {'candidate_id': 4, 'proof': deepcopy(proof)}]}}
        encoded, _ = PROPOSED.compact_repeated_evidence(source)
        table = encoded['observation']['candidate_evidence']
        self.assertIn('table_rows', table)
        self.assertIn('shared_evidence_ref', json.dumps(table['table_rows']))
        self.assertEqual(expand_protocol(encoded), source)
        self.assertEqual(expand_protocol(PROPOSED.jev_state(encoded)), PROPOSED.jev_state(source))

    def test_executable_task_containers_stay_direct_even_when_structures_repeat(self):
        source = prior_tests.ContextProtocolTests().synthetic_state()
        source.update({'target_table': [{'bbox_id': 12, 'category': 'chair'}],
            'reference_table': [{'bbox_id': 5, 'category': 'door'}, {'bbox_id': 7, 'category': 'door'}],
            'category_candidates': [12, 19], 'remaining_candidate_ids': [12, 19],
            'available_dimensions': ['appearance', 'position'],
            'evidence_capabilities': {'position': 'verify every quoted support member'},
            'tools_remaining': 1, 'history': [{'observation': deepcopy(source['observation']),
                'action': {'action': 'evidence', 'dimension': 'position'},
                'remaining_candidate_ids': [12, 19]}]})
        encoded, _ = PROPOSED.compact_repeated_evidence(source)
        self.assertEqual(PROPOSED.evidence_tasks(encoded), PROPOSED.evidence_tasks(source))
        self.assertEqual(expand_protocol(PROPOSED.jev_state(encoded)), PROPOSED.jev_state(source))
        self.assertEqual(expand_protocol(encoded), source)

    def test_second_planner_and_twice_recovered_messages_keep_negative_proofs_and_missing(self):
        negative = {'candidate_id': 2, 'status': 'fail', 'source_quote': 'near the door',
            'reason': 'visible gap under every compatible support anchor ' * 4,
            'explicit_fail_criterion': 'visible separation', 'role_binding_status': 'bound',
            'comparison_scope_status': 'complete', 'comparison_scope_ids': [2],
            'anchor_hypotheses': [{'status': 'unknown', 'anchor_ids': [8],
                'reason': 'complete reference binding not established ' * 4}] * 2,
            'scope_hypotheses': [{'status': 'unknown', 'required_member_count': 2,
                'comparison_pool_ids': [2, 4], 'scope_unknown_ids': [4],
                'reason': 'one counted member has unresolved relation evidence ' * 4,
                'member_binding': [{'candidate_id': 4, 'status': 'unknown',
                    'reason': 'unknown member retained for the quoted pair ' * 4}] * 2}] * 2}
        event = {'event_id': 1, 'candidate_id': 2, 'dimension': 'position',
            'source_quotes': ['near the door'], 'invalidated': False,
            'evidence': [deepcopy(negative)] * 2}
        source = {'query': 'the small chair near the door', 'parsed_query': {
                'condition_records': [{'source_quote': 'small'}, {'source_quote': 'near the door'}],
                '_v4_relation_evidence': [deepcopy(negative)] * 2},
            'evidence_state': {'missing': ['unknown counted member 4', 'smallness unresolved']},
            'observation': {'candidate_evidence': [deepcopy(negative)] * 2,
                'details': {'relation_tasks': [deepcopy(negative)] * 2},
                'next_evidence': [{'action': 'evidence', 'dimension': 'position',
                    'focus_ids': [2, 8], 'reason': 'verify every compatible support anchor'}]},
            'deletion_events': [deepcopy(event)] * 3,
            'history': [{'action': {'action': 'evidence', 'dimension': 'position'},
                'remaining_candidate_ids': [2, 4], 'evidence_state': {'missing': ['unknown counted member 4']}}] * 3}
        with patch.dict(os.environ, {'EVIGROUND_CONTEXT_LIMIT': '1000000000'}):
            compact, _ = FROZEN.planner_state(source)
            expected, _ = FROZEN.planner_state(source, tighter=True)
        encoded, _ = PROPOSED.compact_repeated_evidence(compact)
        self.assertEqual(encoded['deletion_events'], compact['deletion_events'])
        self.assertIn('shared_evidence_ref', json.dumps(encoded['observation']['candidate_evidence']))
        second, audit = PROPOSED.planner_state(encoded, tighter=True, instruction_bytes=4507)
        self.assertEqual(expand_protocol(second), expected)
        self.assertEqual(audit['instruction_bytes'], 4507)
        messages = [{'role': 'system', 'content': 'system instructions'},
            {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps(encoded, ensure_ascii=False)}]}]
        once = PROPOSED.compress_planner_messages(messages)
        twice = PROPOSED.compress_planner_messages(once)
        recovered = json.loads(twice[1]['content'][0]['text'])
        self.assertEqual(once, twice)
        self.assertEqual(expand_protocol(recovered), expected)
        self.assertEqual(recovered['evidence_state']['missing'], source['evidence_state']['missing'])
        with self.assertRaisesRegex(ValueError, 'required planner obligations'):
            PROPOSED.compress_planner_messages([{'role': 'system', 'content': 'large system ' * 3000}, messages[1]])


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(prior_tests.ContextProtocolTests))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ContextAllContainersTests))
    with patch('requests.sessions.Session.request', side_effect=AssertionError('offline network blocked')), \
         patch('subprocess.Popen', side_effect=AssertionError('offline subprocess blocked')):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
