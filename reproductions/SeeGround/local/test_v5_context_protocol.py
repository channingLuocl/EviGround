"""Offline lossless evidence encoding and fixed-budget real-state regressions."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from react_decision import compact_repeated_evidence

def expand_protocol(result):
    definitions = result.get('shared_evidence', {})
    def expand(value):
        if isinstance(value, dict):
            if set(value) == {'shared_evidence_ref'}:
                return expand(definitions[value['shared_evidence_ref']])
            if set(value) == {'table_shared', 'table_columns', 'table_rows'}:
                return [{**expand(value['table_shared']), **dict(zip(expand(value['table_columns']), row))}
                        for row in expand(value['table_rows'])]
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value
    return expand({key: value for key, value in result.items() if key not in ('shared_evidence', 'evidence_encoding')})


DELIVERY = Path(__file__).resolve().parents[1] / 'local_outputs/v5_context_protocol_fixtures'


import react_decision as PROPOSED

def load_frozen_baseline():
    path = DELIVERY / 'react_decision_frozen_r3.py'
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location('context_protocol_frozen_r3', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASELINE = load_frozen_baseline()


class ContextProtocolTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {'SEEGROUND_V5': '1', 'SEEGROUND_V4': '1',
            'SEEGROUND_CONTEXT_LIMIT': '32768', 'SEEGROUND_REACT_BACKEND': 'qwen',
            'SEEGROUND_API_PROVIDER': 'qwen'})
        environment.start()
        self.addCleanup(environment.stop)

    def synthetic_state(self):
        bindings = [{'status': 'bound', 'anchor_ids': [8], 'reference_role': 'support',
            'required_member_count': 3, 'reason': 'the quoted three doors are distinct'}] * 3
        proofs = [{'candidate_ids': [candidate], 'comparison_scope_ids': [candidate],
            'target_role': 'target', 'reference_role': 'support', 'required_member_count': 3,
            'status': 'fail' if candidate == 2 else 'unknown',
            'reason': 'outside support band' if candidate == 2 else 'unknown member remains',
            'unknown_member_ids': [19], 'reference_bindings': deepcopy(bindings),
            'explicit_fail_criterion': 'visible gap under every compatible anchor',
            'source_quote': 'the small chair near the three doors',
            'next_evidence': {'action': 'evidence', 'dimension': 'position', 'focus_ids': [candidate, 8]}}
            for candidate in range(2, 70)]
        return {'parsed_query': {'condition_records': [{'source_quote': 'small'},
                {'source_quote': 'near the three doors', 'required_member_count': 3}]},
            'evidence_state': {'missing': ['unknown member 19']},
            'observation': {'candidate_evidence': proofs, 'details': {'relation_tasks': proofs},
                'next_evidence': [{'action': 'evidence', 'dimension': 'position',
                    'focus_ids': [2, 8], 'reason': 'confirm all three quoted doors'}]},
            'deletion_events': [{'event_id': 4, 'candidate_id': 2, 'invalidated': False,
                'proofs': [deepcopy(proofs[0])]}]}

    def test_roundtrip_preserves_each_scope_role_count_and_negative_reason(self):
        source = self.synthetic_state()
        encoded, blocks = compact_repeated_evidence(source)
        decoded = expand_protocol(encoded)
        self.assertEqual(decoded, source)
        self.assertGreater(blocks, 0)
        self.assertLess(len(json.dumps(encoded)), len(json.dumps(source)))
        self.assertEqual(encoded['parsed_query'], source['parsed_query'])
        self.assertEqual(encoded['evidence_state'], source['evidence_state'])
        rows = decoded['observation']['candidate_evidence']
        self.assertEqual([row['comparison_scope_ids'] for row in rows], [[i] for i in range(2, 70)])
        self.assertEqual(rows[0]['status'], 'fail')
        self.assertTrue(all(row['status'] == 'unknown' for row in rows[1:]))
        self.assertTrue(all(row['required_member_count'] == 3 for row in rows))
        self.assertTrue(all(row['target_role'] == 'target' and row['reference_role'] == 'support' for row in rows))

    def test_next_evidence_is_still_a_direct_executable_dictionary(self):
        source = self.synthetic_state()
        encoded, _ = compact_repeated_evidence(source)
        decoded = expand_protocol(encoded)
        self.assertEqual(encoded['observation']['next_evidence'], source['observation']['next_evidence'])
        self.assertEqual(decoded['observation']['candidate_evidence'][0]['next_evidence'],
                         source['observation']['candidate_evidence'][0]['next_evidence'])
        definitions = json.dumps(encoded['shared_evidence'])
        self.assertNotIn('"action":{"shared_evidence_ref"', definitions.replace(' ', ''))

    def test_second_encoding_keeps_definitions_and_deletion_proofs(self):
        first, blocks = compact_repeated_evidence(self.synthetic_state())
        second, second_blocks = compact_repeated_evidence(first)
        self.assertEqual(second, first)
        self.assertEqual(second_blocks, blocks)
        self.assertEqual(expand_protocol(second)['deletion_events'], self.synthetic_state()['deletion_events'])

    def test_table_only_encoding_is_idempotent(self):
        source = {'observation': {'anchor_hypotheses': [
            {'anchor_ids': [i], 'status': 'unknown'} for i in (1, 2, 3)]}}
        first, blocks = compact_repeated_evidence(source)
        self.assertEqual(blocks, 0)
        second, _ = compact_repeated_evidence(first)
        self.assertEqual(second, first)
        self.assertEqual(expand_protocol(second), source)

    def test_jev_keeps_definitions_and_resolves_container_shape_before_projection(self):
        source = self.synthetic_state()
        encoded, _ = compact_repeated_evidence(source)
        expected = PROPOSED.jev_state(source)
        projected = PROPOSED.jev_state(encoded)
        self.assertEqual(projected['shared_evidence'], encoded['shared_evidence'])
        self.assertEqual(expand_protocol(projected), expected)
        # A repeated details/geometry container may itself be encoded as a reference.
        container = {'relation_tasks': source['observation']['candidate_evidence']}
        source['observation'] = {'details': container, 'candidate_evidence': [container] * 3,
            'geometry': {'support': deepcopy(container)}, 'auxiliary': {'support': deepcopy(container)}}
        encoded, _ = compact_repeated_evidence(source)
        self.assertEqual(expand_protocol(PROPOSED.jev_state(encoded)), PROPOSED.jev_state(source))

    def test_small_request_keeps_existing_protocol_and_audit(self):
        source = {'query': 'chair', 'evidence_state': {'missing': ['near the door']}}
        compact, audit = PROPOSED.planner_state(source, instruction_bytes=200)
        self.assertNotIn('evidence_encoding', compact)
        self.assertNotIn('shared_evidence_blocks', audit)
        self.assertEqual(audit['context_limit'], 32768)

    def test_growing_missing_and_event_never_silently_drop_obligations(self):
        fixture = DELIVERY / 'context_r3_failed_states.json'
        if not fixture.exists() or BASELINE is None:
            self.skipTest('exact frozen r3 fixtures unavailable')
        saved = json.loads(fixture.read_text())['22872']
        for growth in ('missing', 'deletion_event'):
            with self.subTest(growth=growth):
                source = deepcopy(saved['state'])
                if growth == 'missing':
                    source['evidence_state']['missing'].append('新增未知义务：' + 'must retain original source quote; ' * 100)
                else:
                    source['deletion_events'].append({'event_id': 999, 'candidate_id': 4,
                        'dimension': 'position', 'source_quotes': ['near the door'], 'invalidated': False,
                        'evidence': [{'candidate_id': 4, 'status': 'fail', 'source_quote': 'near the door',
                            'reason': 'new independently verified visible separation',
                            'explicit_fail_criterion': 'visible gap for every bound anchor',
                            'role_binding_status': 'bound', 'comparison_scope_status': 'complete'}]})
                instruction_bytes = len(saved['instructions'].encode())
                try:
                    compact, audit = PROPOSED.planner_state(source, instruction_bytes=instruction_bytes)
                except ValueError as exc:
                    self.assertEqual(str(exc), 'required planner obligations exceed configured text context budget')
                    continue
                with patch.dict(os.environ, {'SEEGROUND_CONTEXT_LIMIT': '1000000000'}):
                    original, _ = BASELINE.planner_state(source, tighter=audit['tighter'],
                                                        instruction_bytes=instruction_bytes)
                self.assertEqual(expand_protocol(compact), original)
                self.assertEqual(compact['evidence_state']['missing'], source['evidence_state']['missing'])
                self.assertLessEqual(audit['input_units'], 30720)

    def test_all_six_exact_failures_fit_and_roundtrip_with_full_instructions(self):
        fixture = DELIVERY / 'context_r3_failed_states.json'
        if not fixture.exists() or BASELINE is None:
            self.skipTest('exact frozen r3 fixtures unavailable')
        states = json.loads(fixture.read_text())
        self.assertEqual(set(states), {'4545', '5902', '20637', '22872', '30296', '40372'})
        for row, saved in states.items():
            with self.subTest(csv_row=row):
                instruction_bytes = len(saved['instructions'].encode())
                compact, audit = PROPOSED.planner_state(saved['state'], instruction_bytes=instruction_bytes)
                self.assertEqual(audit['method'], 'utf8_bytes_upper_bound')
                self.assertEqual(audit['context_limit'], 32768)
                self.assertEqual(audit['instruction_bytes'], instruction_bytes)
                payload_bytes = len(json.dumps(compact, ensure_ascii=False, separators=(',', ':')).encode())
                self.assertEqual(audit['input_units'], payload_bytes + instruction_bytes + 64)
                self.assertLessEqual(audit['input_units'], 30720)
                with patch.dict(os.environ, {'SEEGROUND_CONTEXT_LIMIT': '1000000000'}):
                    original, _ = BASELINE.planner_state(saved['state'], tighter=audit['tighter'],
                                                        instruction_bytes=instruction_bytes)
                self.assertEqual(expand_protocol(compact), original)
                self.assertEqual(compact['parsed_query']['condition_records'], saved['state']['parsed_query']['condition_records'])
                self.assertEqual(compact['evidence_state']['missing'], saved['state']['evidence_state']['missing'])
                self.assertEqual(expand_protocol(PROPOSED.jev_state(compact)), PROPOSED.jev_state(original))
                second, _ = PROPOSED.planner_state(compact, tighter=True, instruction_bytes=instruction_bytes)
                self.assertEqual(second['shared_evidence'], compact['shared_evidence'])
                self.assertEqual(second['deletion_events'], compact['deletion_events'])


if __name__ == '__main__':
    unittest.main()
