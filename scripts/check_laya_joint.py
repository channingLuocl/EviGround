"""Offline failure replay for the isolated dual-process contract."""
from copy import deepcopy
import io
import base64
from pathlib import Path
import sys
import unittest

import laya_joint as joint

sys.path[:0] = [str(joint.FROZEN / 'prototype'), str(joint.FROZEN / 'src')]
from jsonschema import ValidationError
from tokenizers import Tokenizer
from PIL import Image
import common
from monitor import DagMonitor
from pair_contract import assemble


class Replay(unittest.TestCase):
    def test_node_map_preserves_explicit_ids_and_body_without_semantic_repairs(self):
        import json
        from jsonschema import Draft202012Validator
        node = lambda name: common.obj({'id': common.enum([name]),
            'status': common.enum(['pending', 'completed']), 'conclusion': {'type': 'string'}})
        original = common.obj({'monitor': common.obj({'nodes': common.arr(
            {'anyOf': [node('n0'), node('n1')]}, 2, 1)})})
        wire = joint.node_map_schema(original)
        emitted = {'monitor': {'nodes': {'n1': {'status': 'pending', 'conclusion': ''},
            'n0': {'status': 'completed', 'conclusion': 'Observed fact'}}}}
        self.assertTrue(Draft202012Validator(wire).is_valid(emitted))
        decoded = joint.decode_node_maps(emitted)
        self.assertEqual(decoded['monitor']['nodes'][0]['id'], 'n1')
        self.assertEqual(decoded['monitor']['nodes'][1]['conclusion'], 'Observed fact')
        self.assertTrue(Draft202012Validator(original).is_valid(decoded))
        self.assertEqual(decoded['monitor']['nodes'][0]['status'], 'pending')
        domain = deepcopy(original)
        domain['properties']['monitor']['properties']['nodes']['items']['anyOf'][0]['properties']['id'] = common.enum(['n0', 'n1'])
        self.assertTrue(Draft202012Validator(joint.node_map_schema(domain)).is_valid(emitted))
        bad = deepcopy(emitted); bad['monitor']['nodes']['n9'] = {'status': 'pending', 'conclusion': ''}
        self.assertFalse(Draft202012Validator(wire).is_valid(bad))
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON field'):
            json.loads('{"n1":{},"n1":{}}', object_pairs_hook=joint.unique_json_object)

    def test_shared_monitor_permits_terminal_proposal_but_not_illegal_tool(self):
        from jsonschema import Draft202012Validator
        pending = {'type': 'object', 'properties': {'status': {'enum': ['pending']}},
            'required': ['status'], 'additionalProperties': False}
        terminal = deepcopy(pending)
        terminal['properties']['status']['enum'].append('completed')
        schema = {'anyOf': [common.obj({'action': {'enum': [name]}, 'monitor': mon})
            for name, mon in [('inspect', pending), ('finish', terminal)]]}
        original = deepcopy(schema)
        shared = joint.shared_monitor_schema(schema)
        self.assertTrue(Draft202012Validator(shared).is_valid({'status': 'completed'}))
        self.assertFalse(Draft202012Validator(schema).is_valid(
            {'action': 'inspect', 'monitor': {'status': 'completed'}}))
        self.assertTrue(Draft202012Validator(schema).is_valid(
            {'action': 'finish', 'monitor': {'status': 'completed'}}))
        self.assertEqual(schema, original)

    def test_consolidated_primary_retains_full_state_schema_and_bound(self):
        import json
        path = joint.ROOT / 'cases/v4-nr3d/calls/004-nr3d-t01-semantic_consultant/request.json'
        original = json.loads(path.read_text())
        candidate = deepcopy(original)
        candidate['messages'][0]['content'] = joint.PRIMARY_RULES
        self.assertEqual(candidate['messages'][1:], original['messages'][1:])
        self.assertEqual(candidate['response_format'], original['response_format'])
        count = joint.window(None, candidate['messages'], candidate['max_tokens'], candidate['response_format'])
        self.assertLessEqual(count['maximum'], 32768)
        self.assertGreater(count['maximum'], 32000)
        for rule in ['Unknown/unseen is not false', 'reopen_node_ids', 'q0', 'n11',
                     'contradicting', '48 Laya tokens', '4000 output tokens', 'EVERY still-pending']:
            self.assertIn(rule, joint.PRIMARY_RULES)

    def setUp(self):
        self.schema = {'type': 'object', 'properties': {
            'action': {'enum': ['inspect']}, 'focus_ids': {'type': 'array',
                'items': {'enum': [1, 2]}, 'minItems': 1, 'maxItems': 2},
            'monitor': {'type': 'object'}, 'reason': {'type': 'string'}},
            'required': ['action', 'focus_ids', 'monitor', 'reason'], 'additionalProperties': False}
        self.state = {'description': 'the chair', 'task_dag': None,
            'allowed_target_ids': [1, 2], 'source_directory': []}
        self.proposal = {'state_version': joint.digest(self.state),
            'monitor': {'current_node_id': 'n0'}, 'decision_basis': 'Inspect both candidates.',
            'options': [{'action': 'inspect', 'focus_ids': [i], 'reason': 'Inspect this object.'}
                        for i in [1, 2]]}

    def test_stale_version_stops_before_selection(self):
        with self.assertRaisesRegex(AssertionError, 'Stale'):
            joint.validate_proposal(self.proposal, 'newer-version', self.schema)

    def test_invalid_type_or_id_cannot_be_selected_away(self):
        for value in [['1'], [3], []]:
            bad = deepcopy(self.proposal)
            bad['options'][1]['focus_ids'] = value
            with self.subTest(value=value), self.assertRaises(ValidationError):
                joint.validate_proposal(bad, bad['state_version'], self.schema)

    def test_schema_factoring_preserves_all_fields_and_enum_domains(self):
        from jsonschema import Draft202012Validator
        domain = {'enum': list(range(41))}
        original = {'type': 'object', 'properties': {'target': domain, 'reference': domain},
            'required': ['target', 'reference'], 'additionalProperties': False}
        factored = joint.factor_schema(original)
        Draft202012Validator.check_schema(factored)
        for value in [{'target': 1, 'reference': 40}, {'target': 41, 'reference': 1},
                      {'target': 1}, {'target': '1', 'reference': 2}]:
            self.assertEqual(Draft202012Validator(original).is_valid(value),
                Draft202012Validator(factored).is_valid(value))
        nested = {'type': 'object', 'properties': original['properties'],
            'required': ['target', 'reference'], 'additionalProperties': False}
        repeated = {'type': 'object', 'properties': {'left': nested, 'right': nested},
            'required': ['left', 'right'], 'additionalProperties': False}
        nested_factor = joint.factor_schema(repeated)
        Draft202012Validator.check_schema(nested_factor)
        self.assertTrue(Draft202012Validator(nested_factor).is_valid(
            {'left': {'target': 1, 'reference': 2}, 'right': {'target': 3, 'reference': 4}}))

    def test_duplicate_or_narration_only_options_rejected(self):
        bad = deepcopy(self.proposal)
        bad['options'][1] = {**bad['options'][0], 'reason': 'A different narrative.'}
        with self.assertRaisesRegex(AssertionError, 'Narration-only'):
            joint.validate_proposal(bad, bad['state_version'], self.schema)
        bad = deepcopy(self.proposal)
        bad['options'][0]['focus_ids'] = [1, 1]
        with self.assertRaises(AssertionError):
            joint.validate_proposal(bad, bad['state_version'], self.schema)

    def test_full_state_and_parameters_retained_and_capacity_guarded(self):
        tokenizer = Tokenizer.from_file(str(joint.ROOT / 'laya-tokenizer/tokenizer.json'))
        tokenizer.no_truncation()
        self.schema['properties']['context'] = {'enum': ['isolated']}
        for option in self.proposal['options']:
            option['context'] = 'isolated'
        actions = joint.validate_proposal(self.proposal, self.proposal['state_version'], self.schema)
        request, count = joint.laya_request(self.state, self.proposal, actions, tokenizer)
        self.assertEqual(request['state']['accepted_state'], self.state)
        self.assertEqual(request['state']['actions']['c1']['focus_ids'], [2])
        self.assertLessEqual(max(count['option_tokens']), 48)
        self.assertEqual(count['full_original_state_sha256'], self.proposal['state_version'])
        huge = {**self.state, 'reports': 'chair ' * 8500}
        with self.assertRaisesRegex(AssertionError, 'nothing was dropped'):
            joint.laya_request(huge, self.proposal, actions, tokenizer)

    def test_images_schema_and_output_counted_before_send(self):
        image = Image.new('RGB', (64, 64), (60, 70, 80))
        stream = io.BytesIO(); image.save(stream, format='PNG')
        uri = 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()
        messages = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': uri}}]}]
        first = joint.window(None, messages, 1500, {'schema': 'x' * 600})
        second = joint.window(None, messages, 2500, {'schema': 'x' * 1200})
        self.assertEqual(first['actual_image_parts'], 1)
        self.assertEqual(first['images'][0]['width'], 64)
        self.assertEqual(second['maximum'] - first['maximum'], 1600)
        messages[0]['content'].append(messages[0]['content'][0])
        self.assertGreater(joint.window(None, messages, 1500)['maximum'], 32768)

    def test_pair_domains_and_current_node_are_not_inferred(self):
        request = {'measurement_type': 'pair_geometry', 'target_ids': [1],
            'reference_ids': [2], 'node_id': 'n0'}
        focus, bindings = assemble(request, [1, 2, 3], {'current_node_id': 'n0', 'nodes': []})
        self.assertEqual(focus, [1, 2]); self.assertEqual(bindings, [])
        for change in [{'reference_ids': [1]}, {'target_ids': ['1']}, {'node_id': 'n1'}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                assemble({**request, **change}, [1, 2, 3], {'current_node_id': 'n0', 'nodes': []})
        proposal = {'state_version': 'version', 'monitor': {'current_node_id': 'n0'},
            'options': [{'action': 'measure', 'measurement_type': 'object_geometry',
                'target_ids': [1], 'reference_ids': [], 'node_id': node} for node in ['n0', 'n1']]}
        with self.assertRaisesRegex(AssertionError, 'shared current node n0; measure.node_id=n1'):
            joint.validate_proposal(proposal, 'version', {'type': 'object'})

    def test_rollback_reactivates_rival_and_invalidates_descendants(self):
        query = 'the middle chair'
        parsed = {'Target': ['chair'], 'Anchor': [], 'Relations': [], 'Viewpoint': '',
            'Attributes': [], 'Unhandled': [], 'Observer': {
                'origin': {'kind': 'unspecified', 'category': '', 'phrase': ''},
                'facing': {'category': '', 'group': False, 'phrase': ''}}}
        monitor = DagMonitor(query, parsed, [1, 2, 3], [1, 2], ['inspect', 'measure', 'finish'])
        monitor.initial_evidence_object_ids = [1, 2, 3]

        def node(name, kind, deps, complete=False, excluded=()):
            return {'id': name, 'goal': name, 'kind': kind, 'source_quotes': [query],
                'condition_ids': ['q0'], 'depends_on': deps,
                'allowed_actions': ['finish'] if kind == 'verification' else ['inspect', 'measure'],
                'status': 'completed' if complete else 'pending', 'bound_object_ids': [1, 2, 3],
                'evidence_ids': ['initial'] if complete else [],
                'conclusion': 'observed contradiction excludes chair 2' if complete else '',
                'excluded_target_ids': list(excluded)}

        proposal = {'nodes': [node('group', 'binding', [], True),
            node('compare', 'comparison', ['group'], True, [2]),
            node('check', 'relation', ['compare']), node('verify', 'verification', ['check'])],
            'current_node_id': 'check', 'reopen_node_ids': [], 'revision_reason': 'check relation'}
        action = {'action': 'inspect', 'focus_ids': [1, 2, 3], 'predicted_id': None,
            'reason': 'inspect', 'context': 'scene', 'frame': None, 'azimuth_deg': 0,
            'elevation_deg': 30, 'evidence': [], 'unresolved': []}
        monitor.update(proposal, action, [])
        self.assertEqual(monitor.active_ids(), [1])
        revised = deepcopy(proposal)
        revised.update(reopen_node_ids=['group'], current_node_id='group')
        event = monitor.update(revised, action, [])
        self.assertEqual(event['reactivated_target_ids'], [2])
        comparison = next(n for n in monitor.plan['nodes'] if n['id'] == 'compare')
        self.assertEqual(comparison['status'], 'pending')
        self.assertEqual(comparison['excluded_target_ids'], [])
        self.assertEqual(comparison['bound_object_ids'], [1, 2, 3])

    def test_budget_patch_charges_slow_before_any_network_attempt(self):
        original = (joint.FROZEN / 'prototype/runner.py').read_text()
        self.assertEqual(original.count("            action = run.call(dataset + f'-t{turn:02d}', 'primary', rules, state,"), 1)
        self.assertEqual(joint.LIMITS, {'B': 10, 'R': 6, 'T': 17, 'decision_output_reserved': 4000})
        used = 0
        for _ in range(6):
            used = joint.charge_slow_consultation(used)
        self.assertEqual(used, 6)
        with self.assertRaisesRegex(AssertionError, 'No slow'):
            joint.charge_slow_consultation(used)
        self.assertGreater(joint.window(None, [{'role': 'user', 'content': 'x' * 32000}], 2200)['maximum'], 32768)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Replay))
    joint.save(joint.ROOT / 'offline-replay.json', {'tests': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors), 'passed': result.wasSuccessful(),
        'script_sha256': joint.file_sha(__file__), 'entry_sha256': joint.file_sha(joint.__file__),
        'network_calls': 0, 'evaluation_labels': False})
    sys.exit(0 if result.wasSuccessful() else 1)
