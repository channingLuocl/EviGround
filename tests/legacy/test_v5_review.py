"""Offline regressions for source-bound review and current evidence."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from eviground.grounding.active_diagnostic import active_ground, compute_observation, execute_v4_inspection, measurement_version
from eviground.geometry.size_semantics import evaluate_size_condition, size_specs
import tests.legacy.test_v4 as test_v4
from tests.legacy.test_v4 import box, measure, parsed
from eviground.grounding.v4_policy import settings, v5_enabled
from eviground.rendering.v4_visual import validate_findings


class V5ReviewTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'EVIGROUND_V4': '1', 'EVIGROUND_V5': '1',
            'EVIGROUND_API_PROVIDER': 'qwen', 'EVIGROUND_REACT_BACKEND': 'qwen',
            'EVIGROUND_V5_SCOPE': '1', 'EVIGROUND_V5_MEASUREMENTS': '1', 'EVIGROUND_V5_CALL_ECONOMY': '1'})
        env.start()
        self.addCleanup(env.stop)

    def test_ablation_flags_preserve_control_and_existing_v4_switches(self):
        with patch.dict(os.environ, {'EVIGROUND_V5_SCOPE': '0'}):
            self.assertTrue(v5_enabled())
            self.assertFalse(v5_enabled('scope'))
            self.assertTrue(settings()['v5_measurements'])
        with patch.dict(os.environ, {'EVIGROUND_V4': '0'}):
            self.assertFalse(v5_enabled('scope'))

    def measurement_review(self, candidate_id):
        rows = [{'id': oid, 'status': 'known', 'source': 'assigned points in world meters',
            'height_m': value, 'metric_reliability': {'height_m': {'status': 'usable'}},
            'observed_completeness': {'status': 'supported'}} for oid, value in [(1, 1.), (2, 2.)]]
        condition = evaluate_size_condition(size_specs('taller', 'chair')[0], rows, [1, 2],
            scope_bound=True, candidate_ids=[1])
        event = {'event_id': 7, 'candidate_id': candidate_id, 'source_quotes': ['taller'], 'invalidated': False}
        finding = {'finding': 'refutes_exclusion', 'candidate_id': candidate_id, 'event_id': 7,
            'source_quote': 'taller', 'target_role': 'target', 'predicate_status': 'pass',
            'evidence_type': 'measurement', 'reason': 'current height ranking refutes the deletion',
            'measurement_keys': ['height_m'], 'comparison': {'key': 'height_m', 'operator': 'gt',
                'reference_id': 3 - candidate_id, 'reference_key': 'height_m'}}
        return rows, condition, event, finding

    def test_measurement_refutation_recomputes_deleted_candidate_from_current_pool(self):
        rows, condition, event, finding = self.measurement_review(2)
        self.assertFalse(any(e['candidate_id'] == 2 for e in condition['candidate_evidence']))
        checked = validate_findings({'findings': [finding]}, 'the taller chair',
            {'focus_ids': [1, 2], 'visible_instance_pixels': {'1': 100, '2': 100}},
            {'selected_id': 1, 'measurements': rows, 'size_constraints': [condition], 'deletions': [event]})
        self.assertEqual(len(checked), 1)

    def test_model_status_and_arbitrary_true_inequality_do_not_refute_size_predicate(self):
        rows, condition, event, finding = self.measurement_review(1)
        condition['candidate_evidence'][0]['status'] = 'pass'
        finding['comparison']['operator'] = 'lt'  # True numerically, opposite to "taller".
        checked = validate_findings({'findings': [finding]}, 'the taller chair',
            {'focus_ids': [1, 2], 'visible_instance_pixels': {'1': 100, '2': 100}},
            {'selected_id': 2, 'measurements': rows, 'size_constraints': [condition], 'deletions': [event]})
        self.assertEqual(checked, [])

    def inspection(self, out, action, p, calls):
        rows = [{'id': 1, 'height_m': 1., 'observed_completeness': {'status': 'supported'}}]
        result = {'details': {'comparison_metrics': rows, 'comparison_pool_ids': [1],
            'size_constraints': [{'original_quote': 'tall', 'metric_ambiguity': True}]},
            'images': [], 'internal_steps': []}
        view = {'image': str(out / 'mock.png'), 'focus_ids': [1],
            'visible_instance_pixels': {'1': 100}, 'source': 'assigned points'}
        with patch('eviground.grounding.active_diagnostic.boxed_view', return_value=view), \
             patch('eviground.grounding.active_diagnostic.size_candidate_evidence', return_value=([], [], {})):
            execute_v4_inspection(action, 'tall chair', p, {'candidate_ids': [1]}, [box(1)], [box(1)],
                np.zeros((1, 3)), np.zeros((1, 3)), np.ones(1), out, 1, calls, result)
        return result

    def test_unknown_completeness_keeps_supported_measurement(self):
        report = {'status': 'unknown', 'reason': 'occluded', 'observations': [
            {'id': 1, 'completeness': 'unknown', 'reason': 'occluded'}], 'metric_choices': []}
        with tempfile.TemporaryDirectory() as directory, patch('eviground.grounding.active_diagnostic.visual_report', return_value=report):
            result = self.inspection(Path(directory), {'dimension': 'size'}, {}, [{'stage': 'mock'}])
        row = result['details']['comparison_metrics'][0]
        self.assertEqual(row['observed_completeness']['status'], 'supported')
        self.assertEqual(row['height_m'], 1.)
        self.assertEqual(row['visual_observations'][0]['completeness'], 'unknown')

    def test_unrelated_invalidation_reuses_view_and_new_camera_calls_again(self):
        report = {'status': 'unknown', 'reason': 'occluded', 'observations': [], 'metric_choices': []}
        p, calls = {}, [{'stage': 'mock'}]
        with tempfile.TemporaryDirectory() as directory, patch('eviground.grounding.active_diagnostic.visual_report', return_value=report) as visual:
            out = Path(directory)
            self.inspection(out, {'dimension': 'size'}, p, calls)
            p['_v4_invalidated_conditions'] = ['red']
            cached = self.inspection(out, {'dimension': 'size'}, p, calls)
            self.assertTrue(cached['details']['visual_inspection']['cache_hit'])
            self.inspection(out, {'dimension': 'size', 'azimuth_deg': 180}, p, calls)
            self.assertEqual(visual.call_count, 2)
            self.assertEqual(p['_v4_budget']['size'], 2)

    def test_same_measurement_reuses_completeness_but_changed_geometry_does_not(self):
        p = parsed('chair', 'tall')
        row = {'id': 1, 'height_m': 1., 'status': 'known', 'source': 'assigned points',
               'observed_completeness': {'status': 'unknown', 'surface_only_risk': True}}
        p['_v5_measurement_observations'] = {'1': {'measurement_version': measurement_version(row),
            'observed_completeness': {'status': 'supported', 'surface_only_risk': True}}}
        with patch('eviground.grounding.active_diagnostic._target_size_scope', return_value=([1], True, [])), \
             patch('eviground.grounding.active_diagnostic.size_candidate_evidence', return_value=([], [], {'comparison_pool_ids': [1]})):
            for height, expected in [(1., 'supported'), (1.2, 'unknown')]:
                with patch('eviground.grounding.active_diagnostic.point_cloud_measurements', return_value=[{**deepcopy(row), 'height_m': height}]):
                    result = compute_observation('size', 'tall chair', p, {'candidate_ids': [1]}, [box(1, 'chair')],
                        [box(1, 'chair')], np.zeros((1, 3)), np.ones(1))
                self.assertEqual(result['comparison_metrics'][0]['observed_completeness']['status'], expected)

    def test_all_reference_binding_proofs_are_retained_and_unknown_roles_block_resolution(self):
        geometry = {'status': 'unknown', 'anchor_ids': [2, 3], 'reference_binding': {'status': 'unknown'},
            'anchor_hypotheses': [{'status': 'pass', 'anchor_ids': [oid]} for oid in (2, 3)]}
        row = {'candidate_id': 1, 'source_quote': 'near the door', 'geometry_evidence': geometry}
        report = {'status': 'unknown', 'relations': [{'candidate_id': 1, 'anchor_id': oid,
            'source_quote': 'near the door', 'status': 'pass', 'reason': 'visible adjacency'} for oid in (2, 3)]}
        result = {'details': {}, 'candidate_evidence': [], 'images': [], 'internal_steps': []}
        view = {'image': 'mock.png', 'focus_ids': [1, 2, 3], 'source': 'assigned points',
                'visible_instance_pixels': {str(oid): 100 for oid in (1, 2, 3)}}
        p = {}
        with patch('eviground.grounding.active_diagnostic.bind_tasks', return_value=[row]), \
             patch('eviground.grounding.active_diagnostic.boxed_view', return_value=view), \
             patch('eviground.grounding.active_diagnostic.visual_report', return_value=report):
            execute_v4_inspection({'dimension': 'distance', 'focus_ids': [1, 2, 3]}, 'chair near the door', p,
                {'candidate_ids': [1]}, [box(1, 'chair')], [box(1, 'chair'), box(2, 'door'), box(3, 'door')],
                np.zeros((1, 3)), np.zeros((1, 3)), np.ones(1), Path('.'), 1, [{'stage': 'mock'}], result)
        proof = p['_v4_relation_evidence'][0]
        self.assertEqual(proof['status'], 'unknown')
        self.assertEqual(proof['role_binding_status'], 'unknown')
        self.assertEqual({b['anchor_id'] for b in proof['reference_bindings']}, {2, 3})
        self.assertTrue(all(b['visual_evidence']['anchor_id'] == b['anchor_id'] for b in proof['reference_bindings']))

    def test_selected_relation_refutation_requires_visible_bound_reference_proofs(self):
        quote = 'near the door'
        finding = {'finding': 'contradicts_selected', 'candidate_id': 1, 'source_quote': quote,
            'target_role': 'target', 'predicate_status': 'fail', 'evidence_type': 'visible_geometry',
            'observation_ids': [1], 'reason': 'separated from door'}
        report = {'findings': [finding], 'observations': [{'id': 1, 'reason': 'separated from door',
            'condition_checks': [{'source_quote': quote, 'status': 'fail', 'reason': 'separated from door'}]},
            {'id': 2, 'reason': 'visible door'}, {'id': 3, 'reason': 'second possible door'}]}
        view = {'focus_ids': [1, 2, 3], 'visible_instance_pixels': {str(oid): 100 for oid in (1, 2, 3)}}
        payload = {'selected_id': 1, 'evidence_inventory': {'items': [{'source_quote': quote,
            'target_role': {'category': 'chair'}, 'reference_role': {'category': 'door'}}]},
            'relations': [{'candidate_id': 1, 'source_quote': quote, 'role_binding_status': 'bound',
                'geometry_evidence': {'anchor_ids': [2, 3]}}]}
        self.assertEqual(validate_findings(report, 'chair near the door', view, payload), [])
        finding['observation_ids'] = [1, 2, 3]
        report['relations'] = [{'candidate_id': 1, 'anchor_id': 2, 'source_quote': quote,
                               'status': 'fail', 'reason': 'visible separation'}]
        self.assertEqual(validate_findings(report, 'chair near the door', view, payload), [])
        report['relations'].append({**report['relations'][0], 'anchor_id': 3})
        self.assertEqual(len(validate_findings(report, 'chair near the door', view, payload)), 1)
        payload['relations'][0]['role_binding_status'] = 'unknown'
        self.assertEqual(validate_findings(report, 'chair near the door', view, payload), [])

    def test_later_relation_binding_refreshes_final_comparison_summary(self):
        helper = test_v4.V4Tests()
        diagnosis = {'candidate_ids': [1, 2], 'stable_candidate_ids': [], 'status': 'missing',
            'missing': ['unbound relation'], 'geometry': {}, 'pending_size_conditions': [], 'hypotheses': []}
        size = helper.operation('size', [])
        old = {'original_quote': 'tall', 'attribute': 'tall', 'scope_status': 'unknown', 'candidate_evidence': []}
        size['details'] = {'comparison_metrics': [{'id': 1}, {'id': 2}], 'size_constraints': [old],
                           'candidate_evidence': []}
        relation = helper.operation('distance', [])
        relation['details']['relation_tasks'] = [{'candidate_id': 1, 'source_quote': 'near the door', 'status': 'pass'}]
        evidence = helper.operation('size', [2])['candidate_evidence']
        current = {**old, 'scope_status': 'bound', 'candidate_evidence': evidence}
        with patch('eviground.grounding.active_diagnostic.size_candidate_evidence', return_value=(evidence, [current], {})):
            result = helper.controller([{'action': 'evidence', 'dimension': 'size', 'reason': 'measure'},
                {'action': 'evidence', 'dimension': 'distance', 'reason': 'bind relation'},
                {'action': 'stop', 'reason': 'remaining unknown'}], [{'status': 'consistent', 'reason': 'visible'}],
                diagnosis=diagnosis, operations=[size, relation])
        self.assertEqual(result['current_evidence']['size_constraints'][0]['scope_status'], 'bound')

    def test_recovery_preserves_unrelated_candidate_relation_and_full_scope(self):
        objects = [box(1, 'chair'), box(2, 'chair', center=(2, 0, .5))]
        _, xyz, ids = measure(objects)
        p = parsed('chair', 'tall')
        binding_key = json.dumps(['target', 'chair', 'tall', 'full_category', None, None])
        p['_size_scope_bindings'] = {binding_key: {'status': 'bound', 'selected_ids': [1, 2]}}
        relations = [{'candidate_id': oid, 'source_quote': quote, 'status': 'pass'}
            for oid, quote in [(1, 'tall'), (2, 'tall'), (2, 'red')]]
        p['_v4_relation_evidence'] = deepcopy(relations)
        initial = {'candidate_ids': [1], 'stable_candidate_ids': [1], 'status': 'missing',
            'missing': ['unverified height'], 'geometry': {}, 'pending_size_conditions': [],
            'hypotheses': [{'candidate_ids': [1]}], 'candidate_removal_audit': [
                {'candidate_id': 2, 'source_quote': 'tall', 'status': 'fail', 'measurement_source': 'initial'}]}
        actions = iter([{'action': 'evidence', 'dimension': 'distance', 'reason': 'inspect'},
            {'action': 'stop', 'reason': 'unknown'}, {'action': 'stop', 'reason': 'need evidence'}])
        report = {'status': 'contradiction', 'findings': [{'finding': 'refutes_exclusion',
            'candidate_id': 2, 'event_id': 0, 'source_quote': 'tall', 'target_role': 'target',
            'predicate_status': 'pass', 'evidence_type': 'visible_geometry', 'observation_ids': [2],
            'reason': 'standing body'}], 'observations': [{'id': 2, 'reason': 'standing body',
            'condition_checks': [{'source_quote': 'tall', 'status': 'pass', 'reason': 'standing body'}]}]}
        def chat(messages, **kwargs):
            content = report if messages[0]['content'].startswith('Inspect the original query') else next(actions)
            return {'_content': json.dumps(content), '_finish_reason': 'stop', '_baseline': {}, 'usage': {}}
        operation = test_v4.V4Tests().operation('distance', [])
        operation['details']['relation_tasks'] = deepcopy(relations)
        with tempfile.TemporaryDirectory() as directory, patch('eviground.grounding.active_diagnostic.chat', side_effect=chat), \
             patch('eviground.grounding.active_diagnostic.diagnose', side_effect=lambda *a, **k: deepcopy(initial)), \
             patch('eviground.grounding.active_diagnostic.execute_evidence', return_value=operation):
            result = active_ground('tall chair', p, objects, objects, objects, xyz,
                np.full(xyz.shape, 100, dtype=np.uint8), ids, Path(directory), [], {}, 3, True)
        self.assertEqual(result['backtrack_count'], 1)
        self.assertIn(binding_key, p['_size_scope_bindings'])
        retained = {(r['candidate_id'], r['source_quote']) for r in result['current_evidence']['relations']}
        self.assertEqual(retained, {(1, 'tall'), (2, 'red')})


if __name__ == '__main__':
    unittest.main()
