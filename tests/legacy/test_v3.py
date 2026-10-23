"""Pending regression tests: copy into v3 local after KESU returns."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from seeground.grounding.active_diagnostic import active_ground, singleton_prediction
from seeground.evaluation.batch_nr3d import summarize
from seeground.grounding.diagnostic import validate_structure, size_conditions, size_condition_scope, _source_attribute_bindings
from seeground.query.query_syntax import analyze_query
from seeground.query.query_rewrite import rewrite_query


def box(i, category='chair', z=1.):
    return {'bbox_id': i, 'target': category, 'bbox_3d': [0, 0, z, 1, 1, 1]}


def raw(category='chair', word='taller', unhandled=()):
    return {'Target': [category], 'Anchor': [], 'Observer': {}, 'Relations': [],
            'Attributes': [{'category': category, 'role': 'target', 'phrase': word}],
            'Unhandled': list(unhandled)}


class V3Tests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'SEEGROUND_SYNTAX_BACKEND': 'spacy'})
        env.start()
        self.addCleanup(env.stop)

    def test_singleton_guard(self):
        for ids in ([], [1, 2], [999], [True], [1, 1]):
            self.assertEqual(singleton_prediction(None, ids, [box(1), box(2)]), (None, False))
        self.assertEqual(singleton_prediction(2, [1], [box(1), box(2)]), (2, False))
        self.assertEqual(singleton_prediction(None, [1], [box(1)]), (1, True))

    def run_controller(self, replies):
        objects = [box(1)]
        parsed = raw(word='tall', unhandled=['tall'])
        parsed['diagnostic_missing'] = ['descriptive height remains unverified']
        messages_seen = []
        replies = iter(replies)
        def chat(messages, **kwargs):
            messages_seen.append(messages)
            reply = next(replies)
            if isinstance(reply, BaseException):
                raise reply
            return {'_content': json.dumps(reply), '_finish_reason': 'stop',
                    '_baseline': {'model': 'mock', 'latency_s': 0}, 'usage': {}}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                'SEEGROUND_REACT_BACKEND': 'qwen'}), patch('seeground.grounding.active_diagnostic.chat', side_effect=chat):
            result = active_ground('tall chair', parsed, objects, objects, objects,
                np.array([[0., 0., 0.], [0., 0., 1.]]), np.zeros((2, 3), dtype=np.uint8),
                np.array([1, 1]), Path(directory), [], {}, 2, True)
        return result, messages_seen

    def test_singleton_does_not_end_evidence_loop_early(self):
        result, messages = self.run_controller([
            {'action': 'evidence', 'dimension': 'size', 'reason': 'check height'},
            {'action': 'stop', 'reason': 'height remains descriptive'}])
        self.assertEqual(len(messages), 2)
        self.assertEqual(result['active_trace']['successful_tools'], 1)
        self.assertEqual(result['predicted_id'], 1)
        self.assertEqual(result['prediction_source'], 'singleton_fallback')
        self.assertEqual(result['decision_status'], 'missing')
        self.assertFalse(result['accepted'])
        self.assertTrue(result['diagnosis']['missing'])
        self.assertEqual(result['active_trace']['stop_reason'], 'model_stop')

    def test_api_error_and_user_interrupt_propagate_without_prediction(self):
        for error in (OSError('API unavailable'), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__), self.assertRaises(type(error)):
                self.run_controller([error])

    def test_answer_and_certification_metrics_are_separate(self):
        result = summarize([
            {'predicted_id': 1, 'correct': True, 'accepted': True, 'decision_status': 'resolved'},
            {'predicted_id': 2, 'correct': False, 'accepted': False, 'decision_status': 'missing',
             'prediction_source': 'singleton_fallback'},
            {'predicted_id': None, 'correct': False, 'accepted': False, 'decision_status': 'missing'}])
        self.assertEqual(result['completed'], 2)
        self.assertEqual(result['answer_coverage'], 2/3)
        self.assertEqual(result['answered_accuracy'], .5)
        self.assertEqual(result['accepted_coverage'], 1/3)
        self.assertEqual(result['accepted_accuracy'], 1.)

    def test_full_phrase_pair_and_three_scene_objects(self):
        query = 'the taller of the two chairs'
        hint, = analyze_query(query)['comparisons']
        self.assertEqual(hint['source_phrase'], query)
        self.assertEqual(hint['head_category'], 'chair')
        self.assertEqual(hint['required_member_count'], 2)
        parsed = validate_structure(query, raw(unhandled=['of the two']))
        self.assertEqual(parsed['Unhandled'], [])
        rewritten = rewrite_query(query, parsed)
        rewritten_spec = rewritten['constraints'][0]['size_conditions'][0]
        self.assertEqual(rewritten_spec['source_phrase'], query)
        self.assertEqual(rewritten_spec['required_member_count'], 2)
        record = next(r for r in parsed['condition_records'] if r['condition_type'] == 'size')
        self.assertEqual(record['language_comparisons'][0]['source_phrase'], query)
        self.assertEqual(record['scene_binding_status'], 'pending')
        spec, = size_conditions(parsed, 'chair')
        _, scope = size_condition_scope(parsed, [box(1), box(2), box(3)], 'chair', spec)
        self.assertEqual(scope['status'], 'unknown')
        self.assertEqual(scope['selected_ids'], [1, 2, 3])

    def test_preserve_quantity_and_than(self):
        hint, = analyze_query('the larger of the three chairs')['comparisons']
        self.assertEqual(hint['required_member_count'], 3)
        hint, = analyze_query('The chair is larger than the table.')['comparisons']
        self.assertEqual(hint['head_category'], 'chair')
        self.assertEqual(hint['reference_category'], 'table')
        self.assertEqual(hint['scope_type'], 'reference_comparison')

    def test_target_restatement_keeps_distinct_reference_roles(self):
        query = 'Facing the taller chair, the target object is the shorter chair near the red chair.'
        parsed = {'Target': ['chair'], 'Anchor': ['chair'], 'query_syntax': analyze_query(query),
                  'Observer': {'facing': {'category': 'chair', 'phrase': 'Facing the taller chair'}}}
        self.assertEqual(_source_attribute_bindings(query, parsed, 'taller'), {('chair', 'reference')})
        self.assertEqual(_source_attribute_bindings(query, parsed, 'shorter'), {('chair', 'target')})
        self.assertEqual(_source_attribute_bindings(query, parsed, 'red'), {('chair', 'reference')})

    def subset(self, relative=True):
        query = 'the larger of the two windows ' + ('that are ' if relative else '') + 'above the door'
        parsed = validate_structure(query, {**raw('window', 'larger', ['of the two windows']),
            'Anchor': ['door'], 'Relations': [{'target_category': 'window', 'anchor_category': 'door',
                'relation': 'above', 'scope': 'relative_anchor', 'phrase': 'above the door'}]})
        return parsed, size_conditions(parsed, 'window')[0]

    def test_relative_subset_freezes_and_rejects_changed_membership(self):
        parsed, spec = self.subset()
        self.assertEqual(parsed['diagnostic_missing'], [])
        self.assertEqual(spec['scope_kind'], 'relation_subset')
        objects = [box(1, 'window', 3), box(2, 'window', 4), box(3, 'window', -3), box(4, 'door', 0)]
        pool, scope = size_condition_scope(parsed, objects, 'window', spec)
        self.assertEqual([o['bbox_id'] for o in pool], [1, 2])
        self.assertEqual(scope['status'], 'bound')
        objects[0]['bbox_3d'][2] = -3
        _, changed = size_condition_scope(parsed, objects, 'window', spec)
        self.assertEqual(changed['status'], 'unknown')
        self.assertEqual(changed['selected_ids'], [1, 2])

    def test_ambiguous_pp_remains_unbound(self):
        parsed, spec = self.subset(relative=False)
        self.assertEqual(parsed['query_syntax']['comparisons'][0]['attachment'], 'ambiguous')
        self.assertEqual(spec['scope_kind'], 'unbound_explicit_scope')

    def test_independent_relation_remains_separate(self):
        query = 'the larger window and above the door'
        parsed = validate_structure(query, {**raw('window', 'larger'), 'Anchor': ['door'],
            'Relations': [{'target_category': 'window', 'anchor_category': 'door',
                          'relation': 'above', 'phrase': 'above the door', 'scope': 'relative_anchor'}]})
        self.assertEqual(size_conditions(parsed, 'window')[0]['scope_kind'], 'full_category')
        self.assertEqual(parsed['Relations'][0]['relation'], 'above')

    def test_room_scope_assumption_is_logged_and_named_subroom_not_recovered(self):
        query = 'The tallest shelf in the room'
        parsed = validate_structure(query, raw('shelf', 'tallest', ['in the room']))
        self.assertEqual(parsed['Unhandled'], [])
        self.assertIn('NR3D', parsed['Attributes'][0]['scene_scope_assumption'])
        query = 'The tallest shelf in the bedroom'
        parsed = validate_structure(query, raw('shelf', 'tallest', ['in the bedroom']))
        self.assertIn('in the bedroom', parsed['Unhandled'])

    def test_unknown_member_retained(self):
        parsed, spec = self.subset()
        pool, scope = size_condition_scope(parsed,
            [box(1, 'window', 3), box(2, 'window', .3), box(3, 'window', -3), box(4, 'door', 0)],
            'window', spec)
        self.assertEqual(scope['status'], 'unknown')
        self.assertEqual([o['bbox_id'] for o in pool], [1, 2])

    def test_ablation_and_negation_preserve_unhandled(self):
        with patch.dict(os.environ, {'SEEGROUND_SYNTAX_BACKEND': 'none'}):
            parsed = validate_structure('the taller of the two chairs', raw(unhandled=['of the two']))
        self.assertIn('of the two', parsed['Unhandled'])
        query = 'the taller of the two chairs, not the red one and near the table'
        parsed = validate_structure(query, raw(unhandled=['not the red one', 'near the table']))
        self.assertIn('not the red one', parsed['Unhandled'])
        self.assertIn('near the table', parsed['Unhandled'])
        self.assertIn('negation/exclusion is unsupported', parsed['diagnostic_missing'])


if __name__ == '__main__':
    unittest.main()
