import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from eviground.grounding.active_diagnostic import (apply_computed_evidence, _check_payload,
                               active_ground, compute_observation,
                               evidence_request, execute_evidence,
                               _appearance_review_requirements,
                               distance_candidate_evidence,
                               size_candidate_evidence,
                               position_candidate_evidence, render_action,
                               _target_size_scope, visual_attributes)
from eviground.grounding.diagnostic import (diagnose, recover_between_relations, validate_structure,
                        size_conditions, size_condition_scope)


def obj(oid, category, x, y, z=.5):
    return {'bbox_id': oid, 'target': category,
            'bbox_3d': [x, y, z, .4, .4, 1.]}


def parsed():
    return {
        'Target': ['window'], 'Anchor': [], 'Relations': [],
        'Observer': {'origin': {}, 'facing': {}}, 'Viewpoint': '',
        'Attributes': [], 'Unhandled': [], 'diagnostic_missing': [],
    }


class ActiveDiagnosticTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict('os.environ', {'EVIGROUND_REACT_BACKEND': 'qwen'})
        env.start()
        self.addCleanup(env.stop)

    def test_visual_payload_preserves_category_separately_from_attribute_failure(self):
        raw = {'checks': [{'id': 1, 'category': 'window', 'role': 'target',
                          'category_verdict': 'yes', 'verdict': 'no',
                          'visibility': 'visible', 'reason': 'Window is blue, not red',
                          'attribute_checks': [{'attribute': 'red', 'verdict': 'no',
                                                'visibility': 'visible', 'reason': 'Blue surface'}]}]}
        data = parsed()
        data['Attributes'] = [{'category': 'window', 'phrase': 'red'}]
        requirement = {'id': 1, 'category': 'window', 'role': 'target',
                       'attributes': ['red'], 'source_attributes': ['red']}
        result, = _check_payload(raw, [1], [obj(1, 'window', 0, 0)], data, [requirement])
        self.assertEqual(result['category_verdict'], 'yes')
        self.assertEqual(result['attribute_checks'][0]['verdict'], 'no')
        self.assertEqual(result['verdict'], 'no')

    def test_isolated_tiles_bind_original_instance_pixels_and_expose_missing_instance(self):
        objects = [obj(1, 'window', -10, 0), obj(2, 'window', 10, 0), obj(3, 'window', 30, 0)]
        offsets = np.array([[x, 0, z] for x in (-.15, 0, .15) for z in (.2, .5, .8)])
        xyz = np.concatenate([offsets + [-10, 0, 0], offsets + [10, 0, 0]])
        rgb = np.repeat(np.array([[20, 60, 180], [30, 160, 40]], dtype=np.uint8), 9, axis=0)
        ids = np.repeat([1, 2], 9)
        with tempfile.TemporaryDirectory() as directory:
            result = render_action({'focus_ids': [1, 2, 3], 'azimuth_deg': 90,
                                    'elevation_deg': 25, 'context': 'isolated'},
                                   objects, xyz, rgb, ids, Path(directory), 1)
            self.assertEqual(result['unmarked_focus_ids'], [3])
            self.assertEqual(result['views_rendered'], 2)
            self.assertIn('no shared spatial frame', result['layout'])
            with Image.open(result['image']) as composite:
                self.assertEqual(composite.size, (1800, 600))
                for panel in result['panels']:
                    oid = panel['id']
                    self.assertEqual(panel['marked_ids'], [oid])
                    self.assertEqual(set(panel['visible_instance_pixels']), {str(oid)})
                    self.assertEqual(panel['at'], objects[oid-1]['bbox_3d'][:3])
                    with Image.open(panel['image']) as tile:
                        pixels = np.asarray(tile)
                        colors = set(map(tuple, pixels.reshape(-1, 3)))
                        self.assertIn(tuple(rgb[(oid-1)*9]), colors)
                        self.assertNotIn(tuple(rgb[(2-oid)*9]), colors)
                        left, top = panel['tile_xy_px']
                        pasted = composite.crop((left, top, left+600, top+600))
                        self.assertTrue(np.array_equal(pixels, np.asarray(pasted)))

    def run_actions(self, actions, limit=4, compute=None, visual=None, render=None, calls=None):
        objects = [obj(1, 'window', 0, 0), obj(2, 'window', 2, 0)]
        messages_seen = []
        responses = iter(actions)
        visual_responses = iter(visual or [])
        def fake_chat(messages, **kwargs):
            if messages[0]['content'].startswith('Verify only'):
                reply = next(visual_responses)
                if isinstance(reply, Exception):
                    raise reply
                return {'_content': json.dumps(reply),
                        '_baseline': {'model': 'mock', 'latency_s': .01},
                        '_finish_reason': 'stop', 'usage': {}}
            messages_seen.append(messages)
            reply = next(responses)
            if '_content' in reply:
                return reply
            return {'_content': json.dumps(reply),
                    '_baseline': {'model': 'mock', 'latency_s': .01},
                    '_finish_reason': 'stop', 'usage': {}}
        with tempfile.TemporaryDirectory() as directory, patch(
                'eviground.grounding.active_diagnostic.chat', side_effect=fake_chat), patch(
                'eviground.grounding.active_diagnostic.render_action', side_effect=render or (lambda action, *args: {
                    'tool': 'render', 'image': 'mock.png', 'focus_ids': action['focus_ids']})), patch(
                'eviground.grounding.active_diagnostic.image_part', return_value={'type': 'text', 'text': 'mock image'}), patch(
                'eviground.grounding.active_diagnostic.compute_observation', wraps=compute or compute_observation):
            trace = {}
            calls = [] if calls is None else calls
            result = active_ground(
                'window', parsed(), objects, objects, objects,
                np.zeros((2, 3)), np.zeros((2, 3), dtype=np.uint8),
                np.array([1, 2]), Path(directory), calls, trace, limit, True)
        return result, messages_seen

    def test_explicit_stop_preserves_unresolved_candidates(self):
        result, _ = self.run_actions([{'action': 'stop', 'reason': 'no distinguishing evidence'}])
        self.assertIsNone(result['predicted_id'])
        self.assertIsNone(result['bbox_3d'])
        self.assertFalse(result['accepted'])
        self.assertEqual(result['decision_status'], 'missing')
        self.assertEqual(result['active_trace']['final_candidate_ids'], [1, 2])

    def test_stop_before_bound_size_comparison_requests_actual_evidence_once(self):
        query = 'small box'
        data = validate_structure(query, {'Target': ['box'], 'Anchor': [], 'Relations': [],
            'Observer': {}, 'Attributes': [{'category': 'box', 'phrase': 'small'}], 'Unhandled': []})
        objects = [obj(24, 'box', 0, 0), obj(81, 'box', 2, 0)]
        actions = iter([{'action': 'stop', 'reason': 'no useful evidence beyond size'},
                        {'action': 'evidence', 'dimension': 'size', 'reason': 'measure frozen pool'},
                        {'action': 'stop', 'reason': 'measurements remain unknown'}])
        messages_seen = []
        def reply(messages, **kwargs):
            messages_seen.append(messages)
            return {'_content': json.dumps(next(actions)), '_finish_reason': 'stop',
                    '_baseline': {'model': 'mock', 'latency_s': .01}, 'usage': {}}
        with tempfile.TemporaryDirectory() as directory, patch('eviground.grounding.active_diagnostic.chat', side_effect=reply):
            result = active_ground(query, data, objects, objects, objects,
                np.zeros((2, 3)), np.zeros((2, 3), dtype=np.uint8), np.array([24, 81]),
                Path(directory), [], {}, 2, True)
        trace = result['active_trace']
        self.assertEqual(trace['rejected_actions'], 1)
        self.assertEqual(trace['successful_tools'], 1)
        self.assertEqual(trace['action_attempts'], 3)
        self.assertEqual(trace['stop_reason'], 'model_stop')
        self.assertEqual(trace['final_candidate_ids'], [24, 81])
        self.assertTrue(trace['final_evidence_state']['pending_size_conditions'])
        self.assertFalse(result['accepted'])
        self.assertIn('size has not been executed', messages_seen[1][0]['content'])

    def test_descriptive_or_unbound_size_does_not_force_comparison_on_stop(self):
        for query, word in (('tall box', 'tall'), ('smaller box than the shelf', 'smaller')):
            with self.subTest(query=query):
                data = validate_structure(query, {'Target': ['box'], 'Anchor': [], 'Relations': [],
                    'Observer': {}, 'Attributes': [{'category': 'box', 'phrase': word}], 'Unhandled': []})
                objects = [obj(24, 'box', 0, 0), obj(81, 'box', 2, 0)]
                response = {'_content': json.dumps({'action': 'stop', 'reason': 'unsupported evidence'}),
                            '_finish_reason': 'stop', '_baseline': {'model': 'mock', 'latency_s': .01}}
                with tempfile.TemporaryDirectory() as directory, patch('eviground.grounding.active_diagnostic.chat', return_value=response):
                    result = active_ground(query, data, objects, objects, objects,
                        np.zeros((2, 3)), np.zeros((2, 3), dtype=np.uint8), np.array([24, 81]),
                        Path(directory), [], {}, 2, True)
                self.assertEqual(result['active_trace']['action_attempts'], 1)
                self.assertEqual(result['active_trace']['successful_tools'], 0)
                self.assertEqual(result['active_trace']['rejected_actions'], 0)
                self.assertFalse(result['accepted'])

    def test_visual_array_response_preserves_id_and_visibility_guards(self):
        for invalid in (None, 'unknown_id', 'low_detail', 'duplicate', 'scalar'):
            with self.subTest(invalid=invalid):
                checks = [{'id': 1, 'verdict': 'yes', 'visibility': 'visible',
                           'reason': 'visible window'},
                          {'id': 2, 'verdict': 'no', 'visibility': 'visible',
                           'reason': 'visible category contradiction'}]
                if invalid == 'unknown_id':
                    checks[1]['id'] = 999
                elif invalid == 'low_detail':
                    checks[1]['visibility'] = 'low_detail'
                elif invalid == 'duplicate':
                    checks.append(dict(checks[0]))
                elif invalid == 'scalar':
                    checks = None
                actions = [{'action': 'evidence', 'dimension': 'appearance', 'reason': 'inspect'},
                           {'action': 'finish', 'predicted_id': 1, 'reason': 'visible support'}]
                if invalid:
                    actions.append({'action': 'stop', 'reason': 'insufficient evidence'})
                result, _ = self.run_actions(actions, visual=[checks])
                self.assertEqual(result['accepted'], invalid is None)
                if invalid:
                    self.assertEqual(result['active_trace']['final_candidate_ids'], [1, 2])

    def test_other_two_between_can_finish_on_applicable_hypotheses(self):
        objects=[obj(6,'box',0,0), obj(7,'box',-2,0), obj(8,'box',2,0)]
        query="It's the box that's between the other two boxes"
        parsed_query=validate_structure(query, recover_between_relations(query, {
            'Target':['box'], 'Anchor':['box','box'], 'Relations':[],
            'Observer':{}, 'Attributes':[], 'Unhandled':[]}))
        responses=iter([{'action':'finish','predicted_id':6,
                         'reason':'only box supported between the other two'}])
        def fake_chat(messages, **kwargs):
            return {'_content':json.dumps(next(responses)),
                    '_baseline':{'model':'mock','latency_s':.01},
                    '_finish_reason':'stop','usage':{}}
        with tempfile.TemporaryDirectory() as directory, patch(
                'eviground.grounding.active_diagnostic.chat', side_effect=fake_chat):
            result=active_ground(
                query, parsed_query, objects, objects, objects,
                np.array([[0.,0.,0.],[-2.,0.,0.],[2.,0.,0.]]),
                np.zeros((3,3), dtype=np.uint8), np.array([6,7,8]),
                Path(directory), [], {}, 2, True)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['predicted_id'], 6)
        self.assertEqual(result['decision_status'], 'resolved')

    def test_jev_confidence_cannot_bypass_unresolved_finish_guard(self):
        decisions = iter(['finish', '1', 'stop'])
        def decision(backend, state, questions):
            choice = next(decisions)
            return {'model': 'jev-mock', 'latency_s': .01, 'validation_warnings': [],
                    'answers': {key: {'type': 'choice', 'choice': choice, 'confidence': 1.,
                        'probabilities': {option: float(option == choice) for option in q['criteria']}}
                        for key, q in questions.items()}}
        with patch.dict('os.environ', {'EVIGROUND_REACT_BACKEND': 'jev'}), patch(
                'eviground.grounding.react_decision.decide', side_effect=decision):
            result, messages = self.run_actions([], limit=2)
        self.assertEqual(messages, [])
        self.assertEqual(result['active_trace']['rejected_actions'], 1)
        self.assertEqual(result['active_trace']['successful_tools'], 0)
        self.assertEqual(result['active_trace']['final_candidate_ids'], [1, 2])
        self.assertIsNone(result['predicted_id'])
        self.assertFalse(result['accepted'])

    def test_qwen_planner_also_receives_text_only_after_visual_evidence(self):
        result, messages = self.run_actions([
            {'action': 'evidence', 'dimension': 'appearance', 'reason': 'inspect'},
            {'action': 'stop', 'reason': 'still uncertain'},
        ], visual=[{'checks': [{'id': 1, 'verdict': 'uncertain', 'reason': 'occluded'}]}])
        for message in messages:
            self.assertTrue(all(part['type'] == 'text' for part in message[1]['content']))
            self.assertNotIn('mock.png', json.dumps(message))
        self.assertEqual(result['images'], ['mock.png'])

    def test_invalid_finish_can_recover(self):
        result, messages = self.run_actions([
            {'action': 'finish', 'predicted_id': 99, 'reason': 'invalid ID'},
            {'action': 'stop', 'reason': 'cannot distinguish candidates'},
        ])
        self.assertEqual(result['active_trace']['rejected_actions'], 1)
        self.assertEqual(result['active_trace']['successful_tools'], 0)
        self.assertEqual(result['active_trace']['action_attempts'], 2)
        self.assertIn('Controller correction', messages[1][0]['content'])

    def test_truncated_planner_action_is_discarded_and_can_stop(self):
        truncated = {'_content': json.dumps({
            'action': 'evidence', 'dimension': 'size', 'reason': 'incomplete decision'}),
            '_finish_reason': 'length', '_baseline': {'model': 'mock', 'latency_s': .01},
            'usage': {'completion_tokens': 900}}
        calls = []
        with patch('eviground.grounding.active_diagnostic.parse_json_object', wraps=json.loads) as decode:
            result, messages = self.run_actions([
                truncated, {'action': 'stop', 'reason': 'no useful evidence'}], calls=calls)
        self.assertEqual(decode.call_count, 1)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['active_trace']['successful_tools'], 0)
        self.assertEqual(result['active_trace']['rejected_actions'], 1)
        self.assertEqual(result['active_trace']['action_attempts'], 2)
        self.assertEqual(result['active_trace']['stop_reason'], 'model_stop')
        self.assertNotIn('action', result['active_trace']['steps'][0])
        self.assertTrue(result['active_trace']['steps'][0]['observation']['no_evidence_acquired'])
        self.assertIn('truncated', messages[1][0]['content'])
        self.assertIn('40 words', messages[1][0]['content'])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]['response'], truncated['_content'])
        self.assertEqual(calls[0]['finish_reason'], 'length')
        self.assertEqual(calls[0]['usage'], {'completion_tokens': 900})
        self.assertIn('no action was executed', calls[0]['error'])

    def test_truncated_finish_cannot_prevent_later_valid_evidence(self):
        truncated = {'_content': json.dumps({
            'action': 'finish', 'predicted_id': 2, 'reason': 'truncated guess'}),
            '_finish_reason': 'length', '_baseline': {'model': 'mock', 'latency_s': .01},
            'usage': {}}
        result, _ = self.run_actions([
            truncated,
            {'action': 'evidence', 'dimension': 'appearance', 'reason': 'compare alternatives'},
            {'action': 'finish', 'predicted_id': 1, 'reason': 'alternative excluded'},
        ], visual=[{'checks': [
            {'id': 1, 'verdict': 'yes', 'reason': 'visible match'},
            {'id': 2, 'verdict': 'no', 'reason': 'visible mismatch'}]}])
        self.assertTrue(result['accepted'])
        self.assertEqual(result['predicted_id'], 1)
        self.assertEqual(result['active_trace']['successful_tools'], 1)
        self.assertEqual(result['active_trace']['rejected_actions'], 1)

    def test_repeated_truncated_planner_stops_at_existing_attempt_budget(self):
        truncated = {'_content': '{"action":"finish","predicted_id":1',
            '_finish_reason': 'length', '_baseline': {'model': 'mock', 'latency_s': .01},
            'usage': {}}
        with patch('eviground.grounding.active_diagnostic.parse_json_object') as decode:
            result, messages = self.run_actions([truncated] * 7, limit=2)
        decode.assert_not_called()
        self.assertFalse(result['accepted'])
        self.assertIsNone(result['predicted_id'])
        self.assertEqual(len(messages), 7)
        self.assertEqual(result['active_trace']['action_attempts'], 7)
        self.assertEqual(result['active_trace']['rejected_actions'], 7)
        self.assertEqual(result['active_trace']['successful_tools'], 0)
        self.assertEqual(result['active_trace']['stop_reason'], 'action_attempt_budget_exhausted')

    def test_malformed_stop_json_is_rejected_without_evidence_or_candidate_loss(self):
        # A normal stop finish_reason does not guarantee a well-formed action.
        malformed = {'_content': '{"action":"finish","predicted_id":1,"reason":"guess without evidence}',
                     '_finish_reason': 'stop', '_baseline': {'model': 'mock', 'latency_s': .01},
                     'usage': {}}
        calls = []
        result, messages = self.run_actions([malformed, {'action': 'stop', 'reason': 'still unresolved'}],
                                           calls=calls)
        self.assertFalse(result['accepted'])
        self.assertIsNone(result['predicted_id'])
        self.assertEqual(result['active_trace']['final_candidate_ids'], [1, 2])
        self.assertEqual(result['active_trace']['successful_tools'], 0)
        self.assertEqual(result['active_trace']['rejected_actions'], 1)
        self.assertEqual(calls[0]['response'], malformed['_content'])
        self.assertIn('error', calls[0])
        self.assertIn('valid JSON action', messages[1][0]['content'])

    def test_repeated_malformed_actions_cannot_escape_existing_attempt_budget(self):
        malformed = {'_content': '{"action":"finish","predicted_id":1,"reason":"unclosed}',
                     '_finish_reason': 'stop', '_baseline': {'model': 'mock', 'latency_s': .01},
                     'usage': {}}
        result, messages = self.run_actions([malformed] * 7, limit=2)
        self.assertFalse(result['accepted'])
        self.assertIsNone(result['predicted_id'])
        self.assertEqual(result['active_trace']['final_candidate_ids'], [1, 2])
        self.assertEqual(result['active_trace']['successful_tools'], 0)
        self.assertEqual(result['active_trace']['action_attempts'], 7)
        self.assertEqual(result['active_trace']['rejected_actions'], 7)
        self.assertEqual(len(messages), 7)
        self.assertEqual(result['active_trace']['stop_reason'], 'action_attempt_budget_exhausted')

    def test_yes_for_one_candidate_requires_comparison_before_finish(self):
        result, messages = self.run_actions([
            {'action': 'evidence', 'dimension': 'appearance', 'focus_ids': [1], 'reason': 'check first'},
            {'action': 'finish', 'predicted_id': 1, 'reason': 'first matches'},
            {'action': 'evidence', 'dimension': 'appearance', 'focus_ids': [2], 'reason': 'check alternative'},
            {'action': 'finish', 'predicted_id': 1, 'reason': 'alternative excluded'},
        ], visual=[{'checks': [{'id': 1, 'verdict': 'yes', 'reason': 'visible match'}]},
                   {'checks': [{'id': 2, 'verdict': 'no', 'reason': 'visible mismatch'}]}])
        self.assertIn('Premature finish', messages[2][0]['content'])
        self.assertIn('evidence operations remaining', messages[2][0]['content'])
        correction = messages[2][0]['content'].split('The last observation suggests: ', 1)[1]
        self.assertIn('"focus_ids": [2]', correction)
        self.assertEqual(result['active_trace']['rejected_actions'], 1)
        self.assertEqual(result['active_trace']['final_candidate_ids'], [1])
        self.assertTrue(result['accepted'])

    def test_budget_finish_remains_unaccepted_with_multiple_candidates(self):
        result, _ = self.run_actions([
            {'action': 'evidence', 'dimension': 'size', 'reason': 'inspect'},
            {'action': 'finish', 'predicted_id': 1, 'reason': 'budget guess'},
        ], limit=1)
        self.assertEqual(result['predicted_id'], 1)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['decision_status'], 'missing')
        self.assertEqual(result['active_trace']['stop_reason'], 'budget_final_choice')

    def test_repeated_actions_stop_at_attempt_limit(self):
        action = {'action': 'evidence', 'dimension': 'size', 'reason': 'inspect'}
        result, _ = self.run_actions([action] * 7, limit=2)
        trace = result['active_trace']
        self.assertEqual(trace['stop_reason'], 'action_attempt_budget_exhausted')
        self.assertEqual(trace['successful_tools'], 1)
        self.assertEqual(trace['rejected_actions'], 6)
        self.assertEqual(trace['action_attempts'], 7)
        self.assertFalse(result['accepted'])

    def test_ignored_tool_budget_does_not_execute_extra_tool(self):
        result, _ = self.run_actions([
            {'action': 'evidence', 'dimension': 'size', 'reason': 'inspect size'},
            {'action': 'evidence', 'dimension': 'appearance', 'reason': 'inspect color'},
        ], limit=1)
        self.assertEqual(result['active_trace']['stop_reason'], 'tool_budget_exhausted')
        self.assertEqual(result['active_trace']['successful_tools'], 1)
        self.assertIsNone(result['predicted_id'])

    def test_visual_check_cannot_revive_excluded_candidate(self):
        result, messages = self.run_actions([
            {'action': 'evidence', 'dimension': 'appearance', 'focus_ids': [1], 'reason': 'exclude'},
            {'action': 'evidence', 'dimension': 'appearance', 'focus_ids': [2], 'reason': 'check remaining'},
            {'action': 'stop', 'reason': 'remaining candidate not verified'},
        ], visual=[{'checks': [{'id': 1, 'verdict': 'no', 'reason': 'visible mismatch'}]},
                   {'checks': [{'id': 1, 'verdict': 'yes', 'reason': 'changed mind'}]}])
        self.assertEqual(result['active_trace']['final_candidate_ids'], [2])
        self.assertIn('appearance verification failed',
                      result['active_trace']['steps'][1]['observation']['missing'][0])
        update = result['active_trace']['steps'][0]['observation']['candidate_update']
        self.assertEqual(update, {'before': [1, 2], 'removed_ids': [1], 'after': [2]})
        state = json.loads(messages[2][1]['content'][-1]['text'])
        self.assertEqual(state['evidence_state']['remaining_candidate_ids'], [2])

    def test_legacy_actions_are_rejected_without_using_budget(self):
        result, _ = self.run_actions([
            {'action': 'compute', 'dimension': 'position', 'reason': 'legacy'},
            {'action': 'render', 'reason': 'legacy'},
            {'action': 'visual_check', 'checks': [], 'reason': 'legacy'},
            {'action': 'stop', 'reason': 'stop after invalid requests'},
        ])
        trace = result['active_trace']
        self.assertEqual(trace['rejected_actions'], 3)
        self.assertEqual(trace['successful_tools'], 0)
        self.assertEqual(trace['internal_usage'], {'compute': 0, 'render': 0, 'visual_check': 0})

    def test_evidence_can_repeat_after_candidate_change(self):
        request = {'action': 'evidence', 'dimension': 'position', 'reason': 'check relation'}
        result, messages = self.run_actions([
            request,
            {'action': 'evidence', 'dimension': 'appearance', 'reason': 'compare targets'},
            request,
            {'action': 'finish', 'predicted_id': 2, 'reason': 'combined evidence'},
        ], visual=[{'checks': [
            {'id': 1, 'verdict': 'no', 'reason': 'visible contradiction'},
            {'id': 2, 'verdict': 'yes', 'reason': 'visible match'}]}])
        self.assertEqual(result['active_trace']['successful_tools'], 3)
        self.assertEqual(result['active_trace']['rejected_actions'], 0)
        state = json.loads(messages[2][1]['content'][-1]['text'])
        self.assertIn('position', state['available_dimensions'])
        self.assertIn('position', state['used_dimensions'])
        self.assertEqual(state['remaining_candidate_ids'], [2])

    def test_visual_failure_preserves_candidates_and_accounts_internal_call(self):
        result, _ = self.run_actions([
            {'action': 'evidence', 'dimension': 'appearance', 'reason': 'inspect'},
            {'action': 'finish', 'predicted_id': 1, 'reason': 'budget guess'},
        ], limit=1, visual=[ValueError('mock verification failure')])
        trace = result['active_trace']
        self.assertEqual(trace['final_candidate_ids'], [1, 2])
        self.assertEqual(trace['successful_tools'], 1)
        self.assertEqual(trace['internal_usage'], {'compute': 1, 'render': 1, 'visual_check': 1})
        observation = trace['steps'][0]['observation']
        self.assertEqual(observation['coverage']['unknown_ids'], [1, 2])
        self.assertEqual(observation['operation_status'], 'partial')
        self.assertEqual(observation['internal_steps'][-1]['call']['error'], 'mock verification failure')
        self.assertFalse(result['accepted'])

    def test_camera_change_is_a_new_appearance_request(self):
        result, _ = self.run_actions([
            {'action': 'evidence', 'dimension': 'appearance', 'azimuth_deg': 90, 'reason': 'first view'},
            {'action': 'evidence', 'dimension': 'appearance', 'azimuth_deg': 180, 'reason': 'new view'},
            {'action': 'stop', 'reason': 'still occluded'},
        ], visual=[{'checks': [{'id': 1, 'verdict': 'uncertain', 'reason': 'occluded'}]}] * 2)
        self.assertEqual(result['active_trace']['successful_tools'], 2)
        self.assertEqual(result['active_trace']['rejected_actions'], 0)

    def test_unmarked_target_cannot_receive_a_visual_no(self):
        result, _ = self.run_actions([
            {'action': 'evidence', 'dimension': 'appearance', 'reason': 'inspect'},
            {'action': 'stop', 'reason': 'unmarked target needs another view'},
        ], visual=[{'checks': [
            {'id': 1, 'verdict': 'no', 'reason': 'invented contradiction'}]}],
            render=lambda action, *args: {'tool': 'render', 'image': 'mock.png',
                                          'focus_ids': [1, 2], 'unmarked_focus_ids': [1]})
        self.assertEqual(result['active_trace']['final_candidate_ids'], [1, 2])
        self.assertEqual(result['active_trace']['steps'][0]['observation']['coverage']['unknown_ids'], [1, 2])

    def test_default_appearance_group_reports_uninspected_candidates(self):
        objects = [obj(index, 'window', index, 0) for index in range(1, 9)]
        request = evidence_request({'action': 'evidence', 'dimension': 'appearance', 'reason': 'inspect'},
                                   list(range(1, 9)), objects)
        diagnosis = diagnose(parsed(), objects, objects, [0, 0, .5], trusted_categories=True)
        reply = {'_content': json.dumps({'checks': [
            {'id': index, 'verdict': 'yes', 'reason': 'visible match'} for index in range(1, 7)]}),
            '_finish_reason': 'stop', '_baseline': {'latency_s': .01}, 'usage': {}}
        with patch('eviground.grounding.active_diagnostic.render_action', return_value={
                'tool': 'render', 'image': 'mock.png', 'focus_ids': list(range(1, 7))}), patch(
                'eviground.grounding.active_diagnostic.image_part', return_value={'type': 'text', 'text': 'image'}), patch(
                'eviground.grounding.active_diagnostic.chat', return_value=reply):
            result = execute_evidence(request, 'window', parsed(), diagnosis, objects, objects,
                                      None, None, None, Path('/tmp'), 1, [])
        self.assertEqual(result['coverage']['uninspected_ids'], [7, 8])
        self.assertEqual(result['operation_status'], 'partial')
        self.assertEqual(result['next_evidence'][0]['focus_ids'], [7, 8])
        for suggestion in result['next_evidence']:
            self.assertEqual(evidence_request(suggestion, list(range(1, 9)), objects), suggestion)
        self.assertEqual(diagnosis['candidate_ids'], list(range(1, 9)))

    def test_summary_only_evidence_does_not_certify_candidates(self):
        result, _ = self.run_actions([
            {'action': 'evidence', 'dimension': 'size', 'reason': 'measure'},
            {'action': 'stop', 'reason': 'measurements insufficient'},
        ])
        observation = result['active_trace']['steps'][0]['observation']
        self.assertEqual(observation['operation_status'], 'information_only')
        self.assertTrue(all(item['status'] == 'not_applicable'
                            for item in observation['candidate_evidence']))
        self.assertEqual(observation['coverage']['uninspected_ids'], [1, 2])

    def test_distance_operation_preserves_mixed_known_unknown_measurements(self):
        targets = [obj(1, 'window', 1, 2), obj(2, 'window', 8, 2)]
        wall = obj(3, 'wall', 0, 0)
        wall['wall_geometry'] = {
            'valid': True, 'origin_xy_m': [0, 0], 'normal_xy': [0, 1],
            'tangent_xy': [1, 0], 'tangent_extent_m': [0, 4],
        }
        query = 'the window nearest the wall'
        data = validate_structure(query, {
            **parsed(), 'Anchor': ['wall'],
            'Relations': [{'target_category': 'window', 'anchor_category': 'wall',
                           'relation': 'nearest', 'metric': 'plane',
                           'phrase': 'nearest the wall'}],
        })
        diagnosis = diagnose(data, targets + [wall], targets, [0, 0, .5],
                             trusted_categories=True)
        observation = compute_observation('distance', query, data, diagnosis,
                                          targets, targets + [wall])
        self.assertEqual([row['status'] for row in observation['candidate_evidence']],
                         ['unknown', 'unknown'])
        ranking = observation['distance_rankings']['rankings'][0]
        self.assertEqual(ranking['unknown_target_ids'], [2])
        self.assertEqual(ranking['best_ids'], [])
        update = apply_computed_evidence(diagnosis, observation['candidate_evidence'],
                                         'distance')
        self.assertEqual(update, {'before': [1, 2], 'removed_ids': [], 'after': [1, 2]})
        self.assertEqual(diagnosis['status'], 'missing')

    def test_distance_operation_uses_bound_observer_hypotheses(self):
        targets = [obj(1, 'window', 1, 2), obj(2, 'window', 9, 2)]
        objects = targets + [obj(3, 'tv', 5, 10), obj(4, 'door', 0, 0),
                             obj(5, 'door', 10, 0)]
        query = 'Standing at the door facing the tv, the window nearest the tv on the right.'
        data = validate_structure(query, {
            **parsed(),
            'Relations': [
                {'target_category': 'window', 'anchor_category': 'tv',
                 'relation': 'nearest', 'metric': 'viewpoint', 'phrase': 'nearest the tv'},
                {'target_category': 'window', 'anchor_category': '',
                 'relation': 'right', 'scope': 'side', 'phrase': 'on the right'},
            ],
            'Observer': {
                'origin': {'kind': 'category', 'category': 'door', 'phrase': 'at the door'},
                'facing': {'category': 'tv', 'group': False, 'phrase': 'facing the tv'},
            },
        })
        diagnosis = diagnose(data, objects, targets, [0, 0, .5],
                             trusted_categories=True)
        observation = compute_observation('distance', query, data, diagnosis,
                                          targets, objects)
        self.assertEqual([ranking['best_ids'] for ranking in
                          observation['distance_rankings']['rankings']], [[1], [2]])
        self.assertEqual([row['status'] for row in observation['candidate_evidence']],
                         ['pass', 'pass'])
        self.assertEqual(observation['distance_rankings']['ignored'], [])

    def test_distance_operation_does_not_retry_incompatible_reference_bindings(self):
        targets = [obj(1, 'window', 1, 0), obj(2, 'window', 9, 0)]
        anchors = [obj(3, 'tv', 0, 0), obj(4, 'tv', 10, 0)]
        query = 'the window nearest the tv and farthest from the tv'
        data = validate_structure(query, {
            **parsed(), 'Relations': [
                {'target_category': 'window', 'anchor_category': 'tv',
                 'relation': 'nearest', 'phrase': 'nearest the tv', 'anchor_id': 3},
                {'target_category': 'window', 'anchor_category': 'tv',
                 'relation': 'farthest', 'phrase': 'farthest from the tv', 'anchor_id': 4},
            ],
        })
        diagnosis = diagnose(data, targets + anchors, targets, [0, 0, .5],
                             trusted_categories=True)
        observation = compute_observation('distance', query, data, diagnosis,
                                          targets, targets + anchors)
        self.assertEqual(observation['distance_rankings']['rankings'], [])
        update = apply_computed_evidence(diagnosis, observation['candidate_evidence'],
                                         'distance')
        self.assertEqual(update['removed_ids'], [])
        self.assertEqual(diagnosis['candidate_ids'], [1, 2])
        self.assertEqual(diagnosis['status'], 'missing')

    def test_skinny_stays_an_appearance_condition_not_a_size_extreme(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 2, 0)]
        measurements = [
            {'id': 1, 'status': 'known', 'volume_m3': 2., 'height_m': 1.,
             'horizontal_aspect_ratio': 4.},
            {'id': 2, 'status': 'known', 'volume_m3': 1., 'height_m': 1.,
             'horizontal_aspect_ratio': 1.},
        ]
        query = parsed()
        query['Target'] = ['cabinet']
        query['Attributes'] = [{'category': 'cabinet', 'phrase': 'skinny'}]
        evidence, constraints, decision = size_candidate_evidence(
            query, objects, measurements)
        self.assertEqual(constraints, [])
        self.assertEqual(decision['attributes'], [])
        self.assertEqual(decision['candidate_relative_ids'], [])
        self.assertEqual([row['status'] for row in evidence],
                         ['not_applicable', 'not_applicable'])
        self.assertTrue(all(item['attribute_evidence'] == [] for item in evidence))

    def test_compound_tall_remains_unknown_after_complete_skinny_visual_evidence(self):
        objects = [obj(5, 'shelf', 0, 0), obj(13, 'shelf', 2, 0)]
        query = parsed()
        query['Target'] = ['shelf']
        query['Attributes'] = [{'category': 'shelf', 'phrase': 'tall skinny'}]
        xyz = np.asarray([[0., 0., 0.], [.24, .83, .94],
                          [2., 0., 0.], [2.45, .70, 1.56]])
        ids = np.asarray([5, 5, 13, 13])
        diagnosis = diagnose(query, objects, objects, [0, 0, 0], trusted_categories=True)
        size = compute_observation('size', 'tall skinny shelf', query, diagnosis,
                                   objects, objects, xyz, ids)
        hints = size['size_decision']['attributes']
        self.assertEqual([(h['attribute'], h['selected_ids']) for h in hints],
                         [('tall', [])])
        self.assertEqual(hints[0]['interpretation'], 'descriptive_attribute')
        self.assertEqual(hints[0]['status'], 'unknown')
        self.assertEqual(size['size_decision']['candidate_relative_ids'], [])
        response = {'_content': json.dumps({'checks': [
            {'id': 5, 'verdict': 'yes', 'reason': 'looks tall and thin'},
            {'id': 13, 'verdict': 'no', 'reason': 'looks blocky'}]}),
            '_finish_reason': 'stop', '_baseline': {'latency_s': .01}, 'usage': {}}
        action = evidence_request({'action': 'evidence', 'dimension': 'appearance',
                                   'context': 'scene', 'reason': 'inspect'}, [5, 13], objects)
        with patch('eviground.grounding.active_diagnostic.render_action', return_value={
                'tool': 'render', 'image': 'mock.png', 'focus_ids': [5, 13]}), patch(
                'eviground.grounding.active_diagnostic.image_part', return_value={'type': 'text', 'text': 'image'}), patch(
                'eviground.grounding.active_diagnostic.chat', return_value=response):
            observation = execute_evidence(action, 'tall skinny shelf', query, diagnosis,
                objects, objects, xyz, None, ids, Path('/tmp'), 1, [])
        self.assertEqual([c['verdict'] for c in observation['checks']], ['uncertain', 'uncertain'])
        self.assertEqual([c['visual_verdict'] for c in observation['checks']], ['yes', 'no'])
        after = diagnose(query, objects, objects, [0, 0, 0], observation['checks'],
                         trusted_categories=True)
        self.assertEqual(after['candidate_ids'], [5, 13])
        self.assertNotEqual(after['status'], 'resolved')
        # Complete silhouette evidence does not certify descriptive physical height.
        response['_content'] = json.dumps({'checks': [
            {'id': 5, 'verdict': 'yes', 'visibility': 'visible', 'reason': 'shelf visible', 'attribute_checks': [
                {'attribute': 'skinny', 'verdict': 'uncertain', 'visibility': 'occluded', 'reason': 'side obscured'}]},
            {'id': 13, 'verdict': 'yes', 'visibility': 'visible', 'reason': 'shelf visible', 'attribute_checks': [
                {'attribute': 'skinny', 'verdict': 'yes', 'visibility': 'visible', 'reason': 'narrow upright silhouette'}]}]})
        with patch('eviground.grounding.active_diagnostic.render_action', return_value={
                'tool': 'render', 'image': 'mock2.png', 'focus_ids': [5, 13]}), patch(
                'eviground.grounding.active_diagnostic.image_part', return_value={'type': 'text', 'text': 'image'}), patch(
                'eviground.grounding.active_diagnostic.chat', return_value=response) as api:
            second = execute_evidence(action, 'tall skinny shelf', query, after,
                objects, objects, xyz, None, ids, Path('/tmp'), 2, [])
        payload = json.loads(api.call_args.args[0][1]['content'][1]['text'])
        self.assertNotIn('candidate_measurements', payload)
        self.assertEqual(payload['review_requirements'][0]['attributes'], ['skinny'])
        self.assertEqual([c['verdict'] for c in second['checks']], ['uncertain', 'yes'])
        resolved = diagnose(query, objects, objects, [0, 0, 0],
                            observation['checks'] + second['checks'], trusted_categories=True)
        self.assertEqual(resolved['candidate_ids'], [5, 13])
        self.assertEqual(resolved['status'], 'missing')
        self.assertIn('descriptive tall', ' '.join(resolved['missing']))


    def test_visual_attribute_evidence_requires_complete_unique_coverage(self):
        query = parsed()
        query['Target'] = ['shelf']
        query['Attributes'] = [{'category': 'shelf', 'phrase': 'tall skinny'}]
        tall = {'attribute': 'tall', 'verdict': 'yes', 'reason': 'upright body'}
        skinny = {'attribute': 'skinny', 'verdict': 'uncertain', 'reason': 'occluded'}
        for attrs in ([], [tall], [tall, tall], [tall, skinny]):
            checks = _check_payload({'checks': [{'id': 1, 'verdict': 'yes',
                'reason': 'shelf visible', 'attribute_checks': attrs}]},
                [1], [obj(1, 'shelf', 0, 0)], query)
            self.assertEqual(checks[0]['verdict'], 'uncertain')

    def test_appearance_records_each_candidate_attribute_and_render_provenance(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 2, 0)]
        query = parsed()
        query['Target'] = ['cabinet']
        query['Attributes'] = [{'category': 'cabinet', 'phrase': 'red wooden'}]
        requirements = _appearance_review_requirements(
            query, {'candidate_ids': [1, 2], 'visual_needs': []}, [1, 2], objects)
        checks = _check_payload({'checks': [
            {'id': 1, 'verdict': 'yes', 'visibility': 'visible', 'reason': 'cabinet visible',
             'attribute_checks': [
                 {'attribute': 'red', 'verdict': 'yes', 'visibility': 'visible', 'reason': 'red paint'},
                 {'attribute': 'wooden', 'verdict': 'yes', 'visibility': 'visible', 'reason': 'wood grain'}]},
            {'id': 2, 'verdict': 'yes', 'visibility': 'visible', 'reason': 'cabinet visible',
             'attribute_checks': [
                 {'attribute': 'red', 'verdict': 'yes', 'visibility': 'visible', 'reason': 'red paint'}]}]},
            [1, 2], objects, query, requirements,
            {'image': 'mock.png', 'source_type': 'colored_point_cloud', 'language_frame_changed': False})
        self.assertEqual([check['verdict'] for check in checks], ['yes', 'uncertain'])
        self.assertEqual([len(check['attribute_checks']) for check in checks], [2, 2])
        self.assertTrue(all(check['render_provenance']['source_type'] == 'colored_point_cloud'
                            for check in checks))
        missing = diagnose(query, objects, objects, [0, 0, .5], checks, trusted_categories=True)
        self.assertEqual(missing['status'], 'missing')
        self.assertEqual(missing['candidate_ids'], [1, 2])

    def test_unmarked_attribute_request_keeps_unknown_and_suggests_targeted_view(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 2, 0)]
        query = parsed()
        query['Target'] = ['cabinet']
        query['Attributes'] = [{'category': 'cabinet', 'phrase': 'wooden'}]
        diagnosis = diagnose(query, objects, objects, [0, 0, .5], trusted_categories=True)
        action = evidence_request({'action': 'evidence', 'dimension': 'appearance',
                                   'reason': 'inspect grain'}, [1, 2], objects)
        reply = {'_content': json.dumps({'checks': [
            {'id': 2, 'verdict': 'yes', 'visibility': 'visible', 'reason': 'cabinet visible',
             'attribute_checks': [{'attribute': 'wooden', 'verdict': 'yes',
                                   'visibility': 'low_detail', 'reason': 'grain too small'}]}]}),
            '_finish_reason': 'stop', '_baseline': {'latency_s': .01}, 'usage': {}}
        with patch('eviground.grounding.active_diagnostic.render_action', return_value={
                'tool': 'render', 'image': 'mock.png', 'focus_ids': [1, 2],
                'unmarked_focus_ids': [1]}), patch(
                'eviground.grounding.active_diagnostic.image_part', return_value={'type': 'text', 'text': 'image'}), patch(
                'eviground.grounding.active_diagnostic.chat', return_value=reply):
            result = execute_evidence(action, 'wooden cabinet', query, diagnosis,
                objects, objects, None, None, None, Path('/tmp'), 1, [])
        self.assertEqual(result['coverage']['unknown_ids'], [2, 1])
        unmarked = next(check for check in result['checks'] if check['id'] == 1)
        self.assertEqual(unmarked['visibility'], 'unmarked')
        self.assertEqual(unmarked['attribute_checks'][0]['verdict'], 'uncertain')
        self.assertFalse(unmarked['render_provenance']['language_frame_changed'])
        self.assertEqual(result['coverage']['checked_ids'], [2])
        self.assertEqual(result['coverage']['uninspected_ids'], [1])
        alternate = result['next_evidence'][1]
        self.assertEqual(result['unknown_attributes'], ['wooden'])
        self.assertEqual(alternate['azimuth_deg'], 180.)
        self.assertEqual(evidence_request(alternate, [1, 2], objects), alternate)

    def test_all_unmarked_instances_skip_visual_api_and_keep_unknown_coverage(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 2, 0)]
        query = parsed()
        query['Target'] = ['cabinet']
        query['Attributes'] = [{'category': 'cabinet', 'phrase': 'wooden'}]
        diagnosis = diagnose(query, objects, objects, [0, 0, .5], trusted_categories=True)
        action = evidence_request({'action': 'evidence', 'dimension': 'appearance',
                                   'reason': 'inspect material'}, [1, 2], objects)
        calls = []
        with patch('eviground.grounding.active_diagnostic.render_action', return_value={
                'tool': 'render', 'image': 'mock.png', 'focus_ids': [1, 2],
                'unmarked_focus_ids': [1, 2]}), patch(
                'eviground.grounding.active_diagnostic.image_part') as encode, patch('eviground.grounding.active_diagnostic.chat') as api:
            result = execute_evidence(action, 'wooden cabinet', query, diagnosis,
                objects, objects, None, None, None, Path('/tmp'), 1, calls)
        api.assert_not_called()
        encode.assert_not_called()
        self.assertEqual(calls, [])
        self.assertEqual(result['coverage']['unknown_ids'], [1, 2])
        self.assertEqual(result['operation_status'], 'partial')
        self.assertTrue(all(check['visibility'] == 'unmarked' for check in result['checks']))
        self.assertTrue(result['internal_steps'][-1]['call']['skipped'])
        self.assertIn('no marked instance', result['internal_steps'][-1]['call']['reason'])
        self.assertEqual(result['unknown_attributes'], ['wooden'])
        self.assertEqual(evidence_request(result['next_evidence'][0], [1, 2], objects),
                         result['next_evidence'][0])

    def test_suggested_alternate_camera_renders_new_pixels_without_changing_language_frame(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 2, 0)]
        query = parsed()
        query['Target'] = ['cabinet']
        query['Attributes'] = [{'category': 'cabinet', 'phrase': 'wooden'}]
        diagnosis = diagnose(query, objects, objects, [0, 0, .5], trusted_categories=True)
        diagnosis['hypotheses'][0]['frame'] = {
            'origin': [0, -2, .5], 'forward': [0, 1, 0], 'right': [1, 0, 0]}
        initial_frame = json.dumps(diagnosis['hypotheses'], sort_keys=True)
        rng = np.random.default_rng(7)
        xyz = np.concatenate([rng.uniform([-.2, -.2, 0], [.2, .2, 1], (200, 3)),
                              rng.uniform([1.8, -.2, 0], [2.2, .2, 1], (200, 3))])
        rgb = np.asarray(rng.uniform(0, 255, (400, 3)), dtype=np.uint8)
        ids = np.repeat([1, 2], 200)
        action = evidence_request({'action': 'evidence', 'dimension': 'appearance',
            'azimuth_deg': 310, 'elevation_deg': 35, 'context': 'isolated',
            'reason': 'inspect wooden finish'}, [1, 2], objects)
        reply = {'_content': json.dumps({'checks': [
            {'id': oid, 'verdict': 'yes', 'reason': 'cabinet visible',
             'attribute_checks': [{'attribute': 'wooden', 'verdict': 'uncertain',
                                  'reason': 'grain not clear'}]} for oid in [1, 2]]}),
            '_finish_reason': 'stop', '_baseline': {'latency_s': .01}, 'usage': {}}
        with tempfile.TemporaryDirectory() as directory, patch(
                'eviground.grounding.active_diagnostic.chat', return_value=reply):
            first = execute_evidence(action, 'wooden cabinet', query, diagnosis,
                objects, objects, xyz, rgb, ids, Path(directory), 1, [])
            alternate = evidence_request(first['next_evidence'][0], [1, 2], objects)
            self.assertEqual(alternate['azimuth_deg'], 40.)
            self.assertEqual(alternate['elevation_deg'], 35)
            self.assertEqual(alternate['context'], 'isolated')
            second = execute_evidence(alternate, 'wooden cabinet', query, diagnosis,
                objects, objects, xyz, rgb, ids, Path(directory), 2, [])
            with Image.open(first['images'][0]) as before, Image.open(second['images'][0]) as after:
                self.assertFalse(np.array_equal(np.asarray(before), np.asarray(after)))
        self.assertEqual(json.dumps(diagnosis['hypotheses'], sort_keys=True), initial_frame)
        self.assertTrue(all(not check['render_provenance']['language_frame_changed']
                            for check in first['checks'] + second['checks']))

    def test_reference_followup_is_executable_and_contains_live_target(self):
        objects = [obj(1, 'cabinet', 0, 0)] + [obj(oid, 'tv', oid, 0) for oid in range(2, 10)]
        query = {**parsed(), 'Target': ['cabinet']}
        diagnosis = {'candidate_ids': [1], 'geometry': {}, 'visual_needs': [
            {'id': oid, 'category': 'tv', 'role': 'reference', 'attributes': ['black']}
            for oid in range(2, 10)]}
        action = evidence_request({'action': 'evidence', 'dimension': 'appearance',
            'focus_ids': [1, 2], 'reason': 'inspect reference'}, [1], objects)
        with patch('eviground.grounding.active_diagnostic.render_action', return_value={
                'tool': 'render', 'image': 'mock.png', 'focus_ids': [1, 2],
                'unmarked_focus_ids': [1, 2]}), patch('eviground.grounding.active_diagnostic.chat') as api:
            result = execute_evidence(action, 'cabinet by black tv', query, diagnosis,
                objects[:1], objects, None, None, None, Path('/tmp'), 1, [])
        api.assert_not_called()
        for suggestion in result['next_evidence']:
            self.assertEqual(evidence_request(suggestion, [1], objects), suggestion)
            self.assertIn(1, suggestion['focus_ids'])
        reference = result['next_evidence'][-1]
        self.assertEqual(len(reference['focus_ids']), 6)
        self.assertEqual(result['coverage']['reference_uninspected_ids'], list(range(2, 10)))

    def test_reference_followup_does_not_focus_a_rejected_target(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 2, 0), obj(3, 'tv', 4, 0)]
        query = {**parsed(), 'Target': ['cabinet']}
        diagnosis = {'candidate_ids': [1, 2], 'geometry': {}, 'visual_needs': [
            {'id': 3, 'category': 'tv', 'role': 'reference', 'attributes': ['black']}]}
        action = evidence_request({'action': 'evidence', 'dimension': 'appearance',
            'focus_ids': [1, 2, 3], 'reason': 'inspect reference'}, [1, 2], objects)
        reply = {'_content': json.dumps({'checks': [
            {'id': 1, 'verdict': 'no', 'reason': 'wrong category'},
            {'id': 2, 'verdict': 'yes', 'reason': 'cabinet visible'},
            {'id': 3, 'role': 'reference', 'category': 'tv', 'verdict': 'yes',
             'reason': 'tv visible', 'attribute_checks': [
                {'attribute': 'black', 'verdict': 'uncertain', 'reason': 'occluded'}]}]}),
            '_finish_reason': 'stop', '_baseline': {'latency_s': .01}, 'usage': {}}
        with patch('eviground.grounding.active_diagnostic.render_action', return_value={
                'tool': 'render', 'image': 'mock.png', 'focus_ids': [1, 2, 3]}), patch(
                'eviground.grounding.active_diagnostic.image_part', return_value={'type': 'text', 'text': 'image'}), patch(
                'eviground.grounding.active_diagnostic.chat', return_value=reply):
            result = execute_evidence(action, 'cabinet by black tv', query, diagnosis,
                objects[:2], objects, None, None, None, Path('/tmp'), 1, [])
        self.assertEqual(result['unknown_attributes'], ['black'])
        reference = result['next_evidence'][0]
        self.assertEqual(reference['focus_ids'], [2, 3])
        self.assertEqual(evidence_request(reference, [2], objects), reference)

    def test_controller_rebinds_suggested_focus_after_reference_evidence_filters_targets(self):
        objects = [obj(1, 'window', 0, 0), obj(2, 'window', 2, 0)]
        initial = {'candidate_ids': [1, 2], 'stable_candidate_ids': [], 'missing': ['appearance'],
            'geometry': {}, 'checks': [], 'hypotheses': [{'candidate_ids': [1, 2]}],
            'visual_needs': [{'id': 3, 'role': 'reference'}]}
        refreshed = {**initial, 'candidate_ids': [2], 'hypotheses': [{'candidate_ids': [2]}]}
        observation = {'tool': 'evidence', 'dimension': 'appearance', 'candidate_evidence': [],
            'images': [], 'checks': [{'id': 3, 'category': 'tv', 'role': 'reference', 'verdict': 'yes'}],
            'internal_steps': [], 'next_evidence': [
                {'action': 'evidence', 'dimension': 'appearance', 'focus_ids': [1, 3],
                 'azimuth_deg': 180, 'elevation_deg': 25, 'context': 'scene',
                 'reason': 'inspect remaining reference attribute'}]}
        responses = iter([
            {'action': 'evidence', 'dimension': 'appearance', 'focus_ids': [1, 3], 'reason': 'inspect'},
            {'action': 'stop', 'reason': 'unknown'}])
        def fake_chat(*args, **kwargs):
            return {'_content': json.dumps(next(responses)), '_finish_reason': 'stop',
                    '_baseline': {'model': 'mock', 'latency_s': .01}, 'usage': {}}
        with tempfile.TemporaryDirectory() as directory, patch(
                'eviground.grounding.active_diagnostic.chat', side_effect=fake_chat), patch(
                'eviground.grounding.active_diagnostic.diagnose', side_effect=[initial, refreshed]), patch(
                'eviground.grounding.active_diagnostic.execute_evidence', return_value=observation):
            result = active_ground('window by tv', parsed(), objects, objects + [obj(3, 'tv', 4, 0)],
                objects, np.zeros((2, 3)), np.zeros((2, 3), dtype=np.uint8), np.array([1, 2]),
                Path(directory), [], {}, 2, True)
        suggestion = result['active_trace']['steps'][0]['observation']['next_evidence'][0]
        self.assertEqual(suggestion['focus_ids'], [2, 3])
        self.assertEqual(evidence_request(suggestion, [2], objects + [obj(3, 'tv', 4, 0)]), suggestion)

    def test_skipped_visual_call_does_not_count_as_internal_api_usage(self):
        result, _ = self.run_actions([
            {'action': 'evidence', 'dimension': 'appearance', 'reason': 'inspect'},
            {'action': 'stop', 'reason': 'unmarked targets need another view'},
        ], render=lambda action, *args: {'tool': 'render', 'image': 'mock.png',
                                        'focus_ids': [1, 2], 'unmarked_focus_ids': [1, 2]})
        self.assertEqual(result['active_trace']['internal_usage'],
                         {'compute': 1, 'render': 1, 'visual_check': 0})
        self.assertEqual(result['active_trace']['final_candidate_ids'], [1, 2])

    def test_reference_attributes_are_separate_from_target_candidate_evidence(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'tv', 2, 0)]
        query = parsed()
        query['Target'] = ['cabinet']
        query['Attributes'] = [{'category': 'television', 'phrase': 'black'}]
        diagnosis = {'candidate_ids': [1], 'geometry': {}, 'visual_needs': [
            {'id': 2, 'category': 'tv', 'role': 'reference', 'attributes': ['black']}]}
        action = evidence_request({'action': 'evidence', 'dimension': 'appearance',
                                   'focus_ids': [1, 2], 'reason': 'check reference color'}, [1], objects)
        reply = {'_content': json.dumps({'checks': [
            {'id': 1, 'verdict': 'yes', 'visibility': 'visible', 'reason': 'cabinet visible'},
            {'id': 2, 'category': 'tv', 'role': 'reference', 'verdict': 'yes',
             'visibility': 'visible', 'reason': 'tv visible', 'attribute_checks': [
                 {'attribute': 'black', 'verdict': 'yes', 'visibility': 'visible', 'reason': 'black body'}]}]}),
            '_finish_reason': 'stop', '_baseline': {'latency_s': .01}, 'usage': {}}
        with patch('eviground.grounding.active_diagnostic.render_action', return_value={
                'tool': 'render', 'image': 'mock.png', 'focus_ids': [1, 2]}), patch(
                'eviground.grounding.active_diagnostic.image_part', return_value={'type': 'text', 'text': 'image'}), patch(
                'eviground.grounding.active_diagnostic.chat', return_value=reply) as api:
            result = execute_evidence(action, 'cabinet by black television', query, diagnosis,
                objects[:1], objects, None, None, None, Path('/tmp'), 1, [])
        self.assertEqual([item['candidate_id'] for item in result['candidate_evidence']], [1])
        self.assertEqual(result['candidate_evidence'][0]['attribute_evidence'], [])
        self.assertEqual(result['reference_evidence'][0]['attribute_checks'][0]['attribute'], 'black')
        self.assertEqual(result['coverage']['reference_checked_ids'], [2])
        self.assertEqual(result['coverage']['reference_unknown_ids'], [])
        payload = json.loads(api.call_args.args[0][1]['content'][1]['text'])
        self.assertEqual(payload['review_requirements'][1]['role'], 'reference')

    def test_one_instance_with_multiple_roles_requires_explicit_role_binding(self):
        objects = [obj(1, 'cabinet', 0, 0)]
        query = parsed()
        query['Target'] = ['cabinet']
        diagnosis = {'candidate_ids': [1], 'visual_needs': [
            {'id': 1, 'category': 'cabinet', 'role': 'observer_facing', 'attributes': []}]}
        requirements = _appearance_review_requirements(query, diagnosis, [1], objects)
        self.assertEqual([item['role'] for item in requirements], ['target', 'observer_facing'])
        with self.assertRaises(ValueError):
            _check_payload({'checks': [{'id': 1, 'verdict': 'yes', 'reason': 'cabinet visible'}]},
                           [1], objects, query, requirements)
        checks = _check_payload({'checks': [
            {'id': 1, 'role': role, 'category': 'cabinet', 'verdict': 'yes', 'reason': 'cabinet visible'}
            for role in ('target', 'observer_facing')]}, [1], objects, query, requirements)
        self.assertEqual([item['role'] for item in checks], ['target', 'observer_facing'])

    def test_unbound_reference_attribute_has_separate_visual_requirement(self):
        query_text = 'brown table next to table with rainbow top'
        query = validate_structure(query_text, {
            **parsed(), 'Target': ['table'], 'Anchor': ['table'],
            'Relations': [{'target_category': 'table', 'anchor_category': 'table',
                           'relation': 'other',
                           'phrase': 'next to table with rainbow top'}],
            'Attributes': [{'category': 'table', 'phrase': 'brown'},
                           {'category': 'table', 'phrase': 'rainbow top'}],
        })
        objects = [obj(1, 'table', 0, 0), obj(2, 'table', 2, 0)]
        diagnosis = diagnose(query, objects, objects, [0, 0, .5], trusted_categories=True)
        requirements = _appearance_review_requirements(query, diagnosis, [1], objects)

        self.assertEqual([(item['role'], item['attributes']) for item in requirements], [
            ('target', ['brown']), ('reference', ['rainbow top'])])
        action = evidence_request({'action': 'evidence', 'dimension': 'appearance',
            'focus_ids': [1], 'reason': 'check target and unbound reference'},
            diagnosis['candidate_ids'], objects)
        reply = {'_content': json.dumps({'checks': [{
            'id': 1, 'category': 'table', 'role': 'target', 'verdict': 'yes',
            'visibility': 'visible', 'reason': 'brown table',
            'attribute_checks': [{'attribute': 'brown', 'verdict': 'yes',
                'visibility': 'visible', 'reason': 'brown surface'}]}]}),
            '_finish_reason': 'stop', '_baseline': {'latency_s': .01}, 'usage': {}}
        with patch('eviground.grounding.active_diagnostic.render_action', return_value={
                'tool': 'render', 'image': 'mock.png', 'focus_ids': [1]}), patch(
                'eviground.grounding.active_diagnostic.image_part', return_value={'type': 'text', 'text': 'image'}), patch(
                'eviground.grounding.active_diagnostic.chat', return_value=reply):
            result = execute_evidence(action, query_text, query, diagnosis,
                objects, objects, None, None, None, Path('/tmp'), 1, [])

        target_check = next(check for check in result['checks'] if check['role'] == 'target')
        reference_check = next(check for check in result['checks'] if check['role'] == 'reference')
        self.assertEqual([item['attribute'] for item in target_check['attribute_checks']], ['brown'])
        self.assertEqual(reference_check['verdict'], 'uncertain')
        self.assertEqual(reference_check['attribute_checks'][0]['attribute'], 'rainbow top')
        self.assertEqual(reference_check['attribute_checks'][0]['verdict'], 'uncertain')
        self.assertEqual(reference_check['attribute_checks'][0]['inspection_status'], 'uninspected')
        self.assertEqual(result['coverage']['reference_checked_ids'], [])
        self.assertEqual(result['coverage']['reference_uninspected_ids'], [1, 2])

    def test_shared_instance_visual_roles_use_independent_calls_and_preserve_partial_results(self):
        objects = [obj(1, 'table', 0, 0)]
        query = {**parsed(), 'Target': ['table'], 'Anchor': ['table'],
                 'Attributes': [{'category': 'table', 'role': 'target', 'phrase': 'brown'},
                                {'category': 'table', 'role': 'reference', 'phrase': 'rainbow top'}]}
        diagnosis = {'candidate_ids': [1], 'geometry': {}, 'visual_needs': [
            {'id': 1, 'category': 'table', 'role': 'reference', 'attributes': ['rainbow top']}]}
        action = evidence_request({'action': 'evidence', 'dimension': 'appearance',
            'focus_ids': [1], 'reason': 'inspect both roles'}, [1], objects)
        for reference_result in ('uncertain', 'omitted_attribute', 'error'):
            with self.subTest(reference_result=reference_result):
                payloads, calls = [], []
                def fake_chat(messages, **kwargs):
                    payload = json.loads(messages[1]['content'][1]['text'])
                    payloads.append(payload)
                    requirements = payload['review_requirements']
                    self.assertEqual(len(requirements), 1)
                    self.assertNotIn('attributes', payload)
                    self.assertNotIn('required_attributes', payload)
                    need = requirements[0]
                    if need['role'] == 'reference' and reference_result == 'error':
                        raise ValueError('reference call failed')
                    checks = [] if need['role'] == 'reference' and reference_result == 'omitted_attribute' else [
                        {'attribute': name, 'verdict': 'uncertain', 'visibility': 'low_detail',
                         'reason': 'not enough detail'} for name in need['attributes']]
                    return {'_content': json.dumps({'checks': [{
                        'id': 1, 'category': 'table', 'role': need['role'], 'verdict': 'yes',
                        'visibility': 'visible', 'reason': 'table visible', 'attribute_checks': checks}]}),
                        '_finish_reason': 'stop', '_baseline': {'latency_s': .01}, 'usage': {}}
                with patch('eviground.grounding.active_diagnostic.render_action', return_value={
                        'tool': 'render', 'image': 'mock.png', 'focus_ids': [1]}), patch(
                        'eviground.grounding.active_diagnostic.image_part', return_value={'type': 'text', 'text': 'image'}), patch(
                        'eviground.grounding.active_diagnostic.chat', side_effect=fake_chat):
                    result = execute_evidence(action, 'brown table next to table with rainbow top',
                        query, diagnosis, objects, objects, None, None, None, Path('/tmp'), 1, calls)
                self.assertEqual([payload['review_requirements'][0]['role'] for payload in payloads],
                                 ['target', 'reference'])
                self.assertEqual(len(calls), 2)
                self.assertEqual(result['coverage']['checked_ids'], [1])
                self.assertEqual(result['coverage']['unknown_ids'], [1])
                self.assertEqual(result['coverage']['reference_checked_ids'],
                                 [1] if reference_result == 'uncertain' else [])
                self.assertEqual(result['coverage']['reference_uninspected_ids'],
                                 [] if reference_result == 'uncertain' else [1])
                self.assertEqual(result['candidate_evidence'][0]['status'], 'unknown')
                self.assertEqual(result['operation_status'], 'partial')

    def test_same_category_target_and_reference_attributes_stay_role_bound(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 2, 0),
                   obj(3, 'cabinet', 4, 0)]
        query = {**parsed(), 'Target': ['cabinet'], 'Anchor': ['cabinet'],
                 'Attributes': [
                     {'category': 'cabinet', 'role': 'target', 'phrase': 'red largest'},
                     {'category': 'cabinet', 'role': 'reference', 'phrase': 'blue tallest'}]}
        diagnosis = {'candidate_ids': [1, 2], 'geometry': {}, 'visual_needs': [
            {'id': 3, 'category': 'cabinet', 'role': 'reference', 'attributes': ['blue']}]}
        requirements = _appearance_review_requirements(query, diagnosis, [1, 3], objects)
        self.assertEqual(requirements, [
            {'id': 1, 'category': 'cabinet', 'role': 'target', 'attributes': ['red'],
             'source_attributes': ['red largest'],
             'attribute_contexts': [{'attribute': 'red', 'source_quotes': ['red largest'],
                                     'qualified_fragment': False}]},
            {'id': 3, 'category': 'cabinet', 'role': 'reference', 'attributes': ['blue'],
             'source_attributes': ['blue'],
             'attribute_contexts': [{'attribute': 'blue', 'source_quotes': ['blue'],
                                     'qualified_fragment': False}]}])
        self.assertEqual(visual_attributes(query), ['red'])
        measurements = [
            {'id': 1, 'status': 'known', 'volume_m3': 2., 'height_m': 1.},
            {'id': 2, 'status': 'known', 'volume_m3': 1., 'height_m': 2.}]
        evidence, constraints, _ = size_candidate_evidence(
            query, objects[:2], measurements, comparison_scope_bound=True)
        self.assertEqual([item['attribute'] for item in constraints], ['largest'])
        self.assertEqual([item['status'] for item in evidence], ['pass', 'fail'])

    def test_independent_color_contradiction_remains_bound_across_visual_entry_points(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 2, 0)]
        objects[1]['bbox_3d'][3:5] = [.8, .8]
        phrase = 'white stripes on top'
        sources = ['white largest', phrase]
        query = {**parsed(), 'Target': ['cabinet'], 'Attributes': [
            {'category': 'cabinet', 'role': 'target', 'phrase': value} for value in sources]}
        requirements = _appearance_review_requirements(
            query, {'candidate_ids': [1, 2], 'visual_needs': []}, [1, 2], objects)
        white = next(item for item in requirements[0]['attribute_contexts']
                     if item['attribute'] == 'white')
        self.assertEqual(white['source_quotes'], sources)
        self.assertFalse(white['qualified_fragment'])
        expected = requirements[0]['attributes']
        def check(oid, names):
            return {'id': oid, 'verdict': 'yes', 'visibility': 'visible',
                    'reason': 'cabinet visible', 'attribute_checks': [
                        {'attribute': name, 'verdict': verdict, 'visibility': 'visible',
                         'reason': 'visible attribute evidence'} for name, verdict in names]}
        views = [check(1, [('white', 'no')]),
                 check(1, [(name, 'yes') for name in expected if name != 'white']),
                 check(2, [(name, 'yes') for name in expected])]
        checks = [_check_payload({'checks': [view]}, [1, 2], objects, query, requirements)[0]
                  for view in views]
        self.assertEqual(checks[0]['verdict'], 'no')
        pool, bound, unknown = _target_size_scope(
            query, {'size_comparison_pool_ids': [1, 2], 'checks': checks}, objects)
        self.assertEqual((pool, bound, unknown), ([1, 2], True, []))
        result = diagnose(query, objects, objects, [0, 0, .5], checks, trusted_categories=True)
        self.assertEqual(result['status'], 'resolved')
        self.assertEqual(result['candidate_ids'], [2])

    def test_full_category_comparative_uses_geometry_extreme_and_keeps_source_phrase(self):
        query = {**parsed(), 'Target': ['cabinet'],
                 'source_query': 'The smaller of the cabinets.', 'Attributes': [
            {'category': 'cabinet', 'role': 'target', 'phrase': 'smaller',
             'comparison_scope': 'full_category',
             'scope_phrase': 'The smaller of the cabinets.'}]}
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 2, 0)]
        evidence, constraints, _ = size_candidate_evidence(query, objects, [
            {'id': 1, 'status': 'known', 'measurement_reliability': 'usable',
             'volume_m3': 1., 'footprint_m2': 1.,
             'oriented_volume_m3': 1., 'oriented_footprint_m2': 1.,
             'metric_reliability': {'oriented_volume_m3': {'status': 'usable'},
                                    'oriented_footprint_m2': {'status': 'usable'}}},
            {'id': 2, 'status': 'known', 'measurement_reliability': 'usable',
             'volume_m3': 2., 'footprint_m2': 2.,
             'oriented_volume_m3': 2., 'oriented_footprint_m2': 2.,
             'metric_reliability': {'oriented_volume_m3': {'status': 'usable'},
                                    'oriented_footprint_m2': {'status': 'usable'}}}])
        self.assertEqual([item['status'] for item in evidence], ['pass', 'fail'])
        self.assertEqual(constraints[0]['attribute'], 'smaller')
        self.assertEqual(constraints[0]['form'], 'comparative')
        self.assertEqual(constraints[0]['original_quote'], 'smaller')
        self.assertEqual(visual_attributes(query), [])
        self.assertEqual(_target_size_scope(query, {'size_comparison_pool_ids': [1, 2]}, objects),
                         ([1, 2], True, []))
        self.assertEqual(query['Attributes'][0]['phrase'], 'smaller')
        self.assertEqual(query['Attributes'][0]['scope_phrase'], 'The smaller of the cabinets.')

    def test_reference_checks_cannot_bind_target_appearance_size_scope(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 2, 0)]
        query = {**parsed(), 'Target': ['cabinet'], 'Anchor': ['cabinet'],
                 'source_query': 'the largest of the red cabinets next to the red cabinet',
                 'Attributes': [
                     {'category': 'cabinet', 'role': 'target', 'phrase': 'red largest'},
                     {'category': 'cabinet', 'role': 'reference', 'phrase': 'red'}]}
        for role in ('reference', None, 'target'):
            with self.subTest(role=role):
                checks = [
                    {'id': oid, 'category': 'cabinet', 'verdict': 'yes',
                     'visibility': 'visible', 'reason': 'cabinet visible',
                     'attribute_checks': [{'attribute': 'red', 'verdict': 'yes',
                                           'visibility': 'visible', 'reason': 'red finish'}],
                     **({'role': role} if role else {})} for oid in [1, 2]]
                pool, scope = size_condition_scope(query, objects, 'cabinet',
                    size_conditions(query, 'cabinet')[0], checks)
                self.assertEqual([item['bbox_id'] for item in pool], [1, 2] if role == 'target' else [])
                self.assertEqual(scope['status'] == 'bound', role == 'target')
                self.assertEqual(scope['unknown_ids'], [] if role == 'target' else [1, 2])

    def test_multiple_size_extremes_require_all_constraints(self):
        query = parsed()
        query['Target'] = ['cabinet']
        query['Attributes'] = [{'category': 'cabinet', 'phrase': 'smallest tallest'}]
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 2, 0)]
        measurements = [
            {'id': 1, 'status': 'known', 'volume_m3': 1., 'height_m': 1.},
            {'id': 2, 'status': 'known', 'volume_m3': 2., 'height_m': 2.},
        ]
        evidence, _, decision = size_candidate_evidence(query, objects, measurements)
        self.assertEqual([row['status'] for row in evidence], ['fail', 'fail'])
        self.assertEqual(decision['candidate_relative_ids'], [])

    def test_size_superlative_can_filter_from_point_cloud_measurements(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 2, 0)]
        measurements = [
            {'id': 1, 'status': 'known', 'volume_m3': 2., 'height_m': 1.,
             'horizontal_aspect_ratio': 1.},
            {'id': 2, 'status': 'known', 'volume_m3': 1., 'height_m': 1.,
             'horizontal_aspect_ratio': 1.},
        ]
        query = parsed()
        query['Target'] = ['cabinet']
        query['Attributes'] = [{'category': 'cabinet', 'phrase': 'smallest'}]
        evidence, constraints, _ = size_candidate_evidence(query, objects, measurements)
        self.assertEqual(constraints[0]['selected_ids'], [2])
        self.assertEqual([row['status'] for row in evidence], ['fail', 'pass'])

    def test_size_extreme_keeps_frozen_pool_after_candidate_shrink(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 2, 0),
                   obj(3, 'cabinet', 4, 0)]
        measurements = [
            {'id': 1, 'status': 'known', 'measurement_reliability': 'usable',
             'volume_m3': 2., 'height_m': 1., 'horizontal_aspect_ratio': 1.},
            {'id': 2, 'status': 'known', 'measurement_reliability': 'usable',
             'volume_m3': 1., 'height_m': 1., 'horizontal_aspect_ratio': 1.},
            {'id': 3, 'status': 'known', 'measurement_reliability': 'usable',
             'volume_m3': 4., 'height_m': 1., 'horizontal_aspect_ratio': 1.},
        ]
        query = parsed()
        query['Target'] = ['cabinet']
        query['Attributes'] = [{'category': 'cabinet', 'phrase': 'largest'}]
        evidence, constraints, decision = size_candidate_evidence(
            query, objects[:2], measurements, comparison_pool_ids=[1, 2, 3],
            comparison_scope_bound=True)
        self.assertEqual(decision['comparison_pool_ids'], [1, 2, 3])
        self.assertTrue(decision['comparison_pool_frozen'])
        self.assertEqual(constraints[0]['selected_ids'], [3])
        self.assertEqual([row['status'] for row in evidence], ['fail', 'fail'])

    def test_size_does_not_rank_unbound_appearance_subset(self):
        objects = [obj(1, 'chair', 0, 0), obj(2, 'chair', 2, 0)]
        measurements = [
            {'id': 1, 'status': 'known', 'measurement_reliability': 'usable',
             'volume_m3': 2., 'height_m': 1., 'horizontal_aspect_ratio': 1.},
            {'id': 2, 'status': 'known', 'measurement_reliability': 'usable',
             'volume_m3': 1., 'height_m': 1., 'horizontal_aspect_ratio': 1.},
        ]
        query = parsed()
        query['Target'] = ['chair']
        query['Attributes'] = [{'category': 'chair', 'phrase': 'red largest'}]
        query['source_query'] = 'the largest of the red chairs'
        evidence, constraints, decision = size_candidate_evidence(query, objects, measurements,
                                                                  objects=objects)
        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0]['scope_status'], 'unknown')
        self.assertIn('not fully bound', decision['scope_missing'][0])
        self.assertEqual(decision['attributes'][0]['selected_ids'], [])
        self.assertTrue(all(row['status'] == 'unknown' for row in evidence))

    def test_mixed_appearance_and_size_phrase_binds_independent_attribute(self):
        objects = [obj(1, 'chair', 0, 0), obj(2, 'chair', 2, 0)]
        query = parsed()
        query['Target'] = ['chair']
        query['Attributes'] = [{'category': 'chair', 'phrase': 'red largest'}]
        diagnosis = {
            'size_comparison_pool_ids': [1, 2],
            'checks': [
                {'id': 1, 'category': 'chair', 'verdict': 'yes',
                 'visibility': 'visible', 'reason': 'chair is visible',
                 'attribute_checks': [{'attribute': 'red', 'verdict': 'yes',
                                       'visibility': 'visible', 'reason': 'red finish'}]},
                {'id': 2, 'category': 'chair', 'verdict': 'no',
                 'visibility': 'visible', 'reason': 'chair is blue',
                 'attribute_checks': [{'attribute': 'red', 'verdict': 'no',
                                       'visibility': 'visible', 'reason': 'blue finish'}]},
            ],
        }
        pool_ids, bound, unknown = _target_size_scope(query, diagnosis, objects)
        self.assertEqual(visual_attributes(query), ['red'])
        self.assertEqual(pool_ids, [1, 2])
        self.assertTrue(bound)
        self.assertEqual(unknown, [])

    def test_size_extreme_tie_stays_unresolved(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 2, 0)]
        measurements = [
            {'id': 1, 'status': 'known', 'measurement_reliability': 'usable',
             'volume_m3': 2., 'height_m': 1., 'horizontal_aspect_ratio': 1.},
            {'id': 2, 'status': 'known', 'measurement_reliability': 'usable',
             'volume_m3': 2., 'height_m': 1., 'horizontal_aspect_ratio': 1.},
        ]
        query = parsed()
        query['Target'] = ['cabinet']
        query['Attributes'] = [{'category': 'cabinet', 'phrase': 'largest'}]
        evidence, constraints, decision = size_candidate_evidence(
            query, objects, measurements, comparison_scope_bound=True)
        self.assertEqual(constraints[0]['selected_ids'], [])
        self.assertEqual(constraints[0]['tie_ids'], [1, 2])
        self.assertEqual(decision['attributes'][0]['status'], 'unknown')
        self.assertEqual(decision['candidate_relative_ids'], [])
        self.assertEqual([row['status'] for row in evidence], ['unknown', 'unknown'])

    def test_size_unreliable_measurement_stays_unknown(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 2, 0)]
        measurements = [
            {'id': 1, 'status': 'known', 'measurement_reliability': 'unreliable',
             'volume_m3': 10., 'height_m': 2., 'horizontal_aspect_ratio': 1.},
            {'id': 2, 'status': 'known', 'measurement_reliability': 'usable',
             'volume_m3': 2., 'height_m': 1., 'horizontal_aspect_ratio': 1.},
        ]
        query = parsed()
        query['Target'] = ['cabinet']
        query['Attributes'] = [{'category': 'cabinet', 'phrase': 'largest'}]
        evidence, constraints, decision = size_candidate_evidence(
            query, objects, measurements, comparison_scope_bound=True)
        self.assertEqual(constraints[0]['selected_ids'], [])
        self.assertEqual(decision['unknown_measurement_ids'], [1])
        self.assertEqual([row['status'] for row in evidence], ['unknown', 'unknown'])

    def test_unknown_comparison_member_cannot_certify_known_size_winner(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 2, 0)]
        measurements = [{'id': 1, 'status': 'known', 'measurement_reliability': 'usable',
                         'volume_m3': 2., 'height_m': 1., 'horizontal_aspect_ratio': 1.}]
        query = parsed()
        query['Target'] = ['cabinet']
        query['Attributes'] = [{'category': 'cabinet', 'phrase': 'largest'}]
        evidence, _, decision = size_candidate_evidence(
            query, objects, measurements, comparison_pool_ids=[1, 2],
            comparison_scope_bound=True)
        self.assertEqual([row['status'] for row in evidence], ['unknown', 'unknown'])
        self.assertEqual(decision['unknown_measurement_ids'], [2])
        self.assertEqual(decision['candidate_relative_ids'], [])

    def test_size_extreme_needs_ordinary_shape_subset_evidence(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 2, 0)]
        query = parsed()
        query['Target'] = ['cabinet']
        query['Attributes'] = [{'category': 'cabinet', 'phrase': 'largest skinny'}]
        query['source_query'] = 'the largest of the skinny cabinets'
        diagnosis = {'size_comparison_pool_ids': [1, 2], 'checks': []}
        pool, scope = size_condition_scope(query, objects, 'cabinet',
            size_conditions(query, 'cabinet')[0], diagnosis['checks'])
        self.assertEqual(pool, [])
        self.assertEqual(scope['status'], 'unknown')
        self.assertEqual(scope['unknown_ids'], [1, 2])

    def test_partial_color_evidence_cannot_bind_size_subset(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 2, 0)]
        query = parsed()
        query['Target'] = ['cabinet']
        query['Attributes'] = [{'category': 'cabinet', 'phrase': 'red largest'}]
        query['source_query'] = 'the largest of the red cabinets'
        diagnosis = {'size_comparison_pool_ids': [1, 2], 'checks': [
            {'id': 1, 'category': 'cabinet', 'verdict': 'yes', 'visibility': 'visible',
             'reason': 'visible cabinet', 'attribute_checks': [
                 {'attribute': 'red', 'verdict': 'yes', 'visibility': 'visible',
                  'reason': 'red surface'}]}]}
        pool, scope = size_condition_scope(query, objects, 'cabinet',
            size_conditions(query, 'cabinet')[0], diagnosis['checks'])
        self.assertEqual(pool, [])
        self.assertEqual(scope['status'], 'unknown')
        self.assertEqual(scope['unknown_ids'], [2])

    def test_incomplete_geometry_cannot_resolve_ordinary_size_adjectives(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 2, 0)]
        xyz = np.asarray([
            [0., 0., 0.], [1., .5, 2.], [2., 0., 0.], [2.4, .3, 1.],
        ])
        ids = np.asarray([1, 1, 2, 2])
        for adjective in ('large', 'tall', 'skinny', 'narrow'):
            with self.subTest(adjective=adjective), tempfile.TemporaryDirectory() as directory:
                query = parsed()
                query['Target'] = ['cabinet']
                query['Attributes'] = [{'category': 'cabinet', 'phrase': adjective}]
                responses = iter([
                    {'action': 'evidence', 'dimension': 'size', 'reason': 'compare size'},
                    {'action': 'finish', 'predicted_id': 1, 'reason': 'geometry hint'},
                    {'action': 'stop', 'reason': 'visual confirmation still missing'},
                ])

                def fake_chat(messages, **kwargs):
                    return {'_content': json.dumps(next(responses)),
                            '_baseline': {'model': 'mock', 'latency_s': .01},
                            '_finish_reason': 'stop', 'usage': {}}

                with patch('eviground.grounding.active_diagnostic.chat', side_effect=fake_chat):
                    trace, calls = {}, []
                    result = active_ground(
                        f'{adjective} cabinet', query, objects, objects, objects,
                        xyz, np.zeros_like(xyz, dtype=np.uint8), ids,
                        Path(directory), calls, trace, 2, True)
                self.assertIsNone(result['predicted_id'])
                self.assertFalse(result['accepted'])
                self.assertEqual(result['diagnosis']['candidate_ids'], [1, 2])
                self.assertTrue(result['diagnosis']['missing'])

    def test_comparative_size_uses_explicit_reliable_candidate_comparison(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 2, 0)]
        measurements = [
            {'id': 1, 'status': 'known', 'measurement_reliability': 'usable',
             'volume_m3': 1., 'footprint_m2': 1., 'height_m': 1.,
             'horizontal_aspect_ratio': 1.,
             'oriented_volume_m3': 1., 'oriented_footprint_m2': 1.,
             'metric_reliability': {'oriented_volume_m3': {'status': 'usable'},
                                    'oriented_footprint_m2': {'status': 'usable'}}},
            {'id': 2, 'status': 'known', 'measurement_reliability': 'usable',
             'volume_m3': 2., 'footprint_m2': 2., 'height_m': 1.,
             'horizontal_aspect_ratio': 1.,
             'oriented_volume_m3': 2., 'oriented_footprint_m2': 2.,
             'metric_reliability': {'oriented_volume_m3': {'status': 'usable'},
                                    'oriented_footprint_m2': {'status': 'usable'}}},
        ]
        query = parsed()
        query['Target'] = ['cabinet']
        query['Attributes'] = [{'category': 'cabinet', 'phrase': 'larger'}]
        evidence, constraints, decision = size_candidate_evidence(query, objects, measurements)
        self.assertEqual(constraints[0]['attribute'], 'larger')
        self.assertEqual(constraints[0]['interpretation'], 'pair_comparison')
        self.assertEqual(decision['candidate_relative_ids'], [2])
        self.assertEqual([row['status'] for row in evidence], ['fail', 'pass'])

    def test_unique_relative_size_hint_can_be_verified_on_finish(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 2, 0)]
        query = {
            'Target': ['cabinet'], 'Anchor': [], 'Relations': [],
            'Observer': {'origin': {}, 'facing': {}}, 'Viewpoint': '',
            'Attributes': [{'category': 'cabinet', 'phrase': 'skinny'}],
            'Unhandled': [], 'diagnostic_missing': [],
        }
        xyz = np.asarray([
            [0., 0., 0.], [4., 1., 1.], [2., .5, .5],
            [10., 0., 0.], [11., 1., 1.], [10.5, .5, .5],
        ])
        ids = np.asarray([1, 1, 1, 2, 2, 2])
        responses = iter([
            {'action': 'evidence', 'dimension': 'size', 'reason': 'compare cabinet widths'},
            {'action': 'finish', 'predicted_id': 1, 'reason': 'cabinet 1 is uniquely skinny'},
            {'action': 'stop', 'reason': 'skinny needs appearance confirmation'},
        ])
        def fake_chat(messages, **kwargs):
            return {'_content': json.dumps(next(responses)),
                    '_baseline': {'model': 'mock', 'latency_s': .01},
                    '_finish_reason': 'stop', 'usage': {}}
        with tempfile.TemporaryDirectory() as directory, patch(
                'eviground.grounding.active_diagnostic.chat', side_effect=fake_chat):
            trace, calls = {}, []
            result = active_ground('skinny cabinet', query, objects, objects, objects,
                                   xyz, np.zeros_like(xyz, dtype=np.uint8), ids,
                                   Path(directory), calls, trace, 2, True)
        self.assertIsNone(result['predicted_id'])
        self.assertFalse(result['accepted'])
        self.assertEqual(result['diagnosis']['candidate_ids'], [1, 2])
        self.assertTrue(result['diagnosis']['missing'])

    def test_size_superlative_parser_outputs_and_finish(self):
        for word, selected, accepted in [('smallest', 2, True), ('largest', 1, True),
                                         ('biggest', 1, True),
                                         ('shortest', 2, True), ('tallest', 1, True),
                                          ('small red', 2, False),
                                          ('unknown', 1, False), ('tied', 1, False)]:
            with self.subTest(word=word):
                text = f'Choose the {word if word not in ("unknown", "tied") else "largest"} cabinet.'
                objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 2, 0)]
                query = {
                    'Target': ['cabinet'], 'Anchor': [], 'Relations': [],
                    'Observer': {'origin': {}, 'facing': {}}, 'Viewpoint': '',
                    'Attributes': [{'category': 'cabinet', 'phrase': word}],
                    'Unhandled': [text] if word != 'tallest' else [], 'diagnostic_missing': [],
                }
                if word in ('smallest', 'shortest'):
                    query['Attributes'] = []
                query = validate_structure(text, query)
                xyz = np.asarray([
                    [0., 0., 0.], [4., 1., 3.], [2., .5, .5],
                    [10., 0., 0.], [11., 1., 1.], [10.5, .5, .5],
                ])
                ids = np.asarray([1, 1, 1, 2, 2, 2])
                if word == 'unknown':
                    ids[3:] = -1
                elif word == 'tied':
                    xyz[4] = [14., 1., 3.]
                responses = iter([
                    {'action': 'evidence', 'dimension': 'size', 'reason': 'compare cabinet widths'},
                    {'action': 'finish', 'predicted_id': selected, 'reason': 'cabinet 1 is uniquely skinny'},
                    {'action': 'stop', 'reason': 'unresolved conditions'},
                ])
                def fake_chat(messages, **kwargs):
                    return {'_content': json.dumps(next(responses)),
                            '_baseline': {'model': 'mock', 'latency_s': .01},
                            '_finish_reason': 'stop', 'usage': {}}
                with tempfile.TemporaryDirectory() as directory, patch(
                        'eviground.grounding.active_diagnostic.chat', side_effect=fake_chat):
                    trace, calls = {}, []
                    result = active_ground(text, query, objects, objects, objects,
                                           xyz, np.zeros_like(xyz, dtype=np.uint8), ids,
                                           Path(directory), calls, trace, 2, True)
                self.assertEqual(result['accepted'], accepted)

                if accepted:
                    self.assertEqual(result['predicted_id'], selected)
                    self.assertEqual(result['diagnosis']['missing'], [])
                    self.assertEqual(result['diagnosis']['visual_needs'], [])

    def run_size_position_actions(self, actions, largest_above=False, limit=3,
                                  unknown_measurement=False):
        query = 'Choose the largest cabinet above the table.'
        objects = [
            {'bbox_id': 1, 'target': 'cabinet',
             'bbox_3d': [0 if largest_above else 3, 0, 2.5, 2, 2, 1]},
            obj(2, 'cabinet', 3 if largest_above else 0, 0, z=2.5),
            obj(3, 'table', 0, 0),
        ]
        parsed_query = validate_structure(query, {
            **parsed(), 'Target': ['cabinet'], 'Anchor': ['table'],
            'Attributes': [{'category': 'cabinet', 'phrase': 'largest'}],
            'Relations': [{'target_category': 'cabinet', 'anchor_category': 'table',
                           'relation': 'above', 'scope': 'relative_anchor',
                           'phrase': 'above the table'}],
        })
        points, instance_ids = [], []
        for item in objects:
            center = np.asarray(item['bbox_3d'][:3], dtype=float)
            half = np.asarray(item['bbox_3d'][3:], dtype=float) / 2
            points.extend([center - half, center + half, center])
            instance_ids.extend([item['bbox_id']] * 3)
        xyz, ids = np.asarray(points), np.asarray(instance_ids)
        if unknown_measurement:
            ids[ids == 1] = -1
        responses = iter(actions)

        def fake_chat(messages, **kwargs):
            return {'_content': json.dumps(next(responses, {
                        'action': 'stop', 'reason': 'size has still not been checked'})),
                    '_baseline': {'model': 'mock', 'latency_s': .01},
                    '_finish_reason': 'stop', 'usage': {}}

        with tempfile.TemporaryDirectory() as directory, patch(
                'eviground.grounding.active_diagnostic.chat', side_effect=fake_chat), patch(
                'eviground.grounding.active_diagnostic.render_action', return_value={
                    'tool': 'render', 'image': 'mock.png',
                    'focus_ids': [1 if largest_above else 2]}), patch(
                'eviground.grounding.active_diagnostic.image_part', return_value={'type': 'text', 'text': 'image'}):
            return active_ground(query, parsed_query, objects, objects, objects[:2],
                                 xyz, np.zeros_like(xyz, dtype=np.uint8), ids,
                                 Path(directory), [], {}, limit, True)

    def test_finish_requires_size_even_when_position_has_one_survivor(self):
        for limit in (0, 2):
            with self.subTest(limit=limit):
                result = self.run_size_position_actions([
                    {'action': 'finish', 'predicted_id': 2, 'reason': 'only cabinet above table'},
                    {'action': 'stop', 'reason': 'size has not been checked'},
                ], limit=limit)
                self.assertFalse(result['accepted'])
                self.assertEqual(result['decision_status'], 'missing')
                self.assertEqual(result['diagnosis']['candidate_ids'], [2])
                self.assertEqual(result['diagnosis']['pending_size_conditions'], [
                    {'attribute': 'largest', 'phrase': 'largest', 'role': 'target'}])

    def test_size_and_position_conflict_is_not_accepted(self):
        result = self.run_size_position_actions([
            {'action': 'evidence', 'dimension': 'size', 'reason': 'check largest over frozen pool'},
            {'action': 'finish', 'predicted_id': 2, 'reason': 'fallback cabinet'},
            {'action': 'stop', 'reason': 'no cabinet meets both conditions'},
        ])
        self.assertFalse(result['accepted'])
        self.assertEqual(result['decision_status'], 'conflict')
        self.assertEqual(result['diagnosis']['candidate_ids'], [])

    def test_size_and_position_resolve_after_complete_size_evidence(self):
        result = self.run_size_position_actions([
            {'action': 'evidence', 'dimension': 'size', 'reason': 'check largest over frozen pool'},
            {'action': 'finish', 'predicted_id': 1, 'reason': 'all conditions checked'},
        ], largest_above=True)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['predicted_id'], 1)
        self.assertEqual(result['diagnosis']['pending_size_conditions'], [])

    def test_unknown_size_evidence_cannot_clear_pending_condition(self):
        result = self.run_size_position_actions([
            {'action': 'evidence', 'dimension': 'size', 'reason': 'check largest over frozen pool'},
            {'action': 'finish', 'predicted_id': 1, 'reason': 'only cabinet above table'},
            {'action': 'stop', 'reason': 'largest measurement missing'},
        ], largest_above=True, unknown_measurement=True)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['decision_status'], 'missing')
        self.assertTrue(result['diagnosis']['pending_size_conditions'])

    def test_appearance_refresh_keeps_size_pending_or_replays_completed_size(self):
        for compute_size in (False, True):
            with self.subTest(compute_size=compute_size):
                actions = ([{'action': 'evidence', 'dimension': 'size',
                             'reason': 'check largest over frozen pool'}] if compute_size else [])
                actions.extend([
                    {'action': 'evidence', 'dimension': 'appearance', 'focus_ids': [1],
                     'reason': 'verify cabinet category'},
                    {'checks': [{'id': 1, 'category': 'cabinet', 'role': 'target',
                                 'category_verdict': 'yes', 'verdict': 'yes',
                                 'reason': 'cabinet visible', 'visibility': 'visible'}]},
                    {'action': 'finish', 'predicted_id': 1, 'reason': 'all conditions checked'},
                    {'action': 'stop', 'reason': 'size still pending'},
                ])
                result = self.run_size_position_actions(actions, largest_above=True)
                self.assertEqual(result['accepted'], compute_size)
                self.assertEqual(bool(result['diagnosis']['pending_size_conditions']), not compute_size)

    def test_finish_cannot_mix_opposed_viewpoint_hypotheses_with_size(self):
        objects = [obj(1, 'window', -1, 1),
                   {'bbox_id': 2, 'target': 'window',
                    'bbox_3d': [1, 1, 1., .8, .8, 2.]},
                   obj(3, 'tv', 0, 3), obj(4, 'tv', 0, -3)]
        query = 'Standing in the middle of the room facing the tv, the largest window on the right.'
        parsed_query = validate_structure(query, {
            'Target': ['window'], 'Anchor': ['tv'],
            'Relations': [{'target_category': 'window', 'anchor_category': '',
                           'relation': 'right', 'scope': 'side',
                           'phrase': 'on the right'}],
            'Observer': {
                'origin': {'kind': 'room_center', 'category': '',
                           'phrase': 'middle of the room'},
                'facing': {'category': 'tv', 'group': False,
                           'phrase': 'facing the tv'},
            },
            'Attributes': [{'category': 'window', 'phrase': 'largest'}],
            'Unhandled': [], 'Viewpoint': 'middle of the room facing the tv',
        })
        # Complete thin vertical faces support the window-area predicate.
        first = [[x, y, z] for x in (-1.2, -.8) for y in (.995, 1.005)
                 for z in (0., 1.)]
        second = [[x, y, z] for x in (.6, 1.4) for y in (.995, 1.005)
                  for z in (0., 2.)]
        xyz = np.asarray(first + second + [[0., 3., .5], [0., -3., .5]])
        ids = np.asarray([1] * 8 + [2] * 8 + [3, 4])
        actions = iter([
            {'action': 'evidence', 'dimension': 'size', 'reason': 'compare largest windows'},
            {'action': 'finish', 'predicted_id': 2, 'reason': 'window 2 is largest'},
            {'action': 'stop', 'reason': 'opposed facing hypotheses remain'},
        ])

        def fake_chat(messages, **kwargs):
            return {'_content': json.dumps(next(actions)),
                    '_baseline': {'model': 'mock', 'latency_s': .01},
                    '_finish_reason': 'stop', 'usage': {}}

        with tempfile.TemporaryDirectory() as directory, patch(
                'eviground.grounding.active_diagnostic.chat', side_effect=fake_chat):
            result = active_ground(
                query, parsed_query, objects, objects, objects[:2], xyz,
                np.zeros_like(xyz, dtype=np.uint8), ids, Path(directory), [], {}, 2, True)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['predicted_id'], 2)
        self.assertEqual(result['prediction_source'], 'singleton_fallback')
        self.assertEqual(result['decision_status'], 'missing')
        self.assertEqual(result['active_trace']['final_candidate_ids'], [2])
        self.assertEqual([h['candidate_ids'] for h in result['diagnosis']['hypotheses']],
                         [[2], []])
        self.assertEqual(result['diagnosis']['stable_candidate_ids'], [])

    def test_computed_filter_preserves_empty_legal_hypothesis(self):
        diagnosis = {
            'candidate_ids': [1, 2], 'stable_candidate_ids': [], 'missing': [],
            'hypotheses': [{'candidate_ids': [1]}, {'candidate_ids': [2]}],
        }
        apply_computed_evidence(diagnosis, [
            {'candidate_id': 1, 'status': 'fail',
             'source_quote': 'the larger window', 'target_role': 'target',
             'role_binding_status': 'bound', 'comparison_scope_ids': [1, 2],
             'comparison_scope_status': 'bound',
             'measurement_source': 'synthetic reliable window face measurements',
             'explicit_fail_criterion': 'candidate 1 face area is below the larger-window threshold'},
            {'candidate_id': 2, 'status': 'pass'},
        ], 'size')
        self.assertEqual(diagnosis['candidate_ids'], [2])
        self.assertEqual(diagnosis['stable_candidate_ids'], [])
        self.assertEqual(diagnosis['status'], 'ambiguous')
        self.assertEqual([item['candidate_ids'] for item in diagnosis['hypotheses']],
                         [[], [2]])
        audit = diagnosis['active_computed_evidence'][0]['candidate_removal_audit'][0]
        self.assertTrue(audit['removal_allowed'])
        self.assertEqual(audit['comparison_scope_ids'], [1, 2])

    def test_unknown_computed_update_cannot_drop_gate_preserved_candidate(self):
        diagnosis = {
            'candidate_ids': [1, 2], 'stable_candidate_ids': [1], 'missing': [],
            # Candidate 2 was retained by the initial removal gate because its
            # relation hypothesis was incomplete.  The next operation has no
            # new fail proof and cannot erase it by recomputing this union.
            'hypotheses': [{'candidate_ids': [1]}],
        }
        update = apply_computed_evidence(diagnosis, [
            {'candidate_id': 1, 'status': 'unknown'},
            {'candidate_id': 2, 'status': 'unknown'},
        ], 'position')
        self.assertEqual(update['removed_ids'], [])
        self.assertEqual(update['after'], [1, 2])
        self.assertEqual(diagnosis['candidate_ids'], [1, 2])
        audit = diagnosis['active_computed_evidence'][0]['candidate_removal_audit']
        self.assertTrue(all(not item['removal_allowed'] for item in audit))

    def test_unreviewed_reference_cannot_be_removed_by_position_or_distance_refresh(self):
        objects = [obj(1, 'window', 0, 0, 2), obj(2, 'window', 4, 0, -.5),
                   obj(7, 'table', 0, 0)]
        for dimension, relation, quote in (
                ('position', 'above', 'above the red table'),
                ('distance', 'nearest', 'nearest the red table')):
            with self.subTest(dimension=dimension):
                query = 'the window ' + quote
                parsed_query = validate_structure(query, {
                    **parsed(), 'Anchor': ['table'],
                    'Relations': [{'target_category': 'window', 'anchor_category': 'table',
                                   'relation': relation, 'phrase': quote}],
                    'Attributes': [{'category': 'table', 'role': 'reference', 'phrase': 'red'}],
                })
                diagnosis = diagnose(parsed_query, objects, objects[:2], [0, 0, 0],
                                     trusted_categories=True)
                self.assertEqual(diagnosis['candidate_ids'], [1, 2])
                self.assertTrue(any(need['role'] == 'reference'
                                    for need in diagnosis['visual_needs']))
                observation = compute_observation(dimension, query, parsed_query, diagnosis,
                                                  objects[:2], objects)
                evidence = observation['candidate_evidence']
                self.assertTrue(all(item['status'] == 'unknown' for item in evidence))
                update = apply_computed_evidence(diagnosis, evidence, dimension)
                self.assertEqual(update['removed_ids'], [])
                self.assertEqual(update['after'], [1, 2])

    def test_unbound_between_references_preserve_candidates_after_position(self):
        objects = [obj(8, 'trash can', 0, 0), obj(9, 'trash can', 1, 0),
                   obj(18, 'trash can', 2, 0), obj(4, 'copier', 0, 1),
                   obj(26, 'table', 2, 1)]
        query = 'The trashcan in between printer and wooden desk'
        parsed_query = validate_structure(query, {
            'Target': ['trashcan'], 'Anchor': ['printer', 'wooden desk'],
            'Relations': [{'target_category': 'trashcan', 'anchor_category': 'printer',
                           'anchor_categories': ['printer', 'wooden desk'],
                           'relation': 'between',
                           'phrase': 'in between printer and wooden desk'}],
            'Attributes': [{'category': 'wooden desk', 'role': 'reference',
                            'phrase': 'wooden desk'}],
            'Observer': {'origin': {}, 'facing': {}}, 'Viewpoint': '', 'Unhandled': [],
        })
        for compute_position in (False, True):
            with self.subTest(compute_position=compute_position):
                actions = iter(([{'action': 'evidence', 'dimension': 'position',
                                  'reason': 'check between references'}]
                                if compute_position else []) + [
                    {'action': 'stop', 'reason': 'reference categories are unbound'}])

                def fake_chat(messages, **kwargs):
                    return {'_content': json.dumps(next(actions)),
                            '_baseline': {'model': 'mock', 'latency_s': .01},
                            '_finish_reason': 'stop', 'usage': {}}

                with tempfile.TemporaryDirectory() as directory, patch(
                        'eviground.grounding.active_diagnostic.chat', side_effect=fake_chat):
                    result = active_ground(
                        query, parsed_query, objects, objects, objects[:3],
                        np.array([[0., 0., 0.], [2., 1., 1.]]),
                        np.zeros((2, 3), dtype=np.uint8), np.array([8, 9]),
                        Path(directory), [], {}, 2, True)
                self.assertFalse(result['accepted'])
                self.assertEqual(result['decision_status'], 'missing')
                self.assertEqual(result['diagnosis']['status'], 'missing')
                self.assertEqual(result['diagnosis']['candidate_ids'], [8, 9, 18])
                initial=result['active_trace']['initial_diagnosis']
                self.assertEqual(initial['candidate_ids'], [8, 9, 18])
                self.assertEqual(initial['geometry']['between_unknown_ids'], [8, 9, 18])
                for reason in ('missing reference category: desk',
                               'missing reference category: printer'):
                    self.assertIn(reason, result['diagnosis']['missing'])
                if compute_position:
                    observation=result['active_trace']['steps'][0]['observation']
                    self.assertEqual(observation['scope_ids'], [8, 9, 18])
                    self.assertEqual([item['status'] for item in
                                      observation['candidate_evidence']],
                                     ['unknown', 'unknown', 'unknown'])
                self.assertEqual(result['active_trace']['final_evidence_state']['status'], 'missing')

    def test_internal_verification_receives_explicit_role_requirements(self):
        objects = [obj(1, 'window', 0, 0), obj(2, 'tv', 2, 0)]
        query = parsed()
        query['Attributes'] = [{'category': 'window', 'phrase': 'large'},
                               {'category': 'television', 'phrase': 'black'}]
        diagnosis = diagnose(query, objects, objects[:1], [0, 0, .5], trusted_categories=True)
        request = evidence_request({'action': 'evidence', 'dimension': 'appearance',
                                    'reason': 'inspect target'}, [1], objects)
        reply = {'_content': json.dumps({'checks': [
            {'id': 1, 'verdict': 'yes', 'reason': 'visible match'}]}),
            '_finish_reason': 'stop', '_baseline': {'latency_s': .01}, 'usage': {}}
        with patch('eviground.grounding.active_diagnostic.render_action', return_value={
                'tool': 'render', 'image': 'mock.png', 'focus_ids': [1]}), patch(
                'eviground.grounding.active_diagnostic.image_part', return_value={'type': 'text', 'text': 'image'}), patch(
                'eviground.grounding.active_diagnostic.chat', return_value=reply) as api:
            execute_evidence(request, 'large window by black television', query, diagnosis,
                             objects[:1], objects, None, None, None, Path('/tmp'), 1, [])
        payload = json.loads(api.call_args.args[0][1]['content'][1]['text'])
        self.assertEqual(payload['review_requirements'][0]['attributes'], [])
        self.assertEqual(payload['review_requirements'][0]['role'], 'target')
        self.assertNotIn('attributes', payload)

    def test_visual_evidence_preserves_computed_exclusions(self):
        def compute(*args):
            return {'tool': 'compute', 'dimension': 'position', 'candidate_evidence': [
                {'candidate_id': 1, 'status': 'fail',
                 'source_quote': 'window above the table', 'target_role': 'target',
                 'reference_role': 'reference', 'role_binding_status': 'bound',
                 'comparison_scope_ids': [1, 7],
                 'comparison_scope_status': 'bound',
                 'measurement_source': 'synthetic annotated-box relation geometry',
                 'explicit_fail_criterion': 'candidate 1 bottom is below the bound table top'},
                {'candidate_id': 2, 'status': 'pass'}]}
        result, _ = self.run_actions([
            {'action': 'evidence', 'dimension': 'position', 'reason': 'check relation'},
            {'action': 'evidence', 'dimension': 'appearance', 'reason': 'check appearance'},
            {'action': 'finish', 'predicted_id': 2, 'reason': 'combined evidence'},
        ], compute=compute, visual=[{'checks': [
            {'id': 2, 'verdict': 'yes', 'reason': 'visible match'}]}])
        self.assertEqual(result['active_trace']['final_candidate_ids'], [2])
        self.assertEqual(result['active_trace']['successful_tools'], 2)

    def test_duplicate_visual_verdicts_are_rejected(self):
        with self.assertRaises(ValueError):
            _check_payload({'checks': [
                {'id': 1, 'verdict': 'yes', 'reason': 'match'},
                {'id': 1, 'verdict': 'no', 'reason': 'mismatch'}]},
                [1], [obj(1, 'window', 0, 0)], parsed())

    def test_invalid_render_does_not_consume_successful_tool_budget(self):
        objects = [obj(1, 'window', 0, 0), obj(2, 'window', 2, 0)]
        responses = iter([
            {'_content': json.dumps({
                'action': 'evidence', 'dimension': 'appearance', 'focus_ids': [1, 2, 3, 4, 5, 6, 7],
                'azimuth_deg': 90, 'elevation_deg': 25,
                'context': 'scene', 'reason': 'too many focus instances',
            }), '_baseline': {'model': 'mock', 'latency_s': .01},
             '_finish_reason': 'stop', 'usage': {}},
            {'_content': json.dumps({
                'action': 'stop',
                'reason': 'no useful evidence after invalid tool request',
            }), '_baseline': {'model': 'mock', 'latency_s': .01},
             '_finish_reason': 'stop', 'usage': {}},
        ])
        parsed_query = parsed()
        with tempfile.TemporaryDirectory() as directory, patch(
                'eviground.grounding.active_diagnostic.chat', side_effect=lambda *args, **kwargs: next(responses)):
            trace, calls = {}, []
            result = active_ground(
                'window', parsed_query, objects, objects, objects,
                np.zeros((2, 3)), np.zeros((2, 3), dtype=np.uint8),
                np.array([1, 2]), Path(directory), calls, trace, 1, True)
        self.assertIsNone(result['predicted_id'])
        self.assertEqual(trace['stop_reason'], 'model_stop')
        self.assertEqual(len(trace['steps']), 2)
        self.assertTrue(trace['steps'][0]['observation']['no_evidence_acquired'])

    def test_duplicate_successful_action_is_recoverable(self):
        objects = [obj(1, 'window', 0, 0), obj(2, 'window', 2, 0)]
        action = {'action': 'evidence', 'dimension': 'size',
                  'reason': 'inspect size'}
        responses = iter([
            {'_content': json.dumps(action),
             '_baseline': {'model': 'mock', 'latency_s': .01},
             '_finish_reason': 'stop', 'usage': {}},
            {'_content': json.dumps(action),
             '_baseline': {'model': 'mock', 'latency_s': .01},
             '_finish_reason': 'stop', 'usage': {}},
            {'_content': json.dumps({'action': 'stop',
                                     'reason': 'no useful evidence after duplicate'}),
             '_baseline': {'model': 'mock', 'latency_s': .01},
             '_finish_reason': 'stop', 'usage': {}},
        ])
        messages_seen = []
        def fake_chat(messages, **kwargs):
            messages_seen.append(messages)
            return next(responses)
        with tempfile.TemporaryDirectory() as directory, patch(
                'eviground.grounding.active_diagnostic.chat', side_effect=fake_chat):
            trace, calls = {}, []
            result = active_ground(
                'window', parsed(), objects, objects, objects,
                np.zeros((2, 3)), np.zeros((2, 3), dtype=np.uint8),
                np.array([1, 2]), Path(directory), calls, trace, 2, True)
        self.assertIsNone(result['predicted_id'])
        self.assertTrue(trace['steps'][1]['observation']['no_evidence_acquired'])
        self.assertIn('Controller correction', messages_seen[2][0]['content'])

    def test_position_evidence_distinguishes_fail_from_unknown(self):
        diagnosis = {
            'candidate_ids': [1, 2, 3], 'stable_candidate_ids': [], 'missing': [],
            'geometry': {
                'relation_pairs': [],
                'between_candidate_evidence': [
                    {'target_id': 1, 'relation_index': 0, 'phrase': 'between the two tables',
                     'passing_reference_pairs': [[7, 8]],
                     'unknown_reference_pairs': [], 'failing_reference_pairs': []},
                    {'target_id': 2, 'relation_index': 0, 'phrase': 'between the two tables',
                     'passing_reference_pairs': [],
                     'unknown_reference_pairs': [], 'failing_reference_pairs': [[7, 8]]},
                    {'target_id': 3, 'relation_index': 0, 'phrase': 'between the two tables',
                     'passing_reference_pairs': [],
                     'unknown_reference_pairs': [[7, 8]], 'failing_reference_pairs': []},
                ],
            },
        }
        evidence = position_candidate_evidence(diagnosis)
        self.assertEqual([item['status'] for item in evidence],
                         ['pass', 'fail', 'unknown'])
        update = apply_computed_evidence(diagnosis, evidence, 'position')
        self.assertEqual(update['removed_ids'], [2])
        self.assertEqual(diagnosis['candidate_ids'], [1, 3])
        audit = diagnosis['active_computed_evidence'][0]['candidate_removal_audit']
        failed = next(item for item in audit if item['candidate_id'] == 2)
        unknown = next(item for item in audit if item['candidate_id'] == 3)
        self.assertTrue(failed['removal_allowed'])
        self.assertEqual(failed['comparison_scope_ids'], [2, 7, 8])
        self.assertEqual(failed['source_quote'], 'between the two tables')
        self.assertFalse(unknown['removal_allowed'])

    def test_distance_clauses_with_different_anchors_are_anded(self):
        rankings = [
            {'relation': 'nearest', 'phrase': 'nearest the table', 'metric': 'center',
             'anchor_binding': {'anchor_category': 'table'}, 'anchor_id': 7,
             'ordered_targets': [{'target_id': 1}, {'target_id': 2}], 'best_ids': [1]},
            {'relation': 'nearest', 'phrase': 'nearest the door', 'metric': 'center',
             'anchor_binding': {'anchor_category': 'door'}, 'anchor_id': 8,
             'ordered_targets': [{'target_id': 1}, {'target_id': 2}], 'best_ids': [2]},
        ]
        evidence = distance_candidate_evidence(rankings, [1, 2])
        self.assertEqual([item['status'] for item in evidence], ['fail', 'fail'])
        self.assertTrue(all(len(item['relations']) == 2 for item in evidence))

    def test_position_clauses_with_same_relation_are_anded(self):
        diagnosis = {'candidate_ids': [1], 'geometry': {'relation_pairs': [
            {'target_id': 1, 'relation': 'above', 'relation_index': 0, 'passes': True},
            {'target_id': 1, 'relation': 'above', 'relation_index': 1, 'passes': False},
        ]}}
        self.assertEqual(position_candidate_evidence(diagnosis)[0]['status'], 'fail')

    def test_unidentified_position_conditions_are_unknown(self):
        diagnosis = {'candidate_ids': [1], 'geometry': {'relation_pairs': [
            {'target_id': 1, 'relation': 'above', 'passes': True},
        ]}}
        evidence = position_candidate_evidence(diagnosis)[0]
        self.assertEqual(evidence['status'], 'unknown')
        self.assertIn('predicate-local', evidence['evidence_scope'])

    def test_render_rejects_overloaded_focus(self):
        objects = [obj(index, 'window', index, 0) for index in range(1, 8)]
        with self.assertRaises(ValueError):
            render_action(
                {'focus_ids': list(range(1, 8)), 'azimuth_deg': 90,
                 'elevation_deg': 25, 'context': 'scene'}, objects,
                np.zeros((1, 3)), np.zeros((1, 3), dtype=np.uint8),
                np.array([1]), Path('/tmp'), 1)

    def test_each_dimension_is_an_explicit_model_selectable_tool(self):
        objects = [obj(1, 'window', 0, 0), obj(2, 'window', 2, 0)]
        diagnosis = {
            'candidate_ids': [1, 2], 'geometry': {
                'relation_pairs': [], 'between_candidate_evidence': [],
                'between_triplets': [], 'candidate_metrics': [],
                'reference_metrics': [],
            }, 'hypotheses': [], 'visual_needs': [],
        }
        for dimension in ('viewpoint', 'position', 'distance', 'appearance', 'size'):
            with self.subTest(dimension=dimension):
                result = compute_observation(dimension, 'window', parsed(), diagnosis,
                                             objects, objects)
                self.assertEqual(result['tool'], 'compute')
                self.assertEqual(result['dimension'], dimension)

    def test_viewpoint_compute_separates_language_frame_from_capture_camera(self):
        objects = [obj(1, 'window', 0, 0), obj(2, 'window', 2, 0), obj(3, 'tv', 0, 2)]
        parsed_query = {
            'Target': ['window'], 'Anchor': ['tv'],
            'Relations': [{'target_category': 'window', 'anchor_category': '',
                           'relation': 'right', 'scope': 'side', 'phrase': 'on the right'}],
            'Observer': {
                'origin': {'kind': 'room_center', 'category': '', 'phrase': 'middle of the room'},
                'facing': {'category': 'tv', 'group': False, 'phrase': 'facing the tv'},
            },
            'Viewpoint': 'middle of the room facing the tv',
            'Attributes': [], 'Unhandled': [], 'diagnostic_missing': [],
        }
        parsed_query = validate_structure('middle of the room facing the tv, the window on the right',
                                           parsed_query)
        diagnosis = diagnose(parsed_query, objects, objects[:2], [0, 0, .5],
                             trusted_categories=True)
        result = compute_observation('viewpoint', parsed_query['Viewpoint'], parsed_query,
                                     diagnosis, objects, objects)
        self.assertEqual(result['status'], 'known')
        self.assertTrue(result['language_reference_frames'])
        self.assertTrue(result['conditions_share_hypothesis'])
        self.assertFalse(result['capture_camera']['affects_language_frame'])
        frame = result['language_reference_frames'][0]['frame']
        self.assertEqual(frame['frame_type'], 'language_reference')

    def test_visual_check_only_accepts_rendered_candidate_ids(self):
        objects = [obj(1, 'window', 0, 0), obj(2, 'window', 2, 0)]
        result = _check_payload(
            {'checks': [{'id': 1, 'verdict': 'no', 'reason': 'visible mismatch'}]},
            [1], objects, parsed())
        self.assertEqual(result[0]['id'], 1)
        self.assertEqual(result[0]['verdict'], 'no')
        with self.assertRaises(ValueError):
            _check_payload(
                {'checks': [{'id': 2, 'verdict': 'no', 'reason': 'not rendered'}]},
                [1], objects, parsed())

    def test_visual_no_excludes_candidate_with_annotated_categories(self):
        objects = [obj(1, 'window', 0, 0), obj(2, 'window', 2, 0)]
        query = validate_structure('the window', parsed())
        initial = diagnose(query, objects, objects, [0, 0, .5], trusted_categories=True)
        self.assertEqual(initial['candidate_ids'], [1, 2])
        checked = diagnose(query, objects, objects, [0, 0, .5],
                           checks=[{'id': 1, 'category': 'window', 'role': 'target',
                                    'verdict': 'no', 'visibility': 'visible',
                                    'reason': 'solid wall surface contradicts a window category'}],
                           trusted_categories=True)
        self.assertEqual(checked['candidate_ids'], [2])
        self.assertEqual(checked['status'], 'resolved')

    def test_visual_uncertain_preserves_candidate(self):
        objects = [obj(1, 'window', 0, 0), obj(2, 'window', 2, 0)]
        checked = diagnose(parsed(), objects, objects, [0, 0, .5],
                           checks=[{'id': 1, 'category': 'window', 'verdict': 'uncertain'}],
                           trusted_categories=True)
        self.assertEqual(checked['candidate_ids'], [1, 2])
        self.assertEqual(checked['status'], 'missing')


if __name__ == '__main__':
    unittest.main()
