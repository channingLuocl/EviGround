import json
from itertools import product
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from seeground.experiments.replay_module_acceptance import replay_batch, replay_case, replay_inference, replay_saved_parse


def saved_parse(raw, finish='stop', extra_calls=()):
    return {'query': 'Choose the skinny cabinet.',
            'parsed_query': {'Target': ['obsolete']},
            'calls': [{'stage': 'parse_query', 'response': json.dumps(raw),
                       'finish_reason': finish}, *extra_calls]}


def raw_parse():
    return {'Target': ['cabinet'], 'Anchor': [], 'Relations': [],
            'Observer': {}, 'Attributes': [], 'Unhandled': []}


class ReplayModuleAcceptanceTests(unittest.TestCase):
    def setUp(self):
        network = patch('requests.sessions.Session.request',
                        side_effect=AssertionError('Offline replay must not access the network'))
        network.start()
        self.addCleanup(network.stop)

    def test_uses_raw_parser_response_instead_of_saved_validated_parse(self):
        parsed, audit = replay_saved_parse('Choose the skinny cabinet.', saved_parse(raw_parse()))
        self.assertEqual(parsed['Target'], ['cabinet'])
        self.assertIn('appearance condition not represented: skinny', parsed['diagnostic_missing'])
        self.assertTrue(audit['attribute_repair_unavailable'])
        self.assertFalse(audit['saved_attribute_repair_replayed'])
        self.assertEqual(audit['network_requests'], 0)

    def test_available_attribute_repair_uses_current_production_validation(self):
        repair = {'stage': 'parse_query_attribute_repair', 'finish_reason': 'stop',
                  'response': json.dumps({'Attributes': [{'category': 'cabinet', 'phrase': 'skinny'}]})}
        parsed, audit = replay_saved_parse('Choose the skinny cabinet.',
                                          saved_parse(raw_parse(), extra_calls=[repair]))
        self.assertEqual(parsed['Attributes'], [{'category': 'cabinet', 'phrase': 'skinny'}])
        self.assertNotIn('appearance condition not represented: skinny', parsed['diagnostic_missing'])
        self.assertTrue(audit['saved_attribute_repair_replayed'])
        self.assertNotIn('error', audit['attribute_repair_calls'][0])

    def test_invalid_saved_repair_keeps_missing_and_reports_rejection(self):
        repair = {'stage': 'parse_query_attribute_repair', 'finish_reason': 'stop',
                  'response': json.dumps({'Attributes': [{'category': 'door', 'phrase': 'skinny'}]})}
        parsed, audit = replay_saved_parse('Choose the skinny cabinet.',
                                          saved_parse(raw_parse(), extra_calls=[repair]))
        self.assertEqual(parsed['Attributes'], [])
        self.assertIn('appearance condition not represented: skinny', parsed['diagnostic_missing'])
        self.assertIn('error', audit['attribute_repair_calls'][0])

    def test_truncated_response_retains_missing_condition(self):
        parsed, audit = replay_saved_parse('cabinet', saved_parse(raw_parse(), finish='length'))
        self.assertIn('query parser response truncated', parsed['diagnostic_missing'])
        self.assertTrue(any(row.get('reason') == 'query parser response truncated'
                            for row in parsed['condition_records']))
        self.assertFalse(audit['saved_attribute_repair_replayed'])

    def test_saved_syntax_repair_does_not_reinterpret_raw_roles(self):
        raw = raw_parse()
        raw.update(Target=['trash can'], Anchor=['door'], Relations=[{
            'target_category': 'trash can', 'anchor_category': 'door',
            'relation': 'nearest', 'scope': '', 'phrase': 'closest to the door'}])
        saved = {'calls': [{'stage': 'parse_query', 'response': '{broken', 'finish_reason': 'stop'},
                          {'stage': 'parse_query_json_repair', 'response': json.dumps(raw),
                           'finish_reason': 'stop'}]}
        parsed, audit = replay_saved_parse('the trash can closest to the door', saved)
        self.assertEqual(audit['raw_parser_output'], raw)
        self.assertEqual(parsed['Relations'][0]['anchor_category'], 'door')
        self.assertTrue(audit['syntax_repair_reused'])

    def test_missing_raw_response_is_reported_without_using_final_parse(self):
        with self.assertRaisesRegex(ValueError, 'raw parse_query'):
            replay_saved_parse('cabinet', {'parsed_query': raw_parse()})

    def test_other_two_replay_uses_original_query_for_applicable_hypotheses(self):
        query = "It's the box that's between the other two boxes"
        raw = {'Target': ['box'], 'Anchor': ['box', 'box'],
               'Relations': [{'target_category': 'box', 'anchor_categories': ['box', 'box'],
                              'relation': 'between', 'phrase': "the box that's between the other two boxes"}],
               'Observer': {'origin': {}, 'facing': {'category': '', 'group': False, 'phrase': ''}},
               'Attributes': [], 'Unhandled': []}
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory)
            objects = [{'bbox_id': oid, 'target': 'box', 'bbox_3d': [x, 0, .5, .4, .4, 1]}
                       for oid, x in ((1, 0), (2, -2), (3, 2))]
            (scene / 'objects_gt.json').write_text(json.dumps(objects))
            np.savez(scene / 'scene.npz', xyz=np.array([[-2, 0, 0], [0, 0, .5], [2, 0, 1]]),
                     ids=np.array([2, 1, 3]))
            current = replay_inference(query, saved_parse(raw), scene)
        diagnosis = current['initial_diagnosis']
        self.assertEqual(diagnosis['status'], 'ambiguous')
        self.assertEqual(diagnosis['candidate_ids'], [1])
        support = diagnosis['candidate_hypothesis_support']
        self.assertTrue(support['enabled'])
        self.assertEqual(next(row for row in support['candidates']
                              if row['candidate_id'] == 1)['status'], 'pass')
        self.assertEqual(current['parser_replay']['network_requests'], 0)

    def test_batch_replays_geometry_without_visuals_or_evaluation_id_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved_root, scene_root = root / 'saved', root / 'scenes'
            scene = scene_root / 'scene0000_00'
            scene.mkdir(parents=True)
            objects = [{'bbox_id': 1, 'target': 'cabinet', 'bbox_3d': [0, 0, .5, .4, .4, 1]},
                       {'bbox_id': 2, 'target': 'cabinet', 'bbox_3d': [2, 0, .5, 1, 1, 1]}]
            (scene / 'objects_gt.json').write_text(json.dumps(objects))
            np.savez(scene / 'scene.npz', xyz=np.array([[0, 0, 0], [.4, .4, 1],
                                                      [2, 0, 0], [3, 1, 1]]),
                     ids=np.array([1, 1, 2, 2]))
            query = 'Choose the largest cabinet.'
            row = {'csv_row': 1, 'scene': scene.name, 'query': query, 'module': 'size',
                   'target_id': 1, 'stimulus_candidate_ids': [1]}
            case_dir = saved_root / 'q00001'
            case_dir.mkdir(parents=True)
            raw = raw_parse()
            raw['Attributes'] = [{'category': 'cabinet', 'phrase': 'largest'}]
            saved = saved_parse(raw)
            saved.update(query=query, diagnosis={'checks': [{'id': 1, 'verdict': 'yes'}]})
            saved['calls'].append({'stage': 'active_evidence_visual_check', 'response': 'stale visual'})
            (case_dir / 'run.json').write_text(json.dumps(saved))
            (saved_root / 'manifest.json').write_text(json.dumps({'selection': {'rows': [row]}}))
            result = replay_batch(saved_root, scene_root, include_size=True)
            changed = replay_case({**row, 'target_id': 2, 'stimulus_candidate_ids': [2]},
                                  saved_root, scene_root, include_size=True)
        current = result['cases'][0]['current']
        self.assertEqual(result['summary']['replay_errors'], 0)
        self.assertEqual(current['initial_candidate_ids'], [1, 2])
        self.assertEqual(current, changed['current'])
        self.assertEqual(current['initial_diagnosis']['status'], 'missing')
        self.assertFalse(current['saved_visual_evidence_reused'])
        self.assertTrue(current['size_observation_applied'])
        self.assertEqual(current['size_diagnosis']['candidate_ids'], [2])
        self.assertFalse(result['cases'][0]['evaluation_only']['target_in_current_candidates'])
        self.assertTrue(changed['evaluation_only']['target_in_current_candidates'])
        self.assertEqual(current['size_observation']['comparison_pool_ids'], [1, 2])
        self.assertEqual(result['network_requests'], 0)

    def test_joint_relative_height_replay_preserves_active_hypotheses(self):
        query = 'The short bookshelf in between the two tall bookshelves'
        raw = {'Target': ['bookshelf'], 'Anchor': ['bookshelf', 'bookshelf'],
               'Relations': [{'target_category': 'bookshelf', 'anchor_categories': ['bookshelf', 'bookshelf'],
                              'relation': 'between', 'phrase': 'in between the two tall bookshelves'}],
               'Observer': {}, 'Attributes': [{'category': 'bookshelf', 'phrase': 'short'}], 'Unhandled': []}
        objects = [{'bbox_id': oid, 'target': 'bookshelf', 'bbox_3d': [x, 0, height / 2, .8, .5, height]}
                   for oid, x, height in ((1, 0, 1), (10, -2, 2), (11, 2, 2), (90, 10, .3))]
        points, ids = [], []
        for obj in objects:
            center = np.asarray(obj['bbox_3d'][:3])
            half = np.asarray(obj['bbox_3d'][3:]) / 2
            points.extend(center + half * np.asarray(signs) for signs in product((-1, 1), repeat=3))
            ids.extend([obj['bbox_id']] * 8)
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory)
            (scene / 'objects_gt.json').write_text(json.dumps(objects))
            np.savez(scene / 'scene.npz', xyz=np.asarray(points), ids=np.asarray(ids))
            current = replay_inference(query, saved_parse(raw), scene, include_size=True)
        observation = current['size_observation']
        joint = observation['joint_size_diagnosis']
        self.assertIsNotNone(joint)
        self.assertEqual(current['size_diagnosis']['candidate_ids'], joint['candidate_ids'])
        self.assertEqual(current['size_diagnosis']['hypotheses'], joint['hypotheses'])
        self.assertEqual(current['size_diagnosis']['candidate_hypothesis_support'],
                         joint['candidate_hypothesis_support'])
        self.assertEqual(current['joint_measurements'], observation['comparison_metrics'])
        self.assertEqual(current['size_diagnosis']['candidate_ids'], [1])
        self.assertEqual(current['size_diagnosis']['pending_size_conditions'], [])
        self.assertEqual(current['parser_replay']['network_requests'], 0)


if __name__ == '__main__':
    unittest.main()
