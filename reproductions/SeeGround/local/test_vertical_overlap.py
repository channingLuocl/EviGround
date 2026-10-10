"""Overlapping boxes keep vertical language unproved without losing recall."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from active_diagnostic import (active_ground, apply_computed_evidence,
                               execute_evidence)
from diagnostic import diagnose, validate_structure


def object_box(oid, category, z, height=1., x=0.):
    return {'bbox_id': oid, 'target': category,
            'bbox_3d': [x, 0., z, .4, .4, height]}


def parse(relation):
    query = f'window {relation} the door'
    return query, validate_structure(query, {
        'Target': ['window'], 'Anchor': ['door'], 'Observer': {},
        'Relations': [{'target_category': 'window', 'anchor_category': 'door',
                       'relation': relation, 'scope': 'relative_anchor',
                       'phrase': f'{relation} the door'}],
        'Attributes': [], 'Unhandled': [],
    })


class VerticalOverlapTests(unittest.TestCase):
    def test_partial_overlap_is_unknown_in_both_vertical_directions(self):
        for relation, center in (('above', 1.3), ('below', .7)):
            with self.subTest(relation=relation):
                _, parsed = parse(relation)
                objects = [object_box(1, 'window', center),
                           object_box(2, 'door', 1.)]
                result = diagnose(parsed, objects, objects[:1], [0, 0, 1],
                                  trusted_categories=True)
                self.assertEqual(result['candidate_ids'], [1])
                self.assertEqual(result['status'], 'missing')
                pair, = result['geometry']['relation_pairs']
                self.assertEqual(pair['status'], 'unknown')
                self.assertIsNone(pair['passes'])
                self.assertFalse(pair['vertical_separation_passes'])

    def test_separation_passes_and_opposite_direction_still_fails(self):
        for relation, positive, negative in (('above', 2.5, .2),
                                              ('below', -.5, 1.8)):
            with self.subTest(relation=relation):
                _, parsed = parse(relation)
                objects = [object_box(1, 'window', positive),
                           object_box(2, 'window', negative),
                           object_box(3, 'door', 1.)]
                result = diagnose(parsed, objects, objects[:2], [0, 0, 1],
                                  trusted_categories=True)
                self.assertEqual(result['candidate_ids'], [1])
                self.assertEqual(result['status'], 'resolved')
                self.assertEqual([pair['status'] for pair in
                                  result['geometry']['relation_pairs']],
                                 ['pass', 'fail'])

    def test_far_horizontal_or_coincident_centers_are_not_retained(self):
        query, parsed = parse('above')
        objects = [object_box(1, 'window', 1.3, x=2.),
                   object_box(2, 'window', 1.), object_box(3, 'door', 1.)]
        result = diagnose(parsed, objects, objects[:2], [0, 0, 1],
                          trusted_categories=True, query=query)
        self.assertEqual(result['candidate_ids'], [])
        self.assertEqual(result['status'], 'conflict')
        self.assertTrue(all(pair['status'] == 'fail' for pair in
                            result['geometry']['relation_pairs']))

    def test_position_operation_preserves_unknown_and_cannot_resolve_it(self):
        query, parsed = parse('above')
        objects = [object_box(1, 'window', 1.3), object_box(2, 'door', 1.)]
        result = diagnose(parsed, objects, objects[:1], [0, 0, 1],
                          trusted_categories=True)
        with tempfile.TemporaryDirectory() as directory:
            with patch('active_diagnostic.chat') as chat:
                observation = execute_evidence(
                    {'dimension': 'position', 'reason': 'verify vertical condition'},
                    query, parsed, result, objects[:1], objects,
                    None, None, None, Path(directory), 1, [])
            chat.assert_not_called()
        self.assertEqual(observation['operation_status'], 'partial')
        self.assertEqual(observation['coverage']['unknown_ids'], [1])
        self.assertEqual(observation['candidate_evidence'][0]['status'], 'unknown')
        update = apply_computed_evidence(
            result, observation['candidate_evidence'], 'position')
        self.assertEqual(update['removed_ids'], [])
        self.assertEqual(result['candidate_ids'], [1])
        self.assertEqual(result['status'], 'missing')

    def test_unique_unknown_candidate_cannot_be_accepted(self):
        query, parsed = parse('above')
        objects = [object_box(1, 'window', 1.3), object_box(2, 'door', 1.)]
        actions = [
            {'action': 'finish', 'predicted_id': 1, 'reason': 'only candidate'},
            {'action': 'stop', 'reason': 'vertical condition remains unproved'},
        ]
        responses = [{'_content': json.dumps(action), '_finish_reason': 'stop',
                      '_baseline': {'model': 'mock', 'latency_s': 0}, 'usage': {}}
                     for action in actions]
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                'os.environ', {'SEEGROUND_REACT_BACKEND': 'qwen'}), patch(
                'active_diagnostic.chat', side_effect=responses):
            trace = {}
            result = active_ground(
                query, parsed, objects, objects, objects[:1],
                np.array([[0., 0., 0.], [0., 0., 2.]]),
                np.zeros((2, 3), dtype=np.uint8), np.array([1, 2]),
                Path(directory), [], trace, 2, True)
        self.assertFalse(result['accepted'])
        self.assertIsNone(result['predicted_id'])
        self.assertEqual(result['decision_status'], 'missing')
        self.assertEqual(trace['rejected_actions'], 1)


if __name__ == '__main__':
    unittest.main()
