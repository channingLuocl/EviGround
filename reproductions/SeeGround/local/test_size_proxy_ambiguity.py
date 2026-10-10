import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from active_diagnostic import active_ground, compute_observation, size_candidate_evidence
from diagnostic import diagnose, validate_structure
from object_geometry import point_cloud_measurements


class SizeProxyAmbiguityTests(unittest.TestCase):
    def setUp(self):
        # The first object has a smaller face but a greater volume.
        self.objects = [
            {'bbox_id': 1, 'target': 'book', 'bbox_3d': [0, 0, .6, .4, .3, .12]},
            {'bbox_id': 2, 'target': 'book', 'bbox_3d': [1, 0, .6, .4, .42, .06]}]
        self.xyz = np.concatenate([np.array(obj['bbox_3d'][:3]) +
            np.array([-.5, .5])[:, None] * np.array(obj['bbox_3d'][3:]) for obj in self.objects])
        self.ids = np.array([1, 1, 2, 2])
        self.query = 'The book that is smallest.'
        self.parsed = validate_structure(self.query, {'Target': ['book'], 'Anchor': [],
            'Relations': [], 'Attributes': [{'category': 'book', 'phrase': 'smallest'}],
            'Observer': {}, 'Unhandled': []})

    def test_box_and_point_cloud_paths_preserve_conflicting_proxy_candidates(self):
        diagnosis = diagnose(self.parsed, self.objects, self.objects, [0, 0, .5], trusted_categories=True)
        self.assertEqual(diagnosis['status'], 'missing')
        self.assertEqual(diagnosis['candidate_ids'], [1, 2])
        self.assertIn('volume and footprint extremes disagree', ' '.join(diagnosis['missing']))
        observation = compute_observation('size', self.query, self.parsed, diagnosis,
                                          self.objects, self.objects, self.xyz, self.ids)
        condition, = observation['size_constraints']
        self.assertEqual(condition['status'], 'unknown')
        self.assertEqual(condition['comparison_pool_ids'], [1, 2])
        self.assertEqual(condition['selected_ids'], [])
        self.assertTrue(all(row['status'] == 'unknown' for row in condition['candidate_evidence']))
        self.assertEqual([c['status'] for c in observation['candidate_evidence']], ['unknown', 'unknown'])
        ambiguity = observation['size_decision']['metric_ambiguities']['smallest']
        self.assertEqual(ambiguity['proxy_selected_ids'], {'volume_m3': [2], 'footprint_m2': [1]})

    def test_reduced_live_pool_cannot_hide_frozen_comparison_disagreement(self):
        measurements = point_cloud_measurements(self.objects, self.xyz, self.ids)
        evidence, constraints, decision = size_candidate_evidence(self.parsed, self.objects[1:],
            measurements, comparison_pool_ids=[1, 2], comparison_scope_bound=True)
        self.assertEqual(evidence[0]['status'], 'unknown')
        condition, = constraints
        self.assertEqual(condition['status'], 'unknown')
        self.assertEqual(condition['comparison_pool_ids'], [1, 2])
        self.assertTrue(condition['comparison_pool_frozen'])
        self.assertEqual(condition['selected_ids'], [])
        self.assertEqual(condition['metric_ambiguity']['proxy_selected_ids'],
                         {'volume_m3': [2], 'footprint_m2': [1]})
        self.assertEqual(decision['candidate_relative_ids'], [])

    def test_planner_finish_cannot_override_proxy_ambiguity(self):
        actions = iter([{'action': 'evidence', 'dimension': 'size', 'reason': 'compare'},
                        {'action': 'finish', 'predicted_id': 2, 'reason': 'volume minimum'},
                        {'action': 'stop', 'reason': 'ambiguous metric'}])
        def chat(*args, **kwargs):
            return {'_content': json.dumps(next(actions)), '_baseline': {'latency_s': .01},
                    '_finish_reason': 'stop', 'usage': {}}
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict('os.environ', {'SEEGROUND_REACT_BACKEND': 'qwen'}), \
                patch('active_diagnostic.chat', side_effect=chat):
            result = active_ground(self.query, self.parsed, self.objects, self.objects,
                self.objects, self.xyz, np.zeros_like(self.xyz, dtype=np.uint8), self.ids,
                Path(directory), [], {}, 2, True)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['active_trace']['final_candidate_ids'], [1, 2])
        self.assertTrue(result['active_trace']['final_evidence_state']['pending_size_conditions'])

    def test_reference_size_disagreement_preserves_both_anchor_interpretations(self):
        query = 'The chair nearest the smallest book.'
        parsed = validate_structure(query, {'Target': ['chair'], 'Anchor': ['book'],
            'Relations': [{'target_category': 'chair', 'anchor_category': 'book',
                           'relation': 'nearest', 'scope': 'relative_anchor',
                           'phrase': 'nearest the smallest book'}],
            'Attributes': [{'category': 'book', 'role': 'reference', 'phrase': 'smallest'}],
            'Observer': {}, 'Unhandled': []})
        chairs = [{'bbox_id': oid, 'target': 'chair', 'bbox_3d': [x, 0, .5, .4, .4, 1]}
                  for oid, x in [(10, 0), (11, 1)]]
        diagnosis = diagnose(parsed, self.objects + chairs, chairs, [0, 0, .5], trusted_categories=True)
        self.assertEqual(diagnosis['status'], 'missing')
        self.assertEqual(diagnosis['candidate_ids'], [10, 11])
        self.assertEqual({h['references']['book'] for h in diagnosis['hypotheses']}, {1, 2})
        self.assertIn('reference unqualified size metric is ambiguous', ' '.join(diagnosis['missing']))

    def test_explicit_height_extreme_keeps_its_measurement_semantics(self):
        parsed = validate_structure('The tallest book.', {'Target': ['book'], 'Anchor': [],
            'Relations': [], 'Attributes': [{'category': 'book', 'phrase': 'tallest'}],
            'Observer': {}, 'Unhandled': []})
        diagnosis = diagnose(parsed, self.objects, self.objects, [0, 0, .5], trusted_categories=True)
        self.assertEqual(diagnosis['status'], 'resolved')
        self.assertEqual(diagnosis['candidate_ids'], [1])


if __name__ == '__main__':
    unittest.main()
