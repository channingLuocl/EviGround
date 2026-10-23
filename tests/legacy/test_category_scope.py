import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eviground.grounding.active_diagnostic import active_ground, compute_observation
from eviground.grounding.diagnostic import diagnose, validate_structure
from eviground.grounding.evidence import match_retrieval_categories


class CategorySizeScopeTests(unittest.TestCase):
    def test_mixed_subtypes_preserve_candidates_and_cannot_certify_size(self):
        query = 'The smallest of the tables.'
        parsed = validate_structure(query, {'Target': ['table'], 'Anchor': [], 'Relations': [],
            'Attributes': [{'category': 'table', 'phrase': 'smallest'}], 'Observer': {}, 'Unhandled': []})
        objects = [{'bbox_id': 1, 'target': 'table', 'bbox_3d': [0, 0, 1, 2, 2, 2]},
                   {'bbox_id': 2, 'target': 'coffee table', 'bbox_3d': [4, 0, .5, 1, 1, 1]}]
        pool = match_retrieval_categories(['table'], objects)
        self.assertEqual(pool, objects)
        diagnostic = diagnose(parsed, objects, pool, [0, 0, .5], trusted_categories=True)
        self.assertEqual(diagnostic['status'], 'missing')
        self.assertEqual(diagnostic['candidate_ids'], [1, 2])
        self.assertEqual(diagnostic['geometry']['size_constraints'][0]['scope_status'], 'unknown')
        diagnostic = diagnose(parsed, objects, pool, [0, 0, .5], trusted_categories=True,
                              apply_size_constraints=False)
        diagnostic['size_comparison_pool_ids'] = [1, 2]
        xyz = np.array([[-1, -1, 0], [1, 1, 2], [3.5, -.5, 0], [4.5, .5, 1]])
        ids = np.array([1, 1, 2, 2])
        observation = compute_observation('size', query, parsed, diagnostic, pool, objects, xyz, ids)
        self.assertEqual(observation['size_decision']['scope_status'], 'unknown')
        self.assertEqual(observation['size_decision']['comparison_category_labels'], ['coffee table', 'table'])
        condition, = observation['size_constraints']
        self.assertEqual(condition['status'], 'unknown')
        self.assertEqual(condition['scope_status'], 'unknown')
        self.assertEqual(condition['comparison_pool_ids'], [1, 2])
        self.assertEqual(condition['scope_unknown_ids'], [1, 2])
        self.assertEqual(condition['selected_ids'], [])
        self.assertTrue(all(row['status'] == 'unknown' for row in condition['candidate_evidence']))
        self.assertTrue(all(row['status'] == 'unknown' for row in observation['candidate_evidence']))
        actions = iter([{'action': 'evidence', 'dimension': 'size', 'reason': 'compare'},
                        {'action': 'finish', 'predicted_id': 2, 'reason': 'small coffee table'},
                        {'action': 'stop', 'reason': 'unbound subtype comparison'}])
        def chat(*args, **kwargs):
            import json
            return {'_content': json.dumps(next(actions)), '_baseline': {'latency_s': .01},
                    '_finish_reason': 'stop', 'usage': {}}
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ', {'EVIGROUND_REACT_BACKEND': 'qwen'}), \
                patch('eviground.grounding.active_diagnostic.chat', side_effect=chat):
            result = active_ground(query, parsed, objects, objects, pool, xyz,
                np.zeros_like(xyz, dtype=np.uint8), ids, Path(directory), [], {}, 2, True)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['active_trace']['final_candidate_ids'], [1, 2])

    def test_homogeneous_subtype_aliases_keep_valid_full_pool(self):
        query = 'The smallest of the cabinets.'
        parsed = validate_structure(query, {'Target': ['cabinet'], 'Anchor': [], 'Relations': [],
            'Attributes': [{'category': 'cabinet', 'phrase': 'smallest'}], 'Observer': {}, 'Unhandled': []})
        objects = [{'bbox_id': 1, 'target': 'kitchen cabinet', 'bbox_3d': [0, 0, 1, 2, 2, 2]},
                   {'bbox_id': 2, 'target': 'kitchen cabinets', 'bbox_3d': [4, 0, .5, 1, 1, 1]}]
        result = diagnose(parsed, objects, objects, [0, 0, .5], trusted_categories=True)
        self.assertEqual(result['status'], 'resolved')
        self.assertEqual(result['candidate_ids'], [2])

    def test_mixed_reference_size_scope_does_not_choose_an_anchor(self):
        query = 'The chair nearest the largest table.'
        parsed = validate_structure(query, {'Target': ['chair'], 'Anchor': ['table'],
            'Relations': [{'target_category': 'chair', 'anchor_category': 'table',
                           'relation': 'nearest', 'scope': 'relative_anchor',
                           'phrase': 'nearest the largest table'}],
            'Attributes': [{'category': 'table', 'role': 'reference', 'phrase': 'largest'}],
            'Observer': {}, 'Unhandled': []})
        objects = [{'bbox_id': oid, 'target': category, 'bbox_3d': [x, 0, .5, width, width, 1]}
                   for oid, category, x, width in [(1, 'chair', 0, .5), (2, 'chair', 6, .5),
                                                   (10, 'table', 0, 2), (11, 'coffee table', 6, 1)]]
        result = diagnose(parsed, objects, objects[:2], [3, 0, .5], trusted_categories=True)
        self.assertEqual(result['status'], 'missing')
        self.assertEqual(result['candidate_ids'], [1, 2])
        self.assertIn('reference size comparison scope is not fully bound', result['missing'])
        condition = next(row for row in result['geometry']['size_constraints']
                         if row['role'] == 'reference')
        self.assertEqual(condition['scope_status'], 'unknown')
        self.assertEqual(condition['comparison_pool_ids'], [10, 11])
        self.assertEqual(condition['selected_ids'], [])
        self.assertTrue(all(row['status'] == 'unknown' for row in condition['candidate_evidence']))
        self.assertEqual({h['references']['table'] for h in result['hypotheses']}, {10, 11})


if __name__ == '__main__':
    unittest.main()
