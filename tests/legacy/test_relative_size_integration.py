import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eviground.grounding.active_diagnostic import (active_ground, apply_size_observation,
                               compute_observation)
from eviground.grounding.diagnostic import diagnose, validate_structure, size_conditions, size_condition_scope
from eviground.geometry.object_geometry import point_cloud_measurements
from eviground.query.query_rewrite import rewrite_query


def bookshelf(oid, x, height, category='bookshelf', z=None):
    return {'bbox_id': oid, 'target': category,
            'bbox_3d': [x, 0., height / 2 if z is None else z, .4, .4, height]}


def cloud_for(objects):
    points, ids = [], []
    for obj in objects:
        center = np.asarray(obj['bbox_3d'][:3])
        extent = np.asarray(obj['bbox_3d'][3:])
        corners = np.asarray(list(itertools.product((-.5, .5), repeat=3)))
        points.extend(center + corners * extent)
        ids.extend([obj['bbox_id']] * 8)
    return np.asarray(points), np.asarray(ids)


def between_parsed():
    query = 'The short bookshelf in between the two tall bookshelves'
    parsed = validate_structure(query, {
        'Target': ['bookshelf'], 'Anchor': ['bookshelf', 'bookshelf'],
        'Attributes': [{'category': 'bookshelf', 'role': 'target', 'phrase': 'short'},
                       {'category': 'bookshelf', 'role': 'reference', 'phrase': 'tall'}],
        'Relations': [{'target_category': 'bookshelf', 'anchor_categories': ['bookshelf', 'bookshelf'],
                       'relation': 'between', 'scope': 'relative_anchors',
                       'phrase': 'in between the two tall bookshelves'}],
        'Observer': {}, 'Unhandled': []})
    parsed['query_rewrite'] = rewrite_query(query, parsed)
    return query, parsed


def ordinary_and_active_size(objects, query, parsed):
    xyz, ids = cloud_for(objects)
    measured = point_cloud_measurements(objects, xyz, ids)
    ordinary = diagnose(parsed, objects, objects, [0., 0., 1.], trusted_categories=True,
                        size_measurements=measured)
    deferred = diagnose(parsed, objects, objects, [0., 0., 1.], trusted_categories=True,
                        apply_size_constraints=False, size_measurements=measured)
    deferred['size_comparison_pool_ids'] = [obj['bbox_id'] for obj in objects
        if obj['target'] == parsed['Target'][0]]
    observation = compute_observation('size', query, parsed, deferred, objects, objects, xyz, ids)
    applied = apply_size_observation(copy.deepcopy(deferred), observation)
    return ordinary, observation, applied


class RelativeSizeIntegrationTests(unittest.TestCase):
    def test_local_short_does_not_require_global_minimum(self):
        objects = [bookshelf(10, 0., 1.), bookshelf(20, -2., 2.),
                   bookshelf(30, 2., 2.), bookshelf(40, 20., .5)]
        query, parsed = between_parsed()
        ordinary, observation, _ = ordinary_and_active_size(objects, query, parsed)
        self.assertEqual(ordinary['candidate_ids'], [10])
        self.assertEqual(ordinary['status'], 'resolved')
        self.assertLess(objects[-1]['bbox_3d'][5], objects[0]['bbox_3d'][5])
        joint = observation['joint_size_diagnosis']
        self.assertEqual(joint['candidate_ids'], [10])
        for hypothesis in joint['hypotheses']:
            self.assertEqual(set(hypothesis['references'].values()), {20, 30})
        proof = next(c for c in joint['geometry']['size_constraints']
                     if c.get('target_id') == 10 and set(c.get('reference_ids', [])) == {20, 30})
        self.assertEqual(proof['comparison_pool_ids'], [10, 20, 30])
        self.assertEqual(proof['candidate_evidence'][0]['status'], 'pass')

    def test_multiple_legal_reference_combinations_keep_both_targets(self):
        objects = [bookshelf(10, -2., 1.), bookshelf(11, 2., 1.),
                   bookshelf(20, -3., 2.), bookshelf(21, -1., 2.),
                   bookshelf(22, 1., 2.), bookshelf(23, 3., 2.)]
        query, parsed = between_parsed()
        ordinary, observation, _ = ordinary_and_active_size(objects, query, parsed)
        self.assertTrue({10, 11}.issubset(ordinary['candidate_ids']))
        self.assertNotEqual(ordinary['status'], 'resolved')
        self.assertEqual(ordinary['stable_candidate_ids'], [])
        hypotheses = observation['joint_size_diagnosis']['hypotheses']
        pairs = {frozenset(h['references'].values()) for h in hypotheses}
        self.assertIn(frozenset({20, 21}), pairs)
        self.assertIn(frozenset({22, 23}), pairs)
        self.assertTrue(all(len(h['references']) == len(set(h['references'].values()))
                            for h in hypotheses))

    def test_normal_compute_apply_have_same_joint_hypotheses_and_status(self):
        objects = [bookshelf(10, 0., 1.), bookshelf(20, -2., 2.),
                   bookshelf(30, 2., 2.), bookshelf(40, 20., .5)]
        query, parsed = between_parsed()
        ordinary, _, applied = ordinary_and_active_size(objects, query, parsed)
        self.assertEqual(applied['candidate_ids'], ordinary['candidate_ids'])
        self.assertEqual(applied['stable_candidate_ids'], ordinary['stable_candidate_ids'])
        self.assertEqual(applied['status'], ordinary['status'])
        self.assertEqual(applied['pending_size_conditions'], [])
        self.assertEqual(applied['hypotheses'], ordinary['hypotheses'])

    def test_equal_tall_shelves_stay_unknown_when_position_is_unique(self):
        objects = [bookshelf(10, 0., 2., 'shelf', 2.5),
                   bookshelf(11, 0., 2., 'shelf', -1.5),
                   bookshelf(20, 0., 1., 'table', .5)]
        query = 'The tall shelf above the table.'
        parsed = validate_structure(query, {
            'Target': ['shelf'], 'Anchor': ['table'],
            'Attributes': [{'category': 'shelf', 'phrase': 'tall'}],
            'Relations': [{'target_category': 'shelf', 'anchor_category': 'table',
                           'relation': 'above', 'scope': 'relative_anchor', 'phrase': 'above the table'}],
            'Observer': {}, 'Unhandled': []})
        ordinary, observation, applied = ordinary_and_active_size(objects, query, parsed)
        self.assertEqual(ordinary['candidate_ids'], [10])
        self.assertEqual(ordinary['status'], 'missing')
        self.assertEqual([e['status'] for e in observation['candidate_evidence']], ['unknown'])
        self.assertEqual(applied['candidate_ids'], [10])
        self.assertTrue(applied['pending_size_conditions'])
        self.assertEqual(observation['size_constraints'][0]['scope_status'], 'unknown')
        self.assertIn('not fully bound', observation['size_constraints'][0]['reason'])
        xyz, ids = cloud_for(objects)
        for limit in (0, 2):
            with self.subTest(limit=limit):
                actions = iter([{'action': 'evidence', 'dimension': 'size', 'reason': 'check tall'},
                                {'action': 'finish', 'predicted_id': 10, 'reason': 'only shelf above table'},
                                {'action': 'stop', 'reason': 'physical tall remains unknown'}])
                def chat(*args, **kwargs):
                    return {'_content': json.dumps(next(actions)), '_baseline': {'latency_s': .01},
                            '_finish_reason': 'stop', 'usage': {}}
                with tempfile.TemporaryDirectory() as directory, \
                        patch.dict('os.environ', {'EVIGROUND_REACT_BACKEND': 'qwen'}), \
                        patch('eviground.grounding.active_diagnostic.chat', side_effect=chat):
                    result = active_ground(query, parsed, objects, objects, objects[:2], xyz,
                        np.zeros_like(xyz, dtype=np.uint8), ids, Path(directory), [], {}, limit, True)
                self.assertFalse(result['accepted'])
                self.assertTrue(result['diagnosis']['pending_size_conditions'])

    def test_explicit_color_subset_stays_frozen_if_members_change_or_become_unknown(self):
        query = 'The largest of the green trash cans.'
        parsed = validate_structure(query, {'Target': ['trash can'], 'Anchor': [], 'Relations': [],
            'Attributes': [{'category': 'trash can', 'role': 'target', 'phrase': 'largest'},
                           {'category': 'trash can', 'role': 'target', 'phrase': 'green'}],
            'Observer': {}, 'Unhandled': []})
        objects = [bookshelf(10, 0., 1., 'trash can'), bookshelf(11, 2., 2., 'trash can'),
                   bookshelf(12, 4., 3., 'trash can')]
        objects[1]['bbox_3d'][3:5] = [.8, .8]
        objects[2]['bbox_3d'][3:5] = [1.2, 1.2]
        checks = [{'id': oid, 'category': 'trash can', 'role': 'target',
                   'category_verdict': 'yes', 'verdict': verdict, 'visibility': 'visible',
                   'reason': 'visible surface', 'attribute_checks': [
                       {'attribute': 'green', 'verdict': verdict, 'visibility': 'visible',
                        'reason': 'visible surface color'}]}
                  for oid, verdict in [(10, 'yes'), (11, 'yes'), (12, 'no')]]
        spec = size_conditions(parsed, 'trash can')[0]
        pool, scope = size_condition_scope(parsed, objects, 'trash can', spec, checks)
        self.assertEqual([obj['bbox_id'] for obj in pool], [10, 11])
        self.assertEqual(scope['status'], 'bound')
        xyz, ids = cloud_for(objects)
        for changed_verdict in ('uncertain', 'no'):
            changed = copy.deepcopy(checks)
            changed[1]['attribute_checks'][0]['verdict'] = changed_verdict
            changed[1]['verdict'] = changed_verdict
            pool, scope = size_condition_scope(parsed, objects, 'trash can', spec, changed)
            self.assertEqual([obj['bbox_id'] for obj in pool], [10, 11])
            self.assertEqual(scope['status'], 'unknown')
            self.assertIn('membership changed or became unknown', scope['reason'])
            diagnosis = diagnose(parsed, objects, objects, [0., 0., 1.], changed,
                trusted_categories=True, apply_size_constraints=False)
            diagnosis['size_comparison_pool_ids'] = [10, 11, 12]
            observation = compute_observation('size', query, parsed, diagnosis, objects,
                                              objects, xyz, ids)
            constraint = observation['size_constraints'][0]
            self.assertEqual(constraint['comparison_pool_ids'], [10, 11])
            self.assertEqual(constraint['scope_status'], 'unknown')
            self.assertEqual(constraint['selected_ids'], [])
            applied = apply_size_observation(copy.deepcopy(diagnosis), observation)
            self.assertNotEqual(applied['status'], 'resolved')
            self.assertTrue(applied['pending_size_conditions'])

    def test_large_green_does_not_compare_only_the_remaining_green_member(self):
        query = 'The large green trash can.'
        parsed = validate_structure(query, {'Target': ['trash can'], 'Anchor': [], 'Relations': [],
            'Attributes': [{'category': 'trash can', 'phrase': 'large green'}],
            'Observer': {}, 'Unhandled': []})
        objects = [bookshelf(10, 0., 1., 'trash can'), bookshelf(11, 2., 2., 'trash can')]
        objects[1]['bbox_3d'][3:5] = [.8, .8]
        checks = [{'id': oid, 'category': 'trash can', 'role': 'target',
                   'category_verdict': 'yes', 'verdict': verdict, 'visibility': 'visible',
                   'reason': 'visible surface', 'attribute_checks': [
                       {'attribute': 'green', 'verdict': verdict, 'visibility': 'visible',
                        'reason': 'visible surface color'}]}
                  for oid, verdict in [(10, 'yes'), (11, 'no')]]
        xyz, ids = cloud_for(objects)
        diagnosis = diagnose(parsed, objects, objects, [0., 0., 1.], checks,
                             trusted_categories=True, apply_size_constraints=False)
        self.assertEqual(diagnosis['candidate_ids'], [10])
        diagnosis['size_comparison_pool_ids'] = [10, 11]
        observation = compute_observation('size', query, parsed, diagnosis, objects, objects, xyz, ids)
        self.assertEqual(observation['size_constraints'][0]['comparison_pool_ids'], [10, 11])
        self.assertEqual(observation['candidate_evidence'][0]['status'], 'fail')
        applied = apply_size_observation(copy.deepcopy(diagnosis), observation)
        self.assertEqual(applied['candidate_ids'], [])
        self.assertEqual(applied['status'], 'conflict')


if __name__ == '__main__':
    unittest.main()
