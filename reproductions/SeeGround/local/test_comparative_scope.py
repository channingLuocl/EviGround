import json
from itertools import product
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from active_diagnostic import active_ground, size_candidate_evidence
from diagnostic import attributes_for, diagnose, validate_structure
from query_rewrite import full_category_size_comparison, rewrite_query


QUERY = 'The smaller of the cabinets.'


def parsed_query(query=QUERY):
    return {
        'Target': ['cabinet'], 'Anchor': [], 'Relations': [],
        'Observer': {'origin': {}, 'facing': {}},
        'Attributes': [{'category': 'cabinet', 'phrase': 'smaller'}],
        'Unhandled': [query],
    }


def cabinet(oid, width):
    return {'bbox_id': oid, 'target': 'cabinet',
            'bbox_3d': [oid * 2., 0., .5, width, 1., 1.]}


def measurement(obj):
    return {'id': obj['bbox_id'], 'status': 'known',
            'source': 'synthetic complete oriented cabinet measurements',
            'measurement_reliability': 'usable',
            'volume_m3': obj['bbox_3d'][3], 'height_m': 1.,
            'oriented_volume_m3': obj['bbox_3d'][3],
            'oriented_footprint_m2': obj['bbox_3d'][3],
            'metric_reliability': {'oriented_volume_m3': {'status': 'usable'},
                                   'oriented_footprint_m2': {'status': 'usable'},
                                   'height_m': {'status': 'usable'}},
            'horizontal_aspect_ratio': 1.}


class ComparativeScopeTests(unittest.TestCase):
    def test_full_category_scope_preserves_original_quote(self):
        attribute = full_category_size_comparison(QUERY, parsed_query())
        self.assertEqual(attribute, {
            'category': 'cabinet', 'role': 'target', 'phrase': 'smaller',
            'comparison_scope': 'full_category',
            'scope_phrase': QUERY,
        })

    def test_parser_unspecified_observer_does_not_block_category_comparison(self):
        parsed = parsed_query()
        parsed['Observer'] = {
            'origin': {'kind': 'unspecified', 'category': '', 'phrase': ''},
            'facing': {'category': '', 'group': False, 'phrase': ''},
        }
        self.assertIsNotNone(full_category_size_comparison(QUERY, parsed))
        data = validate_structure(QUERY, parsed)
        self.assertEqual(data['Unhandled'], [])
        self.assertEqual(data['diagnostic_missing'], [])
        self.assertEqual(data['Attributes'][0]['comparison_scope'], 'full_category')

    def test_same_category_anchor_is_a_comparison_pool_not_a_reference(self):
        objects = [cabinet(1, 2.), cabinet(2, 1.), cabinet(3, 3.)]
        for anchors in (['cabinet'], ['cabinets'], ['cabinet', 'cabinet']):
            with self.subTest(anchors=anchors):
                raw = parsed_query()
                raw['Anchor'] = anchors
                raw['Attributes'][0]['role'] = 'target'
                data = validate_structure(QUERY, raw)
                self.assertEqual(data['Anchor'], [])
                self.assertEqual(data['Unhandled'], [])
                self.assertEqual(data['diagnostic_missing'], [])
                self.assertEqual(data['Attributes'][0]['comparison_scope'], 'full_category')
                data['query_rewrite'] = rewrite_query(QUERY, data)
                self.assertEqual(data['query_rewrite']['references'], [])
                self.assertEqual(data['query_rewrite']['unresolved'], [])
                diagnosis = diagnose(data, objects, objects, [0., 0., .5], trusted_categories=True,
                                     size_measurements=[measurement(obj) for obj in objects])
                self.assertEqual(diagnosis['status'], 'resolved')
                self.assertEqual(diagnosis['candidate_ids'], [2])

    def test_same_category_anchor_cannot_hide_reference_conditions(self):
        for field, value in (
                ('Anchor', ['cabinet', 'table']), ('Anchor', ['red cabinet']),
                ('Attributes', [{'category': 'cabinet', 'role': 'reference', 'phrase': 'smaller'}]),
                ('Attributes', [{'category': 'table', 'role': 'reference', 'phrase': 'smaller'}]),
                ('Attributes', 17),
                ('Viewpoint', 'facing the cabinets')):
            with self.subTest(field=field, value=value):
                raw = {**parsed_query(), 'Anchor': ['cabinet'], field: value}
                self.assertIsNone(full_category_size_comparison(QUERY, raw))
        for query in ('The smaller of the red cabinets.',
                      'The smaller of the cabinets beside the table.'):
            with self.subTest(query=query):
                raw = {**parsed_query(query), 'Anchor': ['cabinet']}
                self.assertIsNone(full_category_size_comparison(query, raw))

    def test_recovery_rejects_underspecified_or_modified_scope(self):
        for query in ('The smaller of the cabinet.',
                      'The smaller of the red cabinets.',
                      'The smaller of the cabinets beside the table.',
                      'The smaller of the cabinets in the room.'):
            with self.subTest(query=query):
                self.assertIsNone(full_category_size_comparison(query, parsed_query(query)))
        for field, value in (
                ('Target', ['red cabinet']), ('Target', ['cabinet', 'table']),
                ('Anchor', ['table']), ('Relations', [{'relation': 'above'}]),
                ('Observer', {'origin': {'kind': 'room_center',
                                         'phrase': 'in the middle of the room'}})):
            with self.subTest(field=field):
                self.assertIsNone(full_category_size_comparison(
                    QUERY, {**parsed_query(), field: value}))

    def test_validation_and_rewrite_keep_scope_separate_from_original_phrase(self):
        data = validate_structure(QUERY, parsed_query())
        self.assertEqual(data['Unhandled'], [])
        self.assertEqual(data['diagnostic_missing'], [])
        self.assertTrue(any(record['source_quote'] == 'smaller'
                            for record in data['condition_records']))
        self.assertEqual(attributes_for(data, 'cabinet', role='target'), ['smaller'])
        self.assertEqual(attributes_for(data, 'cabinet', role='target', geometry=True), ['smaller'])
        rewritten = rewrite_query(QUERY, data)
        constraint, = rewritten['constraints']
        self.assertEqual(constraint['type'], 'size_min')
        self.assertEqual(constraint['phrase'], 'smaller')
        self.assertEqual(constraint['comparison_scope'], 'full_category')
        self.assertNotIn('geometry_attribute', constraint)
        self.assertEqual(constraint['size_conditions'][0]['form'], 'comparative')
        self.assertEqual(constraint['scope_phrase'], QUERY)
        self.assertEqual(rewritten['unresolved'], [])

    def test_baseline_selects_minimum_over_the_full_category(self):
        data = validate_structure(QUERY, parsed_query())
        data['query_rewrite'] = rewrite_query(QUERY, data)
        objects = [cabinet(1, 2.), cabinet(2, 1.), cabinet(3, 3.)]
        diagnosis = diagnose(data, objects, objects, [0., 0., .5], trusted_categories=True,
                             size_measurements=[measurement(obj) for obj in objects])
        self.assertEqual(diagnosis['status'], 'resolved')
        self.assertEqual(diagnosis['candidate_ids'], [2])
        self.assertEqual(diagnosis['missing'], [])

    def test_active_cannot_redefine_comparison_after_candidates_shrink(self):
        data = validate_structure(QUERY, parsed_query())
        objects = [cabinet(1, 2.), cabinet(2, 1.), cabinet(3, .5)]
        evidence, constraints, decision = size_candidate_evidence(
            data, objects[:2], [measurement(obj) for obj in objects],
            comparison_pool_ids=[1, 2, 3], comparison_scope_bound=True)
        self.assertEqual(decision['comparison_pool_ids'], [1, 2, 3])
        self.assertEqual(constraints[0]['attribute'], 'smaller')
        self.assertEqual(constraints[0]['selected_ids'], [3])
        self.assertEqual([row['status'] for row in evidence], ['fail', 'fail'])

    def test_tied_or_unknown_measurements_cannot_certify_one_winner(self):
        data = validate_structure(QUERY, parsed_query())
        objects = [cabinet(1, 1.), cabinet(2, 1.)]
        diagnosis = diagnose(data, objects, objects, [0., 0., .5], trusted_categories=True,
                             size_measurements=[measurement(obj) for obj in objects])
        self.assertEqual(diagnosis['status'], 'missing')
        self.assertEqual(diagnosis['candidate_ids'], [1, 2])
        evidence, _, decision = size_candidate_evidence(
            data, objects, [measurement(objects[0])], comparison_pool_ids=[1, 2],
            comparison_scope_bound=True)
        self.assertEqual(decision['unknown_measurement_ids'], [2])
        self.assertEqual([row['status'] for row in evidence], ['unknown', 'unknown'])

    def test_bare_comparative_and_two_scope_use_reliable_pair(self):
        objects = [cabinet(1, 2.), cabinet(2, 1.)]
        for query in ('smaller cabinet', 'The smaller of the two cabinets.'):
            with self.subTest(query=query):
                data = validate_structure(query, parsed_query(query))
                self.assertNotIn('geometry_attribute', data['Attributes'][0])
                evidence, constraints, _ = size_candidate_evidence(
                    data, objects, [measurement(obj) for obj in objects],
                    comparison_scope_bound=True)
                self.assertEqual(constraints[0]['attribute'], 'smaller')
                self.assertEqual([row['status'] for row in evidence], ['fail', 'pass'])
                diagnosis = diagnose(data, objects, objects, [0., 0., .5], trusted_categories=True,
                                     size_measurements=[measurement(obj) for obj in objects])
                self.assertEqual(diagnosis['status'], 'resolved')
                self.assertEqual(diagnosis['candidate_ids'], [2])

    def test_two_scope_with_additional_or_unknown_member_stays_missing(self):
        query = 'The smaller of the two cabinets.'
        data = validate_structure(query, parsed_query(query))
        objects = [cabinet(1, 2.), cabinet(2, 1.), cabinet(3, .5)]
        for rows, pool in (([measurement(obj) for obj in objects], [1, 2, 3]),
                           ([measurement(objects[0])], [1, 2])):
            evidence, _, _ = size_candidate_evidence(data, objects[:2], rows,
                comparison_pool_ids=pool, comparison_scope_bound=True)
            self.assertTrue(all(row['status'] == 'unknown' for row in evidence))
        diagnosis = diagnose(data, objects, objects, [0., 0., .5], trusted_categories=True,
                             size_measurements=[measurement(obj) for obj in objects])
        self.assertEqual(diagnosis['status'], 'missing')
        self.assertEqual(diagnosis['candidate_ids'], [1, 2, 3])

    def test_bbox_only_diagnosis_does_not_certify_new_overall_comparative(self):
        data = validate_structure(QUERY, parsed_query())
        objects = [cabinet(1, 2.), cabinet(2, 1.)]
        diagnosis = diagnose(data, objects, objects, [0., 0., .5], trusted_categories=True)
        self.assertEqual(diagnosis['status'], 'missing')
        self.assertEqual(diagnosis['candidate_ids'], [1, 2])

    def test_active_entry_requires_size_execution_before_acceptance(self):
        raw = parsed_query()
        raw['Anchor'] = ['cabinet']
        raw['Attributes'][0]['role'] = 'target'
        data = validate_structure(QUERY, raw)
        objects = [cabinet(1, 2.), cabinet(2, 1.)]
        points, ids = [], []
        for obj in objects:
            center = np.asarray(obj['bbox_3d'][:3])
            half = np.asarray(obj['bbox_3d'][3:]) / 2.
            corners = [center + half * np.asarray(signs)
                       for signs in product((-1, 1), repeat=3)]
            points.extend(corners)
            ids.extend([obj['bbox_id']] * len(corners))
        xyz = np.asarray(points)
        for compute in (False, True):
            actions = ([{'action': 'evidence', 'dimension': 'size', 'reason': 'check full category'}]
                       if compute else [])
            actions.extend([
                {'action': 'finish', 'predicted_id': 2, 'reason': 'small cabinet'},
                {'action': 'stop', 'reason': 'size remains unverified'},
            ])
            responses = iter(actions)
            with self.subTest(compute=compute), tempfile.TemporaryDirectory() as directory, patch.dict(
                    'os.environ', {'SEEGROUND_REACT_BACKEND': 'qwen'}), patch(
                    'active_diagnostic.chat', side_effect=lambda *args, **kwargs: {
                    '_content': json.dumps(next(responses, {
                        'action': 'stop', 'reason': 'size has still not been executed'})), '_finish_reason': 'stop',
                        '_baseline': {'model': 'mock', 'latency_s': .01}, 'usage': {}}), patch(
                    'active_diagnostic.render_action', return_value={
                        'tool': 'render', 'image': 'mock.png', 'focus_ids': [1, 2]}), patch(
                    'active_diagnostic.image_part', return_value={'type': 'text', 'text': 'image'}):
                result = active_ground(QUERY, data, objects, objects, objects, xyz,
                                       np.zeros_like(xyz, dtype=np.uint8), np.asarray(ids),
                                       Path(directory), [], {}, 3, True)
            self.assertEqual(result['accepted'], compute)
            self.assertEqual(bool(result['diagnosis']['pending_size_conditions']), not compute)
            if compute:
                self.assertEqual(result['predicted_id'], 2)


if __name__ == '__main__':
    unittest.main()
