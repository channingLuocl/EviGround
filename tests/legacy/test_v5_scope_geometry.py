"""Offline v5 comparison-scope and measured-structure regressions."""
import os
import unittest
from unittest.mock import patch

import numpy as np

from eviground.grounding.diagnostic import size_condition_scope, size_conditions, validate_structure
from eviground.geometry.object_geometry import _oriented_measurements, tabletop_measurement
from eviground.geometry.size_semantics import evaluate_size_condition
from eviground.geometry.v4_relations import bind_tasks


class V5ScopeGeometryTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {'EVIGROUND_V4': '1', 'EVIGROUND_V5': '1',
            'EVIGROUND_V5_SCOPE': '1', 'EVIGROUND_V5_MEASUREMENTS': '1',
            'EVIGROUND_API_PROVIDER': 'qwen', 'EVIGROUND_REACT_BACKEND': 'qwen'})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_counted_antecedent_is_not_widened_to_whole_category(self):
        query = ('There are two chairs facing the square white objects. '
                 'The target object is the taller chair to the left.')
        parsed = validate_structure(query, {'Target': ['chair'], 'Anchor': ['object'],
            'Attributes': [{'category': 'chair', 'phrase': 'taller'}], 'Relations': [],
            'Observer': {}, 'Unhandled': ['facing the square white objects', 'to the left']})
        spec = size_conditions(parsed, 'chair')[0]
        self.assertEqual(spec['required_member_count'], 2)
        self.assertEqual(spec['scope_kind'], 'relation_subset')
        self.assertIn('facing the square white objects', spec['scope_relation_quotes'])
        objects = [{'bbox_id': oid, 'target': 'chair', 'bbox_3d': [oid, 0, 1, 1, 1, 2]}
                   for oid in range(4)]
        _, binding = size_condition_scope(parsed, objects, 'chair', spec)
        self.assertEqual(binding['status'], 'unknown')
        self.assertNotIn('_size_scope_bindings', {key: value for key, value in parsed.items()
                        if value})

    def test_subset_member_state_is_conjunction_of_all_quoted_predicates(self):
        objects = [{'bbox_id': oid, 'target': 'chair'} for oid in (1, 2, 3)]
        spec = {'word': 'smaller', 'original_quote': 'smaller', 'scope_kind': 'relation_subset',
                'scope_relation_quotes': ['near the door', 'on the floor'], 'required_member_count': 2}
        parsed = {'_v4_relation_evidence': [
            {'candidate_id': oid, 'target_category': 'chair', 'source_quote': quote, 'status': state}
            for quote, states in [('near the door', ['pass', 'unknown', 'pass']),
                                   ('on the floor', ['pass', 'pass', 'fail'])]
            for oid, state in zip((1, 2, 3), states)]}
        _, binding = size_condition_scope(parsed, objects, 'chair', spec)
        self.assertEqual(binding['member_states'], {'1': 'pass', '2': 'unknown', '3': 'fail'})
        self.assertEqual(binding['unknown_ids'], [2])
        self.assertEqual(binding['selected_ids'], [1, 2])
        self.assertEqual(binding['status'], 'unknown')
        self.assertEqual(len(binding['member_evidence']), 6)

    def test_planar_area_keeps_orientation_dimensions_and_thickness(self):
        x, z = np.meshgrid(np.linspace(-.4, .4, 30), np.linspace(0, 1.2, 30))
        points = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
        metrics, quality, details = _oriented_measurements({'target': 'mirror'}, points)
        self.assertAlmostEqual(metrics['body_plane_area_m2'], .96, places=5)
        self.assertAlmostEqual(metrics['body_plane_long_extent_m'], 1.2, places=5)
        self.assertAlmostEqual(metrics['body_plane_short_extent_m'], .8, places=5)
        self.assertEqual(metrics['body_plane_thickness_m'], 0)
        self.assertEqual(details['body_plane']['orientation'], 'vertical')
        self.assertEqual(quality['body_plane_area_m2']['status'], 'usable')

    def test_book_area_volume_disagreement_remains_explicit_and_tentative(self):
        spec = {'word': 'smallest', 'original_quote': 'smallest', 'form': 'superlative',
                'category': 'book', 'interpretation': 'set_comparison', 'metric': 'body_plane_area_m2',
                'metric_reason': 'observed body plane', 'direction': 'min', 'scope_kind': 'full_category',
                'tentative_metric': True}
        rows = [{'id': oid, 'status': 'known', 'body_plane_area_m2': area,
                 'oriented_volume_m3': volume, 'metric_reliability': {
                     key: {'status': 'usable'} for key in ('body_plane_area_m2', 'oriented_volume_m3')}}
                for oid, area, volume in [(1, .1, .01), (2, .2, .005)]]
        result = evaluate_size_condition(spec, rows, [1, 2], scope_bound=True)
        self.assertEqual(result['metric_ambiguity']['proxy_selected_ids'],
                         {'body_plane_area_m2': [1], 'oriented_volume_m3': [2]})
        self.assertEqual(result['tentative_selected_ids'], [1])
        self.assertEqual(result['selected_ids'], [])
        self.assertEqual({p['status'] for p in result['candidate_evidence']}, {'unknown'})

    def test_tabletop_ranking_requires_comparable_supported_local_floors(self):
        spec = {'word': 'taller', 'original_quote': 'taller', 'form': 'comparative',
                'interpretation': 'pair_or_set_comparison', 'metric': 'tabletop_height_m',
                'metric_reason': 'tabletop above local floor', 'direction': 'max',
                'scope_kind': 'full_category'}
        rows = [{'id': oid, 'status': 'known', 'tabletop_height_m': height,
                 'metric_reliability': {'tabletop_height_m': {'status': 'usable'}},
                 'tabletop_evidence': {'status': 'usable', 'local_floor_z_m': floor,
                     'floor_plane': {'status': 'usable', 'normal_xyz': [0, 0, 1],
                                     'point_xyz': [oid, 0, floor]}}}
                for oid, height, floor in [(1, .7, 0), (2, .9, .2)]]
        result = evaluate_size_condition(spec, rows, [1, 2], scope_bound=True)
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['selected_ids'], [])
        self.assertIn('floor', result['measurement_ambiguity']['reason'])

    def test_tabletop_records_actual_labelled_support_region(self):
        x, y = np.meshgrid(np.linspace(-.5, .5, 20), np.linspace(-.5, .5, 20))
        floor = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        top = floor + [0, 0, .75]
        obj = {'bbox_id': 1, 'target': 'table', 'bbox_3d': [0, 0, .75, 1, 1, .1]}
        result = tabletop_measurement(obj, top, [{'bbox_id': 2, 'target': 'floor'}],
                                     floor, np.full(len(floor), 2))
        self.assertEqual(result['tabletop_evidence']['floor_instance_ids'], [2])
        self.assertEqual(result['tabletop_evidence']['floor_plane']['support_region_xy'],
                         {'min': [-.5, -.5], 'max': [.5, .5]})

    def test_all_possible_reference_instances_survive_unknown_appearance_binding(self):
        query = 'the smallest pillow on a bed'
        parsed = {'Target': ['pillow'], 'Anchor': ['bed'], 'Relations': [],
                  'Unhandled': ['on a bed']}
        pillow = {'bbox_id': 1, 'target': 'pillow', 'bbox_3d': [0, 0, 1.05, .3, .3, .1]}
        beds = [{'bbox_id': oid, 'target': 'bed', 'bbox_3d': [x, 0, .5, 2, 2, 1]}
                for oid, x in [(2, 0), (3, 5)]]
        with patch('eviground.grounding.diagnostic._attribute_comparison_pool', return_value=(beds, {'status': 'unknown'})):
            row = bind_tasks(query, parsed, [pillow] + beds, [pillow])[0]
        geometry = row['geometry_evidence']
        self.assertEqual(geometry['anchor_ids'], [2, 3])
        self.assertEqual(len(geometry['anchor_hypotheses']), 2)
        self.assertEqual(geometry['status'], 'unknown')

    def test_unknown_body_plane_member_blocks_ranking_without_deleting_it(self):
        spec = {'word': 'smallest', 'original_quote': 'smallest', 'form': 'superlative',
                'category': 'mirror', 'interpretation': 'set_comparison', 'metric': 'body_plane_area_m2',
                'metric_reason': 'observed body plane', 'direction': 'min', 'scope_kind': 'full_category'}
        rows = [{'id': oid, 'status': 'known', 'body_plane_area_m2': area,
                 'measurement_reliability': 'usable', 'metric_reliability': {
                     'body_plane_area_m2': {'status': quality}}}
                for oid, area, quality in [(1, .1, 'usable'), (2, 1, 'unknown')]]
        result = evaluate_size_condition(spec, rows, [1, 2], scope_bound=True)
        self.assertEqual(result['unknown_measurement_ids'], [2])
        self.assertEqual(result['comparison_pool_ids'], [1, 2])
        self.assertEqual(result['selected_ids'], [])
        self.assertEqual({p['status'] for p in result['candidate_evidence']}, {'unknown'})


if __name__ == '__main__':
    unittest.main()
