"""v4 geometry, evidence binding and bounded-controller regressions (offline)."""
import itertools
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from active_diagnostic import active_ground, compute_observation, evidence_request, size_candidate_evidence, execute_evidence
from diagnostic import size_conditions, size_condition_scope, validate_structure, diagnose
from object_geometry import point_cloud_measurements
from query_syntax import analyze_query
from react_decision import jev_state, text_state
from size_semantics import evaluate_size_condition, size_specs
from v4_policy import evidence_inventory, deletion_events, backtrack_events
from v4_relations import geometric_relation, relation_tasks
from v4_visual import boxed_view, normalize_report


def box(oid, category='box', center=(0, 0, .5), extent=(1, 1, 1)):
    return {'bbox_id': oid, 'target': category, 'bbox_3d': list(center) + list(extent)}


def cloud(obj, surfaces=False):
    axes = [np.linspace(-d / 2, d / 2, 8) for d in obj['bbox_3d'][3:]]
    points = np.asarray(list(itertools.product(*axes))) + obj['bbox_3d'][:3]
    if surfaces:
        points = points[points[:, 2] >= np.quantile(points[:, 2], .85)]
    return points


def parsed(category, word, query=None, **fields):
    data = {'Target': [category], 'Anchor': [], 'Relations': [], 'Attributes': [
        {'category': category, 'role': 'target', 'phrase': word}], 'Observer': {}, 'Unhandled': []}
    data.update(fields)
    return validate_structure(query or f'the {word} {category}', data)


def measure(objects):
    xyz = np.concatenate([cloud(o) for o in objects])
    ids = np.concatenate([np.full(len(cloud(o)), o['bbox_id']) for o in objects])
    return point_cloud_measurements(objects, xyz, ids), xyz, ids


class V4Tests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'SEEGROUND_V4': '1', 'SEEGROUND_REACT_BACKEND': 'qwen',
                         'SEEGROUND_API_PROVIDER': 'qwen', 'SEEGROUND_V5': '0',
                         'SEEGROUND_SYNTAX_BACKEND': 'spacy', 'SEEGROUND_V4_DYNAMIC_THRESHOLD': '0'})
        env.start()
        self.addCleanup(env.stop)

    def test_inventory_preserves_quote_count_and_dynamic_dependencies(self):
        query = 'the taller of the two chairs near the door'
        p = parsed('chair', 'taller', query, Anchor=['door'], Relations=[{
            'target_category': 'chair', 'anchor_category': 'door', 'relation': 'near', 'phrase': 'near the door'}])
        inventory = evidence_inventory(query, p)
        size = next(r for r in inventory['items'] if r.get('member_count') == 2)
        self.assertIn('bind comparison members', size['dependencies'])
        self.assertIn('near the door', size['dependencies'])
        self.assertNotIn('schedule', inventory)

    def test_independent_relation_does_not_restrict_pool(self):
        query = 'the larger window and above the door'
        p = parsed('window', 'larger', query, Anchor=['door'], Relations=[{
            'target_category': 'window', 'anchor_category': 'door', 'relation': 'above',
            'phrase': 'above the door', 'scope': 'relative_anchor'}])
        self.assertEqual(size_conditions(p, 'window')[0]['scope_kind'], 'full_category')

    def test_recovered_room_scope_remains_full_category(self):
        for category, word in (('picture', 'largest'), ('shelf', 'tallest')):
            p = parsed(category, word, f'The {word} {category} in the room', Unhandled=['in the room'])
            spec = size_conditions(p, category)[0]
            self.assertEqual(spec['scope_kind'], 'full_category')
            self.assertNotIn('in the room', p['Unhandled'])

    def test_ambiguous_relation_is_not_expanded_to_full_pool(self):
        p = parsed('chair', 'taller', 'the taller chair near the door', Anchor=['door'])
        spec = size_conditions(p, 'chair')[0]
        self.assertEqual(spec['scope_kind'], 'unbound_explicit_scope')
        _, binding = size_condition_scope(p, [box(1, 'chair'), box(2, 'chair')], 'chair', spec)
        self.assertEqual(binding['status'], 'unknown')

    def subset(self):
        query = 'the taller of the two chairs near the door'
        p = parsed('chair', 'taller', query, Anchor=['door'], Relations=[{
            'target_category': 'chair', 'anchor_category': 'door', 'relation': 'near', 'phrase': 'near the door'}])
        objects = [box(i, 'chair', (i, 0, .5)) for i in (1, 2, 3)] + [box(4, 'door')]
        p['_v4_relation_evidence'] = [{'candidate_id': i, 'source_quote': 'near the door',
            'target_category': 'chair', 'status': state} for i, state in ((1, 'pass'), (2, 'pass'), (3, 'fail'))]
        return p, objects, size_conditions(p, 'chair')[0]

    def test_quoted_two_member_subset_binds_from_relation_evidence(self):
        p, objects, spec = self.subset()
        pool, binding = size_condition_scope(p, objects, 'chair', spec)
        self.assertEqual(binding['status'], 'bound')
        self.assertEqual([o['bbox_id'] for o in pool], [1, 2])

    def test_subset_unknown_member_blocks_count_and_ranking(self):
        p, objects, spec = self.subset()
        p['_v4_relation_evidence'][-1]['status'] = 'unknown'
        _, binding = size_condition_scope(p, objects, 'chair', spec)
        self.assertEqual(binding['status'], 'unknown')
        self.assertIn(3, binding['unknown_ids'])

    def test_frozen_subset_change_retains_old_pool_and_invalidates_proof(self):
        p, objects, spec = self.subset()
        size_condition_scope(p, objects, 'chair', spec)
        p['_v4_relation_evidence'][1]['status'] = 'fail'
        p['_v4_relation_evidence'][2]['status'] = 'pass'
        pool, binding = size_condition_scope(p, objects, 'chair', spec)
        self.assertEqual([o['bbox_id'] for o in pool], [1, 2])
        self.assertEqual(binding['status'], 'unknown')
        self.assertEqual(binding['proposed_binding']['selected_ids'], [1, 3])

    def test_facing_subset_of_two_never_becomes_four_chairs(self):
        p = parsed('chair', 'smaller', 'the smaller of the two chairs facing the square white objects',
                   Anchor=['object'], Unhandled=['facing the square white objects'])
        spec = size_conditions(p, 'chair')[0]
        _, binding = size_condition_scope(p, [box(i, 'chair') for i in range(4)], 'chair', spec)
        self.assertEqual(binding['status'], 'unknown')
        self.assertEqual(spec['required_member_count'], 2)

    def test_planar_scan_risk_does_not_assert_incomplete_or_certify_volume(self):
        objects = [box(19, extent=(1, 1, .06)), box(21, ( 'box'), extent=(.5, .5, .34))]
        # Stable, almost horizontal sheet; points alone cannot distinguish a real flat box.
        points = np.asarray(list(itertools.product(np.linspace(0, 1, 10), np.linspace(0, 1, 10), [0., .001])))
        xyz = np.concatenate([points, cloud(objects[1])])
        ids = np.concatenate([np.full(len(points), 19), np.full(len(cloud(objects[1])), 21)])
        rows = point_cloud_measurements(objects, xyz, ids)
        risk = rows[0]['observed_completeness']
        self.assertTrue(risk['surface_only_risk'])
        self.assertEqual(risk['status'], 'unknown')
        spec = size_conditions(parsed('box', 'smaller'), 'box')[0]
        result = evaluate_size_condition(spec, rows, [19, 21], scope_bound=True)
        self.assertEqual(result['selected_ids'], [])
        self.assertIn(19, result['unknown_measurement_ids'])

    def test_axis_specific_quality_preserves_footprint_when_height_unstable(self):
        o = box(18, 'table', extent=(2, 5, 1))
        xyz = cloud(o)
        xyz = np.concatenate([xyz, [[0, 0, 2.5]]])
        row = point_cloud_measurements([o], xyz, np.full(len(xyz), 18))[0]
        self.assertEqual(row['metric_reliability']['height_m']['status'], 'unknown')
        self.assertEqual(row['metric_reliability']['oriented_footprint_m2']['status'], 'usable')
        self.assertIsNone(row['oriented_volume_m3'])
        self.assertGreater(row['oriented_footprint_m2'], 9)
        self.assertIn('annotated_box_metrics', row)
        self.assertIn('not independent', row['provenance']['independence'])

    def test_table_area_proxy_is_tentative_and_never_deletes(self):
        objects = [box(17, 'table'), box(18, 'table', (2, 0, .5), (2, 5, 1))]
        rows, _, _ = measure(objects)
        p = parsed('table', 'smaller')
        rows[1]['oriented_volume_m3'] = None
        rows[1]['metric_reliability']['oriented_volume_m3']['status'] = 'unknown'
        evidence, constraints, decision = size_candidate_evidence(p, objects, rows, objects=objects)
        self.assertEqual(constraints[0]['metric'], 'oriented_footprint_m2')
        self.assertEqual(decision['tentative_selected_ids'], [17])
        self.assertEqual({r['status'] for r in evidence}, {'unknown'})

    def test_reliable_table_volume_with_agreeing_area_keeps_original_certification(self):
        objects = [box(1, 'table'), box(2, 'table', (2, 0, .5), (2, 2, 1))]
        rows, _, _ = measure(objects)
        evidence, constraints, _ = size_candidate_evidence(parsed('table', 'smaller'), objects, rows, objects=objects)
        self.assertEqual(constraints[0]['metric'], 'oriented_volume_m3')
        self.assertEqual(constraints[0]['selected_ids'], [1])
        self.assertEqual([r['status'] for r in evidence], ['pass', 'fail'])

    def test_proxy_disagreement_is_exposed_without_permanent_veto(self):
        objects = [box(10, 'trash can', extent=(.5, .5, 2)), box(25, 'trash can', (2, 0, .5), (1, 1, .4))]
        rows, _, _ = measure(objects)
        p = parsed('trash can', 'bigger')
        _, constraints, decision = size_candidate_evidence(p, objects, rows, objects=objects)
        self.assertIn('metric_ambiguity', constraints[0])
        self.assertEqual(decision['tentative_selected_ids'], [10])
        self.assertEqual(constraints[0]['selected_ids'], [])

    def test_longer_shorter_same_bookshelf_is_two_axes(self):
        p = parsed('bookshelf', 'longer', 'the longer and shorter bookshelf', Attributes=[
            {'category': 'bookshelf', 'role': 'target', 'phrase': 'longer'},
            {'category': 'bookshelf', 'role': 'target', 'phrase': 'shorter'}])
        metrics = {s['word']: s['metric'] for s in size_conditions(p, 'bookshelf')}
        self.assertEqual(metrics['longer'], 'horizontal_oriented_long_extent_m')
        self.assertEqual(metrics['shorter'], 'height_m')

    def test_tabletop_requires_labelled_local_floor_not_minimum_scene_z(self):
        o = box(10, 'desk')
        rows, _, _ = measure([o])
        self.assertIsNone(rows[0]['tabletop_height_m'])
        spec = size_conditions(parsed('desk', 'taller'), 'desk')[0]
        self.assertEqual(spec['metric'], 'tabletop_height_m')

    def test_tabletop_plane_and_local_floor_are_measured_separately(self):
        desk, floor = box(10, 'desk'), box(99, 'floor', (0, 0, 0), (3, 3, .001))
        top = np.asarray(list(itertools.product(np.linspace(-.5, .5, 20), np.linspace(-.5, .5, 20), [1.])))
        legs = cloud(desk)[::8]
        ground = np.asarray(list(itertools.product(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20), [0.])))
        xyz = np.concatenate([top, legs, ground])
        ids = np.concatenate([np.full(len(top) + len(legs), 10), np.full(len(ground), 99)])
        row = point_cloud_measurements([desk], xyz, ids, scene_objects=[desk, floor])[0]
        self.assertAlmostEqual(row['tabletop_height_m'], 1.)
        self.assertEqual(row['tabletop_evidence']['floor_instance_ids'], [99])

    def test_near_is_absolute_support_band_not_nearest(self):
        anchor = box(3, 'door')
        task = {'kind': 'near'}
        self.assertEqual(geometric_relation(task, box(1, center=(.1, 0, .5)), [anchor])['status'], 'pass')
        self.assertEqual(geometric_relation(task, box(2, center=(.2, 0, .5)), [anchor])['status'], 'pass')
        self.assertEqual(geometric_relation(task, box(2, center=(2, 0, .5)), [anchor])['status'], 'unknown')
        self.assertEqual(geometric_relation(task, box(2), [anchor, box(4, 'door')])['status'], 'unknown')

    def test_support_and_corner_need_geometric_anchors(self):
        support = geometric_relation({'kind': 'support'}, box(1, center=(0, 0, 1.5)), [box(2, 'table')])
        self.assertEqual(support['status'], 'pass')
        self.assertEqual(geometric_relation({'kind': 'corner'}, box(1), [])['status'], 'unknown')

    def test_boxed_render_has_common_camera_uniform_color_and_visibility(self):
        objects = [box(1), box(2, center=(2, 0, .5))]
        rows, xyz, ids = measure(objects)
        with tempfile.TemporaryDirectory() as out:
            view = boxed_view(objects, [1, 2], xyz, np.full(xyz.shape, 100, dtype=np.uint8), ids, out, 'boxes', rows)
            self.assertTrue(Path(view['image']).exists())
            self.assertEqual(view['box_color'], [25, 105, 180])
            self.assertTrue(all(v > 20 for v in view['visible_instance_pixels'].values()))
            self.assertNotIn('target_id', view)
            self.assertNotIn('image', text_state(view))

    def test_report_blocks_unquoted_hidden_and_bare_opinion(self):
        view = {'focus_ids': [1, 2], 'visible_instance_pixels': {'1': 100, '2': 0}}
        report = normalize_report({'status': 'contradiction', 'contradictions': [
            {'candidate_ids': [1, 2], 'source_quote': 'smaller', 'evidence_type': 'visible_geometry', 'reason': 'hidden'},
            {'candidate_ids': [1], 'source_quote': 'smaller', 'evidence_type': 'opinion', 'reason': 'prefer 2'},
            {'candidate_ids': [1], 'source_quote': 'blue', 'evidence_type': 'measurement', 'reason': 'not in query'}]}, 'smaller box', view)
        self.assertFalse(report['supported'])
        self.assertEqual(report['status'], 'unknown')

    def test_completeness_support_requires_visible_side_and_bottom(self):
        view = {'focus_ids': [1], 'visible_instance_pixels': {'1': 100}}
        report = normalize_report({'observations': [{'id': 1, 'completeness': 'supported',
                                  'visible_surfaces': ['top'], 'reason': 'top visible'}]}, 'box', view)
        self.assertEqual(report['observations'][0]['completeness'], 'unknown')

    def test_backtrack_selection_shared_once_and_targeted(self):
        events = deletion_events([1, 2, 3], [3], 'size', [
            {'candidate_id': 1, 'source_quote': 'smaller'}, {'candidate_id': 2, 'source_quote': 'smaller'}], {}, 0)
        self.assertEqual([e['candidate_id'] for e in backtrack_events(events, {'supported': True, 'candidate_ids': [1]})], [1])
        self.assertEqual(backtrack_events(events, {'supported': True, 'candidate_ids': [1]}, used=True), [])
        self.assertEqual(backtrack_events(events, {'supported': False, 'candidate_ids': [1]}), [])

    def controller(self, actions, reports, diagnosis=None, operations=None, limit=2):
        objects = [box(1, 'chair'), box(2, 'chair', center=(2, 0, .5))]
        _, xyz, ids = measure(objects)
        p = parsed('chair', 'tall')
        initial = diagnosis or {'candidate_ids': [1], 'stable_candidate_ids': [1], 'status': 'missing',
            'missing': ['unverified height'], 'geometry': {}, 'pending_size_conditions': [],
            'hypotheses': [{'candidate_ids': [1]}], 'candidate_removal_audit': [{
                'candidate_id': 2, 'source_quote': 'tall', 'status': 'fail', 'measurement_source': 'initial'}]}
        action_iter, report_iter = iter(actions), iter(reports)
        def mocked_chat(messages, **kwargs):
            visual = messages[0]['content'].startswith('Inspect the original query')
            content = next(report_iter if visual else action_iter)
            return {'_content': json.dumps(content), '_finish_reason': 'stop', '_baseline': {}, 'usage': {}}
        with tempfile.TemporaryDirectory() as out, patch('active_diagnostic.chat', side_effect=mocked_chat), patch(
                'active_diagnostic.diagnose', side_effect=lambda *a, **k: json.loads(json.dumps(initial))):
            if operations is None:
                return active_ground('tall chair', p, objects, objects, objects, xyz,
                    np.full(xyz.shape, 100, dtype=np.uint8), ids, Path(out), [], {}, limit, True)
            with patch('active_diagnostic.execute_evidence', side_effect=operations):
                return active_ground('tall chair', p, objects, objects, objects, xyz,
                    np.full(xyz.shape, 100, dtype=np.uint8), ids, Path(out), [], {}, limit, True)

    def test_final_review_only_after_prediction_and_pass_never_certifies_fallback(self):
        result = self.controller([{'action': 'stop', 'reason': 'missing'}], [{'status': 'consistent', 'reason': 'visible'}])
        self.assertEqual(result['predicted_id'], 1)
        self.assertEqual(result['final_review']['status'], 'consistent')
        self.assertFalse(result['accepted'])
        self.assertEqual(result['internal_budget_usage']['review'], 1)

    def test_no_prediction_has_no_final_review(self):
        diagnosis = {'candidate_ids': [1, 2], 'status': 'missing', 'missing': ['unknown'],
                     'stable_candidate_ids': [], 'geometry': {}, 'hypotheses': [], 'pending_size_conditions': []}
        result = self.controller([{'action': 'stop', 'reason': 'missing'}], [], diagnosis)
        self.assertIsNone(result['predicted_id'])
        self.assertEqual(result['internal_budget_usage']['review'], 0)

    def test_final_review_restores_deleted_alternative_and_returns_to_planner(self):
        contradiction = {'status': 'contradiction', 'reason': 'height not represented', 'contradictions': [{
            'candidate_ids': [2], 'source_quote': 'tall', 'evidence_type': 'visible_geometry', 'reason': 'deleted chair visibly tall'}]}
        result = self.controller([{'action': 'stop', 'reason': 'missing'}, {'action': 'stop', 'reason': 'need evidence'}], [contradiction])
        self.assertEqual(result['backtrack_count'], 1)
        self.assertEqual(result['active_trace']['final_candidate_ids'], [1, 2])
        self.assertIsNone(result['predicted_id'])
        self.assertEqual(result['active_trace']['backtracks'][0]['restored_ids'], [2])

    def test_review_contradiction_without_deletion_cannot_change_answer(self):
        contradiction = {'status': 'contradiction', 'contradictions': [{
            'candidate_ids': [1], 'source_quote': 'tall', 'evidence_type': 'visible_geometry', 'reason': 'selected object is short'}]}
        diagnosis = {'candidate_ids': [1], 'stable_candidate_ids': [1], 'status': 'missing',
                     'missing': ['unknown'], 'geometry': {}, 'hypotheses': [], 'pending_size_conditions': []}
        result = self.controller([{'action': 'stop', 'reason': 'missing'}], [contradiction], diagnosis)
        self.assertEqual(result['predicted_id'], 1)
        self.assertEqual(result['backtrack_count'], 0)
        self.assertFalse(result['accepted'])

    def test_relaxation_is_tentative_bound_once_and_cannot_select_wrong_table_height(self):
        objects = [box(1, 'box', extent=(1, 1, 1)), box(2, 'box', (2, 0, .5), (1.02, 1.02, 1.04))]
        rows, _, _ = measure(objects)
        spec = {**size_specs('bigger', 'box')[0], 'metric': 'oriented_volume_m3', 'relax_once': True}
        result = evaluate_size_condition(spec, rows, [1, 2], scope_bound=True)
        self.assertEqual(result['tentative_selected_ids'], [2])
        self.assertEqual(result['selected_ids'], [])
        result = evaluate_size_condition({**spec, 'metric': 'height_m'}, rows, [1, 2], scope_bound=True)
        self.assertEqual(result['selected_ids'], [])
        self.assertNotIn('dynamic_threshold', result)
        result = evaluate_size_condition(spec, rows, [1, 2], scope_bound=False)
        self.assertNotIn('dynamic_threshold', result)

    def test_jev_receives_structured_visual_observations_and_inventory(self):
        state = jev_state({'evidence_inventory': {'items': []}, 'observation': {'details': {
            'visual_inspection': {'observations': [{'id': 1, 'reason': 'surface-only'}]}, 'relation_tasks': []}}})
        self.assertIn('evidence_inventory', state)
        self.assertIn('visual_inspection', state['observation']['details'])

    def test_inspection_request_rejects_invalid_ids_and_camera(self):
        objects = [box(1), box(2)]
        for changes in ({'focus_ids': [999]}, {'inspect': 'yes'}, {'azimuth_deg': float('nan')}):
            with self.assertRaises(ValueError):
                evidence_request({'action': 'evidence', 'dimension': 'size', 'reason': 'inspect', **changes}, [1, 2], objects)

    def operation(self, dimension, fail_ids):
        return {'dimension': dimension, 'details': {}, 'images': [], 'checks': [], 'internal_steps': [],
                'next_evidence': [], 'candidate_evidence': [{'candidate_id': oid, 'status': 'fail',
                 'source_quote': 'tall', 'target_role': 'target', 'role_binding_status': 'bound',
                 'comparison_scope_ids': [1, 2], 'comparison_scope_status': 'bound',
                 'measurement_source': 'new bound geometry', 'explicit_fail_criterion': 'contradicted tall'}
                for oid in fail_ids]}

    def test_reprediction_is_reviewed_again_without_second_backtrack(self):
        contradiction = {'status': 'contradiction', 'contradictions': [{
            'candidate_ids': [2], 'source_quote': 'tall', 'evidence_type': 'visible_geometry', 'reason': 'deleted alternative is tall'}]}
        second = {'status': 'contradiction', 'contradictions': [{
            'candidate_ids': [1], 'source_quote': 'tall', 'evidence_type': 'visible_geometry', 'reason': 'new deletion conflicts'}]}
        result = self.controller([{'action': 'stop', 'reason': 'missing'},
            {'action': 'evidence', 'dimension': 'position', 'reason': 'fresh bound evidence'},
            {'action': 'stop', 'reason': 'remaining unknown'}], [contradiction, second],
            operations=[self.operation('position', [1])])
        self.assertEqual(result['predicted_id'], 2)
        self.assertEqual(result['backtrack_count'], 1)
        self.assertEqual(result['internal_budget_usage']['review'], 2)
        self.assertFalse(result['accepted'])
        self.assertEqual(len(result['active_trace']['final_reviews']), 2)

    def test_empty_pool_runtime_recovery_uses_same_single_opportunity(self):
        diagnosis = {'candidate_ids': [1, 2], 'stable_candidate_ids': [], 'status': 'missing',
                     'missing': ['unknown'], 'geometry': {}, 'hypotheses': [], 'pending_size_conditions': []}
        result = self.controller([{'action': 'evidence', 'dimension': 'position', 'reason': 'check'},
            {'action': 'stop', 'reason': 'unresolved conflict'}], [], diagnosis,
            operations=[self.operation('position', [1, 2])])
        self.assertEqual(result['backtrack_count'], 1)
        self.assertEqual(result['active_trace']['final_candidate_ids'], [1, 2])
        self.assertIsNone(result['predicted_id'])

    def test_size_internal_visual_report_reaches_text_planner_and_is_budgeted(self):
        objects = [box(17, 'table'), box(18, 'table', (2, 0, .5), (2, 2, 1))]
        _, xyz, ids = measure(objects)
        p = parsed('table', 'smaller')
        diagnosis = {'candidate_ids': [17, 18], 'size_comparison_pool_ids': [17, 18], 'geometry': {}}
        report = {'status': 'unknown', 'reason': 'unscanned underside', 'observations': [{
            'id': 17, 'completeness': 'unknown', 'reason': 'top visible', 'visible_surfaces': ['top']}],
            'metric_choices': [{'source_quote': 'smaller', 'category': 'table', 'word': 'smaller',
                               'metric': 'oriented_footprint_m2', 'reason': 'broad table surface'}]}
        calls = []
        with tempfile.TemporaryDirectory() as out, patch('active_diagnostic.chat', return_value={
            '_content': json.dumps(report), '_finish_reason': 'stop', '_baseline': {}, 'usage': {}}):
            result = execute_evidence({'action': 'evidence', 'dimension': 'size', 'reason': 'measure', 'inspect': True},
                'the smaller table', p, diagnosis, objects, objects, xyz, np.full(xyz.shape, 100, dtype=np.uint8),
                ids, Path(out), 1, calls)
        self.assertEqual(p['_v4_budget']['size'], 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['details']['visual_inspection']['reason'], 'unscanned underside')
        self.assertEqual({r['status'] for r in result['candidate_evidence']}, {'unknown'})
        self.assertIn('visual_inspection', text_state(result)['details'])

    def test_near_visual_and_geometry_agree_before_exclusion_and_subset_binding(self):
        targets = [box(1, 'chair', (-.4, 0, .5), (.2, .2, 1)),
                   box(2, 'chair', (-2, 0, .5), (.2, .2, 1))]
        objects = targets + [box(3, 'door', (0, 0, 1), (.5, .1, 2))]
        _, xyz, ids = measure(objects)
        query = 'the chair near the door'
        p = validate_structure(query, {'Target': ['chair'], 'Anchor': ['door'], 'Attributes': [],
            'Observer': {}, 'Unhandled': [], 'Relations': [{'target_category': 'chair',
            'anchor_category': 'door', 'relation': 'near', 'phrase': 'near the door'}]})
        diagnosis = diagnose(p, objects, targets, [0, 0, 0], trusted_categories=True,
                             apply_size_constraints=False, query=query)
        report = {'status': 'consistent', 'relations': [{'candidate_id': i,
            'source_quote': 'near the door', 'status': state, 'reason': 'visible bound separation'}
            for i, state in ((1, 'pass'), (2, 'fail'))]}
        with tempfile.TemporaryDirectory() as out, patch('active_diagnostic.chat', return_value={
                '_content': json.dumps(report), '_finish_reason': 'stop', '_baseline': {}, 'usage': {}}):
            result = execute_evidence({'action': 'evidence', 'dimension': 'distance', 'reason': 'bind proximity'},
                query, p, diagnosis, targets, objects, xyz, np.full(xyz.shape, 100, dtype=np.uint8),
                ids, Path(out), 1, [])
        states = {r['candidate_id']: r['status'] for r in p['_v4_relation_evidence']}
        self.assertEqual(states, {1: 'pass', 2: 'fail'})
        self.assertTrue(next(r for r in result['candidate_evidence'] if r['candidate_id'] == 2)['removal_allowed'])


if __name__ == '__main__':
    unittest.main()
