"""General v7 behaviors and counterexamples; synthetic IDs and no API calls."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from seeground.grounding.diagnostic import validate_structure, size_conditions, appearance_attributes_for, diagnose
from seeground.grounding.evidence import match_retrieval_categories
from seeground.query.query_graph import exclusion_clauses, polarity
from seeground.query.query_syntax import analyze_query
from seeground.geometry.size_semantics import evaluate_size_condition
from seeground.grounding.active_diagnostic import _appearance_review_requirements, active_ground
from seeground.grounding.appearance_evidence import normalize_visual_check
from seeground.grounding.v4_policy import settings, v62_enabled
from seeground.grounding.v7_evidence import box_rows, relation_preferences, supported_choice


def obj(oid, category, xyz=(0, 0, 1), dims=(1, 1, 1)):
    return {'bbox_id': oid, 'target': category, 'bbox_3d': [*xyz, *dims]}


def parse(query, category, attrs, anchors=(), relations=(), unhandled=()):
    return validate_structure(query, {'Target': [category], 'Anchor': list(anchors),
        'Attributes': [{'category': category, 'phrase': p, 'role': 'target'} for p in attrs],
        'Relations': list(relations), 'Observer': {}, 'Unhandled': list(unhandled)})


def check(oid, attrs, visibility='visible'):
    return {'id': oid, 'category': 'table', 'role': 'target', 'category_verdict': 'yes',
        'verdict': 'yes', 'visibility': visibility, 'reason': 'visible table',
        'attribute_checks': [{'attribute': name, 'verdict': value, 'visibility': visibility,
                              'reason': 'visible brown top' if name == 'brown' else 'visible incompatible brown top'}
                             for name, value in attrs.items()]}


class V7Tests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'SEEGROUND_V4': '1', 'SEEGROUND_V7': '1', 'SEEGROUND_V62': '0',
            'SEEGROUND_V6': '0', 'SEEGROUND_V5': '0', 'SEEGROUND_SYNTAX_BACKEND': 'spacy',
            'SEEGROUND_V4_FINAL_REVIEW': '0'})
        env.start()
        self.addCleanup(env.stop)

    def test_version_and_strict_contract_inherited(self):
        self.assertEqual(settings()['version'], 'relative-size-v7.1')
        self.assertTrue(v62_enabled())
        from seeground.api.response_schemas import schema_kwargs, parser_schema
        with patch('seeground.api.qwen_api.api_provider', return_value='openai'):
            self.assertTrue(schema_kwargs('parser', parser_schema('diagnostic'))['response_schema']['strict'])

    def test_width_comparatives_use_horizontal_short_extent(self):
        objects = [obj(3, 'rail', dims=(.16, .08, .1)), obj(10, 'rail', dims=(.09, .07, .1))]
        parsed = parse('the wider of the two rails', 'rail', ['wider'])
        spec = size_conditions(parsed, 'rail')[0]
        self.assertEqual(spec['metric'], 'horizontal_short_extent_m')
        self.assertEqual(spec['direction'], 'max')
        self.assertEqual(spec['required_member_count'], 2)
        result = evaluate_size_condition(spec, box_rows(objects), [3, 10], scope_bound=True)
        self.assertEqual(result['tentative_selected_ids'], [3])
        parsed = parse('the narrower of the two rails', 'rail', ['narrower'])
        spec = size_conditions(parsed, 'rail')[0]
        result = evaluate_size_condition(spec, box_rows(objects), [3, 10], scope_bound=True)
        self.assertEqual(result['tentative_selected_ids'], [10])

    def test_synonyms_and_distinct_categories(self):
        objects = [obj(81, 'blackboard'), obj(23, 'whiteboard')]
        for category in ('chalk board', 'chalkboards', 'blackboard'):
            self.assertEqual([o['bbox_id'] for o in match_retrieval_categories([category], objects)], [81])
        with patch.dict(os.environ, {'SEEGROUND_V7': '0'}):
            self.assertEqual(match_retrieval_categories(['chalk board'], objects), [])

    def test_postnominal_negation_scope_and_actual_visual_contract(self):
        query = 'The big brown table not the white one'
        exclusions = exclusion_clauses(query, analyze_query(query))
        self.assertEqual([polarity(query, p, exclusions, attribute=True) for p in ('big', 'brown', 'white')],
                         ['positive', 'positive', 'exclude'])
        parsed = parse(query, 'table', ['big brown', 'white'], unhandled=['not the white one'])
        self.assertEqual(appearance_attributes_for(parsed, 'table'), ['brown', 'not white'])
        self.assertEqual([s['word'] for s in size_conditions(parsed, 'table')], ['big'])
        objects = [obj(71, 'table'), obj(82, 'table', dims=(2, 2, 1))]
        req = _appearance_review_requirements(parsed, {'candidate_ids': [71, 82]}, [71, 82], objects)
        self.assertEqual(req[0]['attributes'], ['brown', 'not white'])
        from seeground.query.query_rewrite import rewrite_query
        rewritten = rewrite_query(query, parsed)
        self.assertTrue(any(c.get('polarity') == 'exclude' and c.get('executor') == 'negative_color'
                            for c in rewritten['constraints']))
        normalized = normalize_visual_check(check(71, {'brown': 'yes', 'not white': 'yes'}), req[0]['attributes'])
        self.assertEqual(normalized['verdict'], 'yes')
        hidden = normalize_visual_check(check(71, {'brown': 'yes', 'not white': 'yes'}, 'occluded'), req[0]['attributes'])
        self.assertEqual(hidden['verdict'], 'uncertain')

    def test_negative_relation_does_not_negate_reference_color(self):
        query = 'the smaller chair not near the white table'
        exclusions = exclusion_clauses(query, analyze_query(query))
        self.assertEqual(polarity(query, 'white', exclusions, attribute=True), 'positive')
        parsed = parse('the table that is not smaller', 'table', ['smaller'])
        self.assertEqual(size_conditions(parsed, 'table'), [])

    def test_part_category_recall(self):
        parsed = parse('the largest section of table located in a corner', 'section', ['largest'],
                       anchors=['table', 'corner'], unhandled=['located in a corner'])
        self.assertEqual(parsed['Target'], ['table'])
        self.assertEqual(size_conditions(parsed, 'table')[0]['metric'], 'footprint_m2')

    def test_valid_boxes_survive_failed_point_fit_and_scan_risk(self):
        parsed = parse('the largest trash can', 'trash can', ['largest'])
        objects = [obj(91, 'trash can', dims=(1, 1, 2)), obj(62, 'trash can', dims=(1, 1, .5))]
        rows = box_rows(objects, [{'id': 91, 'oriented_volume_m3': None,
                   'observed_completeness': {'status': 'suspected_incomplete'}}])
        result = evaluate_size_condition(size_conditions(parsed, 'trash can')[0], rows, [91, 62], scope_bound=True)
        self.assertEqual(result['tentative_selected_ids'], [91])
        self.assertEqual(result['selected_ids'], [])
        self.assertEqual(result['risks']['91']['status'], 'suspected_incomplete')
        self.assertEqual({e['status'] for e in result['candidate_evidence']}, {'unknown'})

    def test_height_small_gap_translation_and_exact_tie(self):
        parsed = parse('the taller desk', 'desk', ['taller'])
        spec = size_conditions(parsed, 'desk')[0]
        self.assertEqual(spec['metric'], 'height_m')
        objects = [obj(45, 'desk', xyz=(0, 0, .5), dims=(1, 1, 1.22)),
                   obj(12, 'desk', xyz=(2, 0, 5), dims=(1, 1, 1.19))]
        for pool in (objects, objects[::-1]):
            result = evaluate_size_condition(spec, box_rows(pool), [45, 12], scope_bound=True)
            self.assertEqual(result['tentative_selected_ids'], [45])
            self.assertLess(result['relative_gap'], .10)
        objects[1]['bbox_3d'][5] = 1.22
        result = evaluate_size_condition(spec, box_rows(objects), [45, 12], scope_bound=True)
        self.assertEqual(result['tentative_selected_ids'], [])
        objects[1]['bbox_3d'][3] = float('nan')
        self.assertEqual(evaluate_size_condition(spec, box_rows(objects), [45, 12], scope_bound=True)['tentative_selected_ids'], [])

    def test_planar_box_metric_ignores_thickness(self):
        parsed = parse('the larger whiteboard', 'whiteboard', ['larger'])
        objects = [obj(28, 'whiteboard', dims=(.1, 2, 1.5)), obj(43, 'whiteboard', dims=(.4, 2, .5))]
        spec = size_conditions(parsed, 'whiteboard')[0]
        self.assertEqual(evaluate_size_condition(spec, box_rows(objects), [28, 43], scope_bound=True)['tentative_selected_ids'], [28])

    def test_multi_anchor_consensus_disagreement_and_far_nearest(self):
        query = 'the window near the monitor'
        parsed = parse(query, 'window', [], anchors=['monitor'], relations=[{
            'target_category': 'window', 'anchor_category': 'monitor', 'relation': 'near', 'phrase': 'near the monitor'}])
        windows = [obj(65, 'window'), obj(17, 'window', xyz=(4, 0, 1))]
        monitors = [obj(3, 'monitor', xyz=(1, 0, 1)), obj(8, 'monitor', xyz=(1, .3, 1))]
        group = relation_preferences(query, parsed, windows+monitors, windows, [])[0]
        self.assertEqual(group['preferred_ids'], [65])
        monitors[1]['bbox_3d'][0] = 3
        self.assertEqual(relation_preferences(query, parsed, windows+monitors, windows, [])[0]['preferred_ids'], [])
        for m in monitors:
            m['bbox_3d'][0] = -20
        self.assertEqual(relation_preferences(query, parsed, windows+monitors, windows, [])[0]['preferred_ids'], [])

    def test_corner_and_support_counterexamples(self):
        from seeground.geometry.v4_relations import geometric_relation
        walls = [obj(4, 'wall', xyz=(0, 3, 1), dims=(.1, 6, 2)),
                 obj(9, 'wall', xyz=(3, 0, 1), dims=(6, .1, 2))]
        task = {'kind': 'corner'}
        self.assertEqual(geometric_relation(task, obj(63, 'table', xyz=(.5, .5, 1)), walls)['status'], 'pass')
        self.assertEqual(geometric_relation(task, obj(63, 'table', xyz=(.5, 3, 1)), walls)['status'], 'unknown')
        table = obj(31, 'table', xyz=(0, 0, .5))
        task = {'kind': 'support'}
        self.assertEqual(geometric_relation(task, obj(63, 'book', xyz=(0, 0, 1.05), dims=(.2, .2, .1)), [table])['status'], 'pass')
        self.assertEqual(geometric_relation(task, obj(63, 'book', xyz=(0, 0, 2), dims=(.2, .2, .1)), [table])['status'], 'fail')

    def test_verified_color_contradiction_vetoes_size_and_no_evidence_abstains(self):
        query = 'the big brown table not the white one'
        parsed = parse(query, 'table', ['big brown', 'white'])
        objects = [obj(71, 'table'), obj(82, 'table', dims=(2, 2, 1))]
        diagnosis = {'candidate_ids': [71, 82], 'missing': ['unresolved'], 'candidate_removal_audit': []}
        checks = [check(71, {'brown': 'yes', 'not white': 'yes'}), check(82, {'brown': 'no', 'not white': 'no'})]
        result = supported_choice(query, parsed, objects, objects, diagnosis, checks)
        self.assertEqual(result['selected_id'], 71)
        self.assertIn(82, result['contradicted_ids'])
        plain = parse('the table', 'table', [])
        self.assertIsNone(supported_choice('the table', plain, objects, objects, diagnosis)['selected_id'])

    def test_incomparable_independent_evidence_does_not_vote(self):
        query = 'the big brown table'
        parsed = parse(query, 'table', ['big brown'])
        objects = [obj(71, 'table'), obj(82, 'table', dims=(2, 2, 1))]
        result = supported_choice(query, parsed, objects, objects, {'candidate_ids': [71, 82]},
                                  [check(71, {'brown': 'yes'})])
        self.assertIsNone(result['selected_id'])

    def test_fitted_corner_rotation_and_finite_extent(self):
        import numpy as np
        from seeground.grounding.v7_evidence import fitted_corner
        walls = []
        for oid, normal, tangent in ((16, [1, 0], [0, 1]), (36, [0, 1], [1, 0])):
            walls.append({'bbox_id': oid, 'wall_geometry': {'valid': True,
                'normal_xy': normal, 'tangent_xy': tangent, 'origin_xy_m': [0, 0],
                'tangent_extent_m': [-3, 3]}})
        target = obj(85, 'table', xyz=(.4, .4, 1))
        self.assertEqual(fitted_corner(target, walls)['status'], 'pass')
        angle = .63
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated = []
        for wall in walls:
            plane = wall['wall_geometry']
            rotated.append({'bbox_id': wall['bbox_id'], 'wall_geometry': {**plane,
                'normal_xy': (rotation @ plane['normal_xy']).tolist(),
                'tangent_xy': (rotation @ plane['tangent_xy']).tolist()}})
        # Isotropic target keeps its bounding footprint under this rotation.
        xy = (rotation @ np.array(target['bbox_3d'][:2])).tolist()
        target['bbox_3d'][:2] = xy
        self.assertEqual(fitted_corner(target, rotated)['status'], 'pass')
        rotated[0]['wall_geometry']['tangent_extent_m'] = [10, 12]
        self.assertEqual(fitted_corner(target, rotated)['status'], 'unknown')

    def test_negation_executor_ledger_and_qualified_absence_unknown(self):
        parsed = parse('the brown table not the white one', 'table', ['brown', 'white'],
                       unhandled=['not the white one'])
        negative = [c for c in parsed['query_structure']['conditions'] if c.get('executor') == 'negative_color']
        self.assertTrue(negative)
        self.assertNotIn('negation/exclusion is unsupported', parsed['diagnostic_missing'])
        parsed = parse('the table without white legs', 'table', ['white'])
        self.assertNotIn('not white', appearance_attributes_for(parsed, 'table'))
        self.assertIn('negation/exclusion is unsupported', parsed['diagnostic_missing'])

    def test_unbound_scope_measures_full_category_without_certifying(self):
        from seeground.grounding.active_diagnostic import compute_observation
        query = 'the smaller window near the monitor'
        parsed = parse(query, 'window', ['smaller'], anchors=['monitor'], relations=[{
            'target_category': 'window', 'anchor_category': 'monitor', 'relation': 'near', 'phrase': 'near the monitor'}])
        objects = [obj(68, 'window', dims=(.2, 1, 2)), obj(27, 'window', dims=(.2, 3, 2)), obj(36, 'monitor')]
        diagnosis = diagnose(parsed, objects, objects[:2], [0, 0, 0], trusted_categories=True, apply_size_constraints=False, query=query)
        diagnosis['size_comparison_pool_ids'] = [68, 27]
        with patch('seeground.grounding.active_diagnostic.point_cloud_measurements', return_value=[]):
            details = compute_observation('size', query, parsed, diagnosis, objects[:2], objects)
        self.assertEqual({m['id'] for m in details['comparison_metrics']}, {68, 27})
        self.assertEqual(details['size_decision']['tentative_selected_ids'], [68])
        self.assertEqual({e['status'] for e in details['candidate_evidence']}, {'unknown'})

    def test_stop_schedules_unchecked_size_with_existing_budget(self):
        import numpy as np
        query = 'the taller desk'
        parsed = parse(query, 'desk', ['taller'])
        objects = [obj(45, 'desk', dims=(1, 1, 1.22)), obj(12, 'desk', dims=(1, 1, 1.19))]
        data = {'_content': json.dumps({'action': 'stop', 'reason': 'unknown'}),
                '_finish_reason': 'stop', '_baseline': {}, 'usage': {}}
        evidence = {'tool': 'evidence', 'dimension': 'size', 'details': {'candidate_evidence': [], 'size_constraints': []},
                    'candidate_evidence': [], 'checks': [], 'images': [], 'internal_steps': []}
        with tempfile.TemporaryDirectory() as tmp, patch('seeground.grounding.active_diagnostic.chat', return_value=data), \
                patch('seeground.grounding.active_diagnostic.execute_evidence', return_value=evidence) as execute, \
                patch('seeground.grounding.active_diagnostic.react_backend', return_value='openai'):
            result = active_ground(query, parsed, objects, objects, objects,
                np.array([[0., 0., 0.], [3., 3., 3.]]), np.zeros((2, 3)), np.array([45, 12]),
                Path(tmp), [], {}, 1, True)
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(execute.call_args.args[0]['dimension'], 'size')
        self.assertEqual(result['active_trace']['tool_budget'], 1)
        self.assertTrue(result['active_trace']['v7_schedule_corrections'])

    def test_tentative_end_to_end_stop_and_unchanged_budget(self):
        import numpy as np
        query = 'the taller desk'
        parsed = parse(query, 'desk', ['taller'])
        objects = [obj(45, 'desk', dims=(1, 1, 1.22)), obj(12, 'desk', dims=(1, 1, 1.19))]
        data = {'_content': json.dumps({'action': 'stop', 'reason': 'insufficient evidence'}),
                '_finish_reason': 'stop', '_baseline': {}, 'usage': {}}
        with tempfile.TemporaryDirectory() as tmp, patch('seeground.grounding.active_diagnostic.chat', return_value=data), \
                patch('seeground.grounding.active_diagnostic.react_backend', return_value='openai'):
            result = active_ground(query, parsed, objects, objects, objects,
                np.array([[0., 0., 0.], [3., 3., 3.]]), np.zeros((2, 3)), np.array([45, 12]),
                Path(tmp), [], {}, 0, True)
        self.assertEqual(result['predicted_id'], 45)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['implementation_version'], 'relative-size-v7.1')
        self.assertEqual(result['active_trace']['tool_budget'], 0)


if __name__ == '__main__':
    unittest.main()
