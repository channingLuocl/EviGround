"""Reference size must use the active shared measurement gate before finish."""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from seeground.grounding.active_diagnostic import (active_ground, apply_computed_evidence,
                               apply_size_observation, compute_observation)
from seeground.grounding.diagnostic import diagnose, validate_structure
from seeground.geometry.object_geometry import point_cloud_measurements
from seeground.query.query_rewrite import rewrite_query
from seeground.experiments.replay_module_acceptance import replay_inference
from seeground.geometry.size_semantics import evaluate_size_condition, size_specs


QUERY = 'The chair nearest the taller table.'


def raw_parse():
    return {'Target': ['chair'], 'Anchor': ['table'],
            'Attributes': [{'category': 'table', 'phrase': 'taller', 'role': 'reference'}],
            'Relations': [{'target_category': 'chair', 'anchor_category': 'table',
                           'relation': 'nearest', 'scope': 'relative_anchor',
                           'phrase': 'nearest the taller table'}], 'Observer': {}, 'Unhandled': []}


def fixture(missing_reference=False):
    objects = [
        {'bbox_id': 1, 'target': 'chair', 'bbox_3d': [0, 0, .5, .5, .5, 1]},
        {'bbox_id': 2, 'target': 'chair', 'bbox_3d': [10, 0, .5, .5, .5, 1]},
        {'bbox_id': 10, 'target': 'table', 'bbox_3d': [0, 0, .5, 1, 1, 1]},
        {'bbox_id': 20, 'target': 'table', 'bbox_3d': [10, 0, 1, 1, 1, 2]}]
    points, ids = [], []
    for obj in objects[:-1] if missing_reference else objects:
        corners = np.array(list(itertools.product((-.5, .5), repeat=3)))
        points.extend(corners * obj['bbox_3d'][3:] + obj['bbox_3d'][:3])
        ids.extend([obj['bbox_id']] * len(corners))
    xyz, ids = np.asarray(points), np.asarray(ids)
    parsed = validate_structure(QUERY, raw_parse())
    parsed['query_rewrite'] = rewrite_query(QUERY, parsed)
    return objects, xyz, ids, parsed


def active(objects, xyz, ids, parsed, actions, limit):
    actions = iter(actions)

    def mock_chat(*args, **kwargs):
        return {'_content': json.dumps(next(actions)), '_baseline': {'latency_s': .01, 'model': 'mock'},
                '_finish_reason': 'stop', 'usage': {}}

    with tempfile.TemporaryDirectory() as directory, \
            patch.dict('os.environ', {'SEEGROUND_REACT_BACKEND': 'qwen'}), \
            patch('seeground.grounding.active_diagnostic.chat', side_effect=mock_chat):
        return active_ground(QUERY, parsed, objects, objects, objects[:2], xyz,
            np.zeros_like(xyz, dtype=np.uint8), ids, Path(directory), [], {}, limit, True)


def replay(objects, xyz, ids):
    saved = {'calls': [{'stage': 'parse_query', 'response': json.dumps(raw_parse()),
                        'finish_reason': 'stop'}]}
    with tempfile.TemporaryDirectory() as directory:
        scene = Path(directory)
        (scene / 'objects_gt.json').write_text(json.dumps(objects))
        np.savez(scene / 'scene.npz', xyz=xyz, ids=ids)
        return replay_inference(QUERY, saved, scene, include_size=True)


class RelativeReferenceSizeTests(unittest.TestCase):
    def setUp(self):
        network = patch('requests.sessions.Session.request',
                        side_effect=AssertionError('Reference gate tests must not access the network'))
        network.start()
        self.addCleanup(network.stop)

    def test_reference_comparative_is_deferred_before_instance_measurement(self):
        objects, _, _, parsed = fixture()
        diagnosis = diagnose(parsed, objects, objects[:2], [5, 0, 1], trusted_categories=True,
                             apply_size_constraints=False)
        self.assertEqual(diagnosis['status'], 'missing')
        self.assertEqual(diagnosis['candidate_ids'], [1, 2])
        self.assertTrue(any(c['attribute'] == 'taller' and c['role'] == 'reference'
                            for c in diagnosis['pending_size_conditions']))

    def test_direct_finish_and_exhausted_budget_do_not_prove_reference_size(self):
        for missing_reference in (False, True):
            for limit in (0, 1):
                with self.subTest(missing_reference=missing_reference, limit=limit):
                    objects, xyz, ids, parsed = fixture(missing_reference)
                    result = active(objects, xyz, ids, parsed, [
                        {'action': 'finish', 'predicted_id': 2, 'reason': 'closest to the taller OLT box'},
                        {'action': 'stop', 'reason': 'reference size has not been measured'}], limit)
                    self.assertFalse(result['accepted'])
                    self.assertEqual(result['active_trace']['successful_tools'], 0)
                    self.assertTrue(result['diagnosis']['pending_size_conditions'])

    def test_target_same_word_pass_cannot_clear_unmeasured_reference_obligation(self):
        query = 'The taller chair nearest the taller table.'
        objects, _, _, _ = fixture()
        raw = raw_parse()
        raw['Attributes'].append({'category': 'chair', 'phrase': 'taller', 'role': 'target'})
        parsed = validate_structure(query, raw)
        diagnosis = diagnose(parsed, objects, objects[:2], [5, 0, 1],
                             trusted_categories=True, query=query, apply_size_constraints=False)
        self.assertEqual(parsed['diagnostic_missing'], [])
        self.assertEqual({c['role'] for c in diagnosis['pending_size_conditions']}, {'target', 'reference'})
        spec, = size_specs('taller', 'chair', query=query, role='target')
        measured_target = [{'id': oid, 'status': 'known', 'height_m': height,
                            'source': 'synthetic complete target height measurement',
                            'metric_reliability': {'height_m': {'status': 'usable'}}}
                           for oid, height in ((1, 1.), (2, 2.))]
        target_condition = evaluate_size_condition(spec, measured_target, [1, 2],
                                                   scope_bound=True, candidate_ids=[1, 2])
        apply_computed_evidence(diagnosis, target_condition['candidate_evidence'], 'size',
                                size_constraints=[target_condition])
        self.assertEqual(diagnosis['candidate_ids'], [2])
        self.assertTrue(any(c['attribute'] == 'taller' and c['role'] == 'reference'
                            for c in diagnosis['pending_size_conditions']))
        self.assertEqual(diagnosis['status'], 'missing')
        audit = diagnosis['active_computed_evidence'][0]['candidate_removal_audit']
        removed = next(item for item in audit if item['candidate_id'] == 1)
        self.assertEqual(removed['target_role'], 'target')
        self.assertEqual(removed['source_quote'], 'taller')
        self.assertEqual(removed['comparison_scope_ids'], [1, 2])
        self.assertEqual(removed['measurement_source'],
                         'synthetic complete target height measurement')
        self.assertTrue(removed['removal_allowed'])

    def test_missing_reference_points_remain_unknown_after_size_operation(self):
        objects, xyz, ids, parsed = fixture(missing_reference=True)
        measured = point_cloud_measurements(objects, xyz, ids)
        ordinary = diagnose(parsed, objects, objects[:2], [5, 0, 1], trusted_categories=True,
                            query=QUERY, size_measurements=measured)
        initial = diagnose(parsed, objects, objects[:2], [5, 0, 1], trusted_categories=True,
                           apply_size_constraints=False)
        observation = compute_observation('size', QUERY, parsed, initial, objects[:2], objects, xyz, ids)
        applied = apply_size_observation(copy.deepcopy(initial), observation)
        replayed = replay(objects, xyz, ids)['size_diagnosis']
        for result in (ordinary, applied, replayed):
            self.assertEqual(result['status'], 'missing')
            self.assertEqual(result['candidate_ids'], [1, 2])
            condition = next(c for c in result['geometry']['size_constraints'] if c['role'] == 'reference')
            self.assertEqual(condition['unknown_measurement_ids'], [20])
            self.assertTrue(all(e['status'] == 'unknown' for e in condition['candidate_evidence']))
        result = active(objects, xyz, ids, parsed, [
            {'action': 'evidence', 'dimension': 'size', 'reason': 'measure both reference heights'},
            {'action': 'finish', 'predicted_id': 2, 'reason': 'taller reference remains unavailable'}], 1)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['active_trace']['successful_tools'], 1)

    def test_unknown_measured_reference_cannot_be_excluded_by_distance_refresh(self):
        objects, xyz, ids, parsed = fixture(missing_reference=True)
        diagnosis = diagnose(parsed, objects, objects[:2], [5, 0, 1], trusted_categories=True,
                             query=QUERY, size_measurements=point_cloud_measurements(objects, xyz, ids))
        self.assertEqual(diagnosis['candidate_ids'], [1, 2])
        self.assertTrue(any(item['role'] == 'reference'
                            for item in diagnosis['pending_size_conditions']))
        evidence = compute_observation('distance', QUERY, parsed, diagnosis,
                                       objects[:2], objects)['candidate_evidence']
        self.assertTrue(all(item['status'] == 'unknown' for item in evidence))
        update = apply_computed_evidence(diagnosis, evidence, 'distance')
        self.assertEqual(update['removed_ids'], [])
        self.assertEqual(diagnosis['candidate_ids'], [1, 2])

    def test_complete_reference_points_resolve_normal_active_and_replay_identically(self):
        objects, xyz, ids, parsed = fixture()
        measured = point_cloud_measurements(objects, xyz, ids)
        ordinary = diagnose(parsed, objects, objects[:2], [5, 0, 1], trusted_categories=True,
                            query=QUERY, size_measurements=measured)
        replayed = replay(objects, xyz, ids)['size_diagnosis']
        result = active(objects, xyz, ids, parsed, [
            {'action': 'evidence', 'dimension': 'size', 'reason': 'measure both reference heights'},
            {'action': 'finish', 'predicted_id': 2, 'reason': 'nearest the reliably taller table'}], 1)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['active_trace']['successful_tools'], 1)
        for diagnosis in (ordinary, replayed, result['diagnosis']):
            self.assertEqual(diagnosis['status'], 'resolved')
            self.assertEqual(diagnosis['candidate_ids'], [2])
            self.assertEqual(diagnosis['pending_size_conditions'], [])
            condition = next(c for c in diagnosis['geometry']['size_constraints'] if c['role'] == 'reference')
            self.assertEqual(condition['comparison_pool_ids'], [10, 20])
            self.assertEqual(condition['selected_ids'], [20])
            self.assertEqual(condition['status'], 'known')
        self.assertEqual(replayed['hypotheses'], ordinary['hypotheses'])
        self.assertEqual(result['diagnosis']['hypotheses'], ordinary['hypotheses'])


if __name__ == '__main__':
    unittest.main()
