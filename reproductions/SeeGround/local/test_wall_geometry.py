import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import numpy as np
from diagnostic import _between_geometry, diagnose, validate_structure, recover_between_relations
from wall_geometry import attach_wall_geometry, fit_wall, wall_reference_point, main


def box(oid, category, x, y):
    return {'bbox_id': oid, 'target': category, 'bbox_3d': [x, y, .5, .4, .4, 1.]}


class WallGeometryTests(unittest.TestCase):
    def wall(self):
        points = np.column_stack([np.full(100, 2.), np.linspace(-4, 4, 100), np.ones(100)])
        return {**box(3, 'wall', 2., 0.), 'wall_geometry': fit_wall(points)}

    def test_wall_uses_local_surface_instead_of_distant_center(self):
        wall = self.wall()
        target, stove = box(1, 'cabinet', 1., 3.), box(2, 'stove', 0., 3.)
        result = _between_geometry(target, stove, wall)
        self.assertTrue(result['passes'])
        self.assertFalse(result['center_point_comparison']['passes'])
        self.assertEqual(result['wall_reference_xy_m'], [2., 3.])
        self.assertAlmostEqual(result['projection_t'], .5)
        for target in (box(1, 'cabinet', 1., 1.), box(1, 'cabinet', 3., 3.), box(1, 'cabinet', -1., 3.)):
            self.assertFalse(_between_geometry(target, stove, wall)['passes'])

    def test_reference_order_rotation_and_translation_do_not_change_passes(self):
        wall = self.wall()
        target, stove = box(1, 'cabinet', 1., 3.), box(2, 'stove', 0., 3.)
        self.assertTrue(_between_geometry(target, wall, stove)['passes'])
        angle = .6
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        shift = np.array([7., -3.])
        items = copy.deepcopy([target, stove, wall])
        for item in items:
            item['bbox_3d'][:2] = (rotation @ np.asarray(item['bbox_3d'][:2]) + shift).tolist()
        fit = items[-1]['wall_geometry']
        fit['origin_xy_m'] = (rotation @ np.asarray(fit['origin_xy_m']) + shift).tolist()
        for key in ('tangent_xy', 'normal_xy'):
            fit[key] = (rotation @ np.asarray(fit[key])).tolist()
        self.assertTrue(_between_geometry(*items)['passes'])

    def test_finite_wall_projection_is_clamped(self):
        point, clamped = wall_reference_point(self.wall()['wall_geometry'], box(2, 'stove', 0, 8))
        np.testing.assert_allclose(point, [2., 4.])
        self.assertTrue(clamped)

    def test_insufficient_and_nonplanar_points_stay_unknown(self):
        self.assertFalse(fit_wall(np.zeros((5, 3)))['valid'])
        self.assertFalse(fit_wall(np.ones((30, 3)))['valid'])
        grid = np.array([[x, y, 1.] for x in np.linspace(-2, 2, 10) for y in np.linspace(-2, 2, 10)])
        self.assertFalse(fit_wall(grid)['valid'])
        objects = [box(1, 'cabinet', 1, 0), box(2, 'stove', 0, 0), box(3, 'wall', 2, 0)]
        query = 'cabinet between the stove and the wall'
        data = validate_structure(query, recover_between_relations(query, {
            'Target': ['cabinet'], 'Anchor': ['stove', 'wall'], 'Relations': [],
            'Observer': {}, 'Attributes': [], 'Unhandled': []}))
        result = diagnose(data, objects, objects[:1], [0, 0, .5], trusted_categories=True)
        self.assertEqual(result['status'], 'missing')
        self.assertEqual(result['candidate_ids'], [1])
        self.assertIsNone(result['geometry']['between_triplets'][0]['passes'])

    def test_passing_and_unknown_pairs_are_reported_separately_without_filtering(self):
        objects=[box(1,'cabinet',1,3), box(4,'cabinet',1,0), box(2,'stove',0,3),
                 self.wall(), box(5,'wall',2,2)]
        query='cabinet between the stove and the wall'
        data=validate_structure(query, recover_between_relations(query, {
            'Target':['cabinet'],'Anchor':['stove','wall'],'Relations':[],
            'Observer':{},'Attributes':[],'Unhandled':[]}))
        result=diagnose(data, objects, objects[:2], [0,0,.5], trusted_categories=True)
        geometry=result['geometry']
        self.assertEqual(geometry['between_pass_ids'],[1])
        self.assertEqual(geometry['between_unknown_ids'],[1,4])
        self.assertEqual(result['candidate_ids'],[1,4])
        self.assertEqual(result['status'],'missing')
        by_id={row['target_id']:row for row in geometry['between_candidate_evidence']}
        self.assertEqual(by_id[1]['passing_reference_pairs'],[[2,3]])
        self.assertEqual(by_id[4]['passing_reference_pairs'],[])
        self.assertEqual(by_id[4]['unknown_reference_pairs'],[[2,5]])
        self.assertEqual(by_id[4]['failing_reference_pairs'],[[2,3]])

    def test_fit_uses_only_points_bound_to_wall_id(self):
        objects = [box(1, 'stove', 0, 0), box(3, 'wall', 2, 0)]
        points = np.column_stack([np.full(100, 2.), np.linspace(-4, 4, 100), np.ones(100)])
        result = attach_wall_geometry(objects, np.vstack([points, [100., 100., 0.]]),
                                      np.r_[np.full(100, 3), 1])
        self.assertNotIn('wall_geometry', result[0])
        self.assertTrue(result[1]['wall_geometry']['valid'])
        self.assertEqual(result[1]['wall_geometry']['point_count'], 100)

    def test_offline_command_uses_saved_candidates_and_never_calls_api(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            objects = [box(1, 'cabinet', 1, 3), box(2, 'stove', 0, 3)]
            (root/'objects_gt.json').write_text(json.dumps(objects))
            (root/'scene_instances.json').write_text(json.dumps([box(3, 'wall', 2, 0)]))
            points = np.column_stack([np.full(100, 2.), np.linspace(-4, 4, 100), np.ones(100)])
            np.savez(root/'scene.npz', xyz=points, ids=np.full(100, 3))
            path = root/'run.json'
            path.write_text(json.dumps({
                'query': 'skinny cabinet between the stove and the wall', 'object_table': 'gt',
                'parsed_query': {'Target': ['cabinet'], 'Anchor': ['stove', 'wall'],
                                 'Relations': [], 'Observer': {}, 'Attributes': [], 'Unhandled': []},
                'review_proposal': {'ids': [1]}, 'diagnosis': {'checks': []}}))
            original = path.read_bytes()
            output = io.StringIO()
            with patch('sys.argv', ['wall_geometry', '--scene-dir', str(root), '--run', str(path)]), \
                 patch('qwen_api.requests.post') as api, redirect_stdout(output):
                main()
            result = json.loads(output.getvalue())
            api.assert_not_called()
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(result['network_request'])
            self.assertEqual(result['status'], 'missing')
            self.assertEqual(result['reference_groups']['2,3']['wall_pass_ids'], [1])
            self.assertEqual(result['reference_groups']['2,3']['center_pass_ids'], [])
            self.assertEqual(result['between_pass_ids'],[1])
            self.assertEqual(result['between_unknown_ids'],[])
            self.assertEqual(result['candidate_shape_metrics'][0]['horizontal_long_extent_m'],.4)


if __name__ == '__main__':
    unittest.main()
