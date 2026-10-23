"""Offline checks for label isolation, saved predictions and subset resumption."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import eviground.classifiers.object_classifier_v8 as classifier
import eviground.experiments.run_object_classification as batch
from eviground.rendering.mesh_renderer_v8 import read_scene_faces, render_mesh
from eviground.api.qwen_api import QwenAPIError


ENV = {'EVIGROUND_MODEL': 'test-model', 'EVIGROUND_BASE_URL': 'https://example.test/v1',
       'EVIGROUND_API_KEY': 'test-key', 'EVIGROUND_API_PROVIDER': 'openai',
       'EVIGROUND_V4': '0'}


def response(label='chair', **answer):
    return {'_content': json.dumps({'predicted_object': label, 'alternatives': [],
                                   'visibility': 'adequate', 'reason': 'visible backrest', **answer}),
            '_finish_reason': 'stop', '_baseline': {'model': 'test-model'}, 'usage': {}}


def fixture(root):
    scene_root = root/'scenes'
    manifest = root/'manifest.json'
    manifest.write_text(json.dumps({'rows': [{'scene': 'scene0001_00', 'query': 'secret query',
                                             'target_id': 3, 'instance_type': 'secret-gt'}]}))
    scene = scene_root/'scene0001_00'
    scene.mkdir(parents=True)
    xyz = np.random.default_rng(4).uniform(-.4, .4, (160, 3))
    xyz[80:, 0] += 1.2
    ids = np.repeat([3, 9], 80)
    np.savez_compressed(scene/'scene.npz', xyz=xyz, rgb=np.full((160, 3), 100, dtype=np.uint8), ids=ids)
    objects = [{'bbox_id': i, 'target': 'author-wrong', 'bbox_3d': [0, 0, 0, 1, 1, 1]} for i in [3, 9]]
    (scene/'objects.json').write_text(json.dumps(objects))
    (scene/'objects_gt.json').write_text(json.dumps([{**o, 'target': 'chair'} for o in objects]))
    args = argparse.Namespace(output=root/'output', scene_root=scene_root, manifest=manifest,
                              limit=1, seed=42, prepare_only=False, retry_errors=False, workers=1)
    return args, xyz, ids


class ObjectClassificationTests(unittest.TestCase):
    def test_real_render_and_request_have_no_gt_author_or_query(self):
        with tempfile.TemporaryDirectory() as td, patch.dict('os.environ', ENV):
            args, xyz, ids = fixture(Path(td))
            with np.load(args.scene_root/'scene0001_00/scene.npz') as cloud:
                rgb = cloud['rgb']
            with patch('eviground.classifiers.object_classifier_v8.chat', return_value=response()) as chat:
                prediction = classifier.classify_instance(xyz, rgb, ids, 3, Path(td)/'views', ['chair', 'stool'])
            messages = chat.call_args.args[0]
            text = json.dumps(messages)
            for forbidden in ('secret query', 'secret-gt', 'author-wrong'):
                self.assertNotIn(forbidden, text)
                self.assertNotIn(forbidden, (Path(td)/'views/request.json').read_text())
            payload = json.loads(messages[1]['content'][0]['text'])
            self.assertEqual(set(payload), {'bbox_id', 'allowed_categories', 'extent_xyz_m', 'point_count',
                                           'observed_bottom_above_floor_m'})
            self.assertNotIn('correct', text)
            self.assertEqual(prediction['predicted_object'], 'chair')
            images = [part for part in messages[1]['content'] if part['type'] == 'image_url']
            self.assertEqual(len(images), 4)
            self.assertNotIn('enum', chat.call_args.kwargs['response_schema']['schema']['properties']['predicted_object']['anyOf'][0])
            with Image.open(prediction['image']) as image:
                self.assertEqual(image.size, (1024, 1088))
            self.assertEqual(len(prediction['geometry']['views']), 4)

    def test_invalid_model_output_saved_and_rejected(self):
        for reply in [response('invented-class'), response('chair', alternatives=['chair']),
                      response(None), {**response(), '_finish_reason': 'length'}]:
            with self.subTest(reply=reply), tempfile.TemporaryDirectory() as td, \
                    patch.dict('os.environ', ENV), \
                    patch('eviground.classifiers.object_classifier_v8.render_instance', return_value=(Path(td)/'image.png',
                          {'extent_xyz_m': [1, 1, 1], 'point_count': 10,
                           'observed_bottom_above_floor_m': 0.,
                           'views': [{'image': str(Path(td)/'image.png')}]})), \
                    patch('eviground.classifiers.object_classifier_v8.image_part', return_value={'type': 'image_url'}), \
                    patch('eviground.classifiers.object_classifier_v8.chat', return_value=reply):
                with self.assertRaises(Exception):
                    classifier.classify_instance(None, None, None, 3, Path(td), ['chair', 'stool'])
                self.assertTrue((Path(td)/'response.json').exists())

    def test_camera_fits_long_flat_and_tall_instances_without_changing_scale_ratios(self):
        rng = np.random.default_rng(7)
        for extent in ([6., .3, 1.], [.01, 2., 2.], [.2, .3, .8]):
            points = rng.uniform(-.5, .5, (800, 3))*extent
            center = (points.min(0)+points.max(0))/2
            angles = classifier.instance_angles(points, center)
            self.assertEqual(len(angles), 3)
            for angle in angles:
                eye = classifier.camera_for_instance(points, center, angle)
                uv, depth = classifier.project(points, eye, center, classifier.VIEW_SIZE)
                self.assertTrue((depth > 0).all())
                self.assertGreaterEqual(uv.min(), classifier.VIEW_SIZE*.074)
                self.assertLessEqual(uv.max(), classifier.VIEW_SIZE*.926)
                self.assertGreater(np.ptp(uv, axis=0).max(), classifier.VIEW_SIZE*.5)
                self.assertLessEqual(classifier.point_radius(points, eye, center), 6)

    def test_mesh_depth_test_keeps_nearest_surface_and_instance_marker(self):
        with tempfile.TemporaryDirectory() as td:
            # Both triangles overlap in projection; the far triangle must not
            # overwrite the near triangle's colour or claim its instance ID.
            xyz = np.array([[-1., -1., 0.], [1., -1., 0.], [0., 1., 0.],
                            [-1., -1., -1.], [1., -1., -1.], [0., 1., -1.]])
            rgb = np.array([[200, 20, 30]]*3 + [[20, 20, 200]]*3, dtype=np.uint8)
            faces = np.array([[0, 1, 2], [3, 4, 5]])
            stats = render_mesh(xyz, rgb, faces, Path(td)/'surface.png',
                                eye=[0, 0, 3], at=[0, 0, 0], ids=np.array([3]*3+[9]*3))
            self.assertEqual(stats['marked_ids'], [3])
            with Image.open(Path(td)/'surface.png') as image:
                pixels = np.asarray(image)
            self.assertFalse(((pixels[:, :, 2] > 150) & (pixels[:, :, 0] < 50)).any())
            self.assertGreater(stats['visible_pixels'], 1000)

    def test_mesh_vertex_mismatch_is_rejected_before_rendering(self):
        from plyfile import PlyData, PlyElement
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)/'downloads'
            directory.mkdir()
            vertices = np.array([(0., 0., 0., 10, 20, 30), (1., 0., 0., 10, 20, 30),
                                 (0., 1., 0., 10, 20, 30)],
                                dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                       ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
            faces = np.array([([0, 1, 2],)], dtype=[('vertex_indices', 'i4', (3,))])
            PlyData([PlyElement.describe(vertices, 'vertex'), PlyElement.describe(faces, 'face')]).write(
                str(directory/'test_vh_clean_2.ply'))
            (directory/'test.txt').write_text('axisAlignment = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n')
            xyz = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
            rgb = np.array([[10, 20, 30]]*3, dtype=np.uint8)
            np.testing.assert_array_equal(read_scene_faces(Path(td), 'test', xyz, rgb), [[0, 1, 2]])
            with self.assertRaisesRegex(ValueError, 'do not match'):
                read_scene_faces(Path(td), 'test', xyz[::-1], rgb)

    def test_selection_is_unique_reproducible_and_expandable_without_labels(self):
        with tempfile.TemporaryDirectory() as td:
            args, _, _ = fixture(Path(td))
            first = batch.select_instances(args.scene_root, args.manifest, 1, 42)
            all_jobs = batch.select_instances(args.scene_root, args.manifest, 0, 42)
            self.assertEqual(first, all_jobs[:1])
            self.assertEqual(len(all_jobs), 2)
            self.assertTrue(all('target' not in job for job in all_jobs))

    def test_resume_reuses_wrong_predictions_and_expansion_calls_only_new_objects(self):
        with tempfile.TemporaryDirectory() as td, patch.dict('os.environ', ENV), \
                patch('eviground.experiments.run_object_classification.classify_instance') as classify:
            args, _, _ = fixture(Path(td))
            classify.side_effect = lambda xyz, rgb, ids, oid, out, vocab, faces=None: {
                'bbox_id': oid, 'predicted_object': 'stool', 'status': 'completed'}
            with redirect_stdout(StringIO()) as stdout, redirect_stderr(StringIO()) as stderr:
                self.assertEqual(batch.run(args), 0)
                self.assertEqual(batch.run(args), 0)
                self.assertEqual(classify.call_count, 1)
                args.limit = 0
                self.assertEqual(batch.run(args), 0)
            self.assertEqual(classify.call_count, 2)
            lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
            self.assertEqual(len(lines), 2)
            self.assertEqual(set(lines[0]), {'bbox_id', 'target', 'predicted_object', 'correct'})
            self.assertFalse(lines[0]['correct'])
            self.assertIn('2/2 100%', stderr.getvalue())
            results = json.loads((args.output/'results.json').read_text())
            self.assertEqual(results['summary']['completed'], 2)
            self.assertEqual(results['summary']['correct'], 0)
            self.assertEqual(len(list(args.output.glob('scene*/object_*/evaluation.json'))), 2)

    def test_model_input_change_and_frozen_output_stop_before_requests(self):
        with tempfile.TemporaryDirectory() as td, patch.dict('os.environ', ENV), \
                patch('eviground.experiments.run_object_classification.classify_instance') as classify, \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            args, _, _ = fixture(Path(td))
            args.prepare_only = True
            batch.run(args)
            args.prepare_only = False
            with patch.dict('os.environ', {'EVIGROUND_MODEL': 'changed'}), \
                    self.assertRaisesRegex(ValueError, 'changed'):
                batch.run(args)
            path = args.scene_root/'scene0001_00/objects_gt.json'
            path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError, 'Scene inputs changed'):
                batch.run(args)
            (args.output/'FROZEN.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Frozen'):
                batch.run(args)
            classify.assert_not_called()

    def test_service_error_stops_scheduling_and_retry_archives_error(self):
        with tempfile.TemporaryDirectory() as td, patch.dict('os.environ', ENV), \
                patch('eviground.experiments.run_object_classification.classify_instance') as classify, \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            args, _, _ = fixture(Path(td))
            args.limit = 0
            classify.side_effect = QwenAPIError('quota', diagnostic={'http_status': 429})
            self.assertEqual(batch.run(args), 1)
            self.assertEqual(classify.call_count, 1)
            classify.side_effect = lambda xyz, rgb, ids, oid, out, vocab, faces=None: {
                'bbox_id': oid, 'predicted_object': 'chair', 'status': 'completed'}
            args.retry_errors = True
            self.assertEqual(batch.run(args), 0)
            self.assertEqual(len(list(args.output.glob('scene*/object_*/attempts/*/prediction.json'))), 1)
            summary = json.loads((args.output/'results.json').read_text())['summary']
            self.assertEqual(summary['correct'], 2)
            self.assertEqual(summary['errors'], 0)

    def test_early_service_stop_exports_unstarted_scenes_as_partial(self):
        with tempfile.TemporaryDirectory() as td, patch.dict('os.environ', ENV), \
                patch('eviground.experiments.run_object_classification.classify_instance') as classify, \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            args, _, _ = fixture(Path(td))
            shutil.copytree(args.scene_root/'scene0001_00', args.scene_root/'scene0002_00')
            args.manifest.write_text(json.dumps({'rows': [{'scene': name} for name in
                                                ['scene0001_00', 'scene0002_00']]}))
            args.limit = 0
            classify.side_effect = QwenAPIError('quota', diagnostic={'http_status': 429})
            self.assertEqual(batch.run(args), 1)
            tables = [json.loads(path.read_text()) for path in args.output.glob('scene*/predicted_objects.json')]
            self.assertEqual(len(tables), 2)
            self.assertTrue(all(table['complete'] is False for table in tables))
            self.assertEqual(sum(len(table['objects']) for table in tables), 1)

    def test_lexical_scoring_does_not_merge_subtypes(self):
        for a, b in [('books', 'book'), ('shoes', 'shoe'), ('shelves', 'shelf'),
                     ('office-chairs', 'office chair'), ('sofa', 'couch')]:
            self.assertEqual(classifier.normalized_category(a), classifier.normalized_category(b))
        for a, b in [('office chair', 'chair'), ('table', 'desk'), ('shelf', 'bookshelf')]:
            self.assertNotEqual(classifier.normalized_category(a), classifier.normalized_category(b))


if __name__ == '__main__':
    unittest.main()
