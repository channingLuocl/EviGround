"""Offline tests for v8.2 data isolation and deterministic preprocessing."""
import json
import argparse
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from category_common_v82 import category_vocabulary, load_cloud, normalized_category
from pointcloud_classifier_v82 import prepare_points, load_text_features, PROMPT, CLIP_REVISION
from run_categories_v82 import run


class V82Tests(unittest.TestCase):
    def test_offline_text_features_reject_wrong_semantic_space(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'text.npz'
            metadata = dict(vocabulary=list(category_vocabulary()), prompt=PROMPT,
                            text_model='openai/clip-vit-base-patch16', text_revision=CLIP_REVISION)
            np.savez(path, embeddings=np.ones((197, 512), dtype=np.float32), metadata=json.dumps(metadata))
            self.assertEqual(load_text_features(path)[0].shape, (197, 512))
            metadata['text_revision'] = 'wrong-model-version'
            np.savez(path, embeddings=np.ones((197, 512), dtype=np.float32), metadata=json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'text space/vocabulary'):
                load_text_features(path)

    def test_cloud_loader_reads_only_numeric_arrays(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'scene.npz'
            np.savez(path, xyz=np.zeros((3, 3)), rgb=np.full((3, 3), 128, dtype=np.uint8),
                     ids=np.array([0, 0, 1]), secret=np.array(['gt-chair']))
            xyz, rgb, ids = load_cloud(path)
            self.assertEqual(xyz.shape, (3, 3))
            self.assertEqual(rgb.dtype, np.uint8)
            np.testing.assert_array_equal(ids, [0, 0, 1])

    def test_preprocessing_shapes_and_repeatability(self):
        xyz = np.arange(30, dtype=np.float64).reshape(10, 3)
        rgb = np.full((10, 3), 255, dtype=np.uint8)
        np.testing.assert_array_equal(prepare_points(xyz, rgb, 'pointnext', 4),
                                      prepare_points(xyz, rgb, 'pointnext', 4))
        self.assertEqual(prepare_points(xyz, rgb, 'pointnext', 4).shape, (2048, 4))
        self.assertEqual(prepare_points(xyz, rgb, 'openshape', 4).shape, (6, 10000))

    def test_vocabulary_and_matching_do_not_collapse_subtypes(self):
        self.assertEqual(len(category_vocabulary()), 197)
        self.assertEqual(normalized_category('books'), 'book')
        self.assertEqual(normalized_category('sofa'), 'couch')
        self.assertNotEqual(normalized_category('office chair'), normalized_category('chair'))

    def fixture(self, directory):
        root = Path(directory)
        scene = root/'scene_test'
        scene.mkdir()
        xyz = np.array([[0., 0, 0], [1, 0, 0], [0, 0, 1],
                        [3, 0, 0], [4, 0, 0], [3, 0, 1]])
        np.savez(scene/'scene.npz', xyz=xyz, ids=np.array([0, 0, 0, 1, 1, 1]),
                 hidden_label=np.array([{'target': 'do not read'}], dtype=object))
        (scene/'objects.json').write_text(json.dumps([
            {'bbox_id': i, 'target': 'author label must be ignored'} for i in (0, 1)]))
        (scene/'objects_gt.json').write_text(json.dumps([
            {'bbox_id': 0, 'target': 'books'}, {'bbox_id': 1, 'target': 'office chair'}]))
        manifest = root/'manifest.json'
        manifest.write_text(json.dumps({'rows': [{'scene': scene.name}]}))
        return argparse.Namespace(cloud=None, manifest=manifest, scene_root=root,
                                  bbox_id=None, seed=42, limit=0, output=root/'results',
                                  backend='pointnext', checkpoint=root/'weights.pth',
                                  text_features=None,
                                  retry_errors=False, prepare_only=False)

    def test_prediction_evaluation_isolation_and_resume(self):
        with tempfile.TemporaryDirectory() as td, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            args = self.fixture(td)
            with patch('run_categories_v82.PointCloudClassifier') as factory:
                model = factory.return_value
                model.metadata = {'test': True}
                model.predict.side_effect = lambda xyz, rgb, seed: {
                    'predicted_object': 'book' if xyz[0, 0] == 0 else 'chair',
                    'status': 'completed', 'top5': [{'category': 'book'}, {'category': 'chair'}]}
                self.assertEqual(run(args), 0)
                self.assertEqual(model.predict.call_count, 2)
                for call in model.predict.call_args_list:
                    xyz, rgb, seed = call.args
                    self.assertEqual(xyz.shape, (3, 3))
                    self.assertIsNone(rgb)
                    self.assertIsInstance(seed, int)
            paths = sorted(args.output.glob('scene_test/object_*/prediction.json'))
            before = {p: p.read_bytes() for p in paths}
            for value in before.values():
                self.assertNotIn('target', json.loads(value))
                self.assertNotIn(b'author label', value)
            report = json.loads((args.output/'results.json').read_text())['summary']
            self.assertEqual(report['correct'], 1)  # books == book, office chair != chair
            self.assertEqual(report['exact_correct'], 0)
            with patch('run_categories_v82.PointCloudClassifier') as factory:
                self.assertEqual(run(args), 0)
                factory.assert_not_called()
            self.assertEqual(before, {p: p.read_bytes() for p in paths})
            args.checkpoint.write_bytes(b'changed checkpoint')
            with self.assertRaisesRegex(ValueError, 'checkpoint differs'):
                run(args)
            args.checkpoint.unlink()
            self.assertEqual(before, {p: p.read_bytes() for p in paths})
            with np.load(args.scene_root/'scene_test/scene.npz', allow_pickle=False) as cloud:
                np.savez(args.scene_root/'scene_test/scene.npz', xyz=cloud['xyz']+1, ids=cloud['ids'])
            with self.assertRaisesRegex(ValueError, 'Scene inputs changed'):
                run(args)
            self.assertEqual(before, {p: p.read_bytes() for p in paths})

    def test_retry_preserves_error_and_does_not_retry_wrong_answers(self):
        with tempfile.TemporaryDirectory() as td, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            args = self.fixture(td)
            with patch('run_categories_v82.PointCloudClassifier') as factory:
                model = factory.return_value
                model.metadata = {'test': True}
                model.predict.side_effect = [ValueError('invalid instance'),
                                            {'status': 'completed', 'predicted_object': 'wrong'}]
                self.assertEqual(run(args), 1)
            error_path = next(p for p in args.output.glob('scene_test/object_*/prediction.json')
                              if json.loads(p.read_text())['status'] == 'error')
            error_bytes = error_path.read_bytes()
            args.retry_errors = True
            with patch('run_categories_v82.PointCloudClassifier') as factory:
                model = factory.return_value
                model.metadata = {'test': True}
                model.predict.return_value = {'status': 'completed', 'predicted_object': 'book'}
                self.assertEqual(run(args), 0)
                self.assertEqual(model.predict.call_count, 1)
            archived = list(error_path.parent.glob('attempts/*/prediction.json'))
            self.assertEqual(len(archived), 1)
            self.assertEqual(archived[0].read_bytes(), error_bytes)

    def test_cloud_mode_needs_no_label_files_and_frozen_outputs_are_protected(self):
        with tempfile.TemporaryDirectory() as td, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            args = self.fixture(td)
            args.cloud = args.scene_root/'scene_test/scene.npz'
            for path in args.cloud.parent.glob('objects*.json'):
                path.unlink()
            args.manifest.unlink()
            args.manifest = None
            args.bbox_id = [1]
            with patch('run_categories_v82.PointCloudClassifier') as factory:
                model = factory.return_value
                model.metadata = {'test': True}
                model.predict.return_value = {'status': 'completed', 'predicted_object': 'chair'}
                self.assertEqual(run(args), 0)
                self.assertEqual(model.predict.call_count, 1)
            self.assertFalse((args.output/'results.json').exists())
            (args.output/'FROZEN.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Frozen run is read-only'):
                run(args)


if __name__ == '__main__':
    unittest.main()
