"""CPU checks for Uni3D input compatibility and text/checkpoint isolation."""
import json
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from seeground.categories.category_common_v82 import category_vocabulary
from seeground.classifiers.pointcloud_classifier_v82 import PROMPT, prepare_points
from seeground.experiments.run_categories_v82 import run
from seeground.classifiers.uni3d_classifier import (TEXT_MODEL, TEXT_REVISION, TEXT_SHA256,
                              load_uni3d_text, point_encoder_state, prepare_uni3d_points)


class Uni3DTests(unittest.TestCase):
    def test_input_matches_existing_sampling_and_keeps_rgb_aligned(self):
        xyz = np.arange(60).reshape(20, 3)
        rgb = np.arange(60, dtype=np.uint8).reshape(20, 3)
        actual = prepare_uni3d_points(xyz, rgb, 42)
        self.assertEqual(actual.shape, (10000, 6))
        self.assertTrue(actual.flags.c_contiguous)
        self.assertEqual(actual.dtype, np.float32)
        original = prepare_points(xyz, rgb, 'openshape', 42).T
        expected = original.copy()
        expected[:, :3] = original[:, [0, 2, 1]]
        expected[:, 2] *= -1
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[:, 1], original[:, 2])
        np.testing.assert_allclose(np.linalg.norm(actual[:, :3], axis=1),
                                   np.linalg.norm(original[:, :3], axis=1), rtol=1e-6)
        rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        self.assertEqual(np.linalg.det(rotation), 1)
        np.testing.assert_array_equal(original[:, :3] @ rotation, actual[:, :3])
        indices = np.random.RandomState(42).choice(20, 10000, replace=True)
        np.testing.assert_allclose(actual[:, 3:], rgb[indices]/255., rtol=1e-6)
        np.testing.assert_allclose(prepare_uni3d_points(xyz, None, 42)[:, 3:], .4)

    def test_rejects_other_text_towers_prompts_and_vocabulary(self):
        metadata = dict(vocabulary=list(category_vocabulary()), prompt=PROMPT,
                        text_model=TEXT_MODEL, text_revision=TEXT_REVISION,
                        text_checkpoint_sha256=TEXT_SHA256)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'text.npz'
            embeddings = np.ones((197, 1024), dtype=np.float32)
            np.savez(path, embeddings=embeddings, metadata=json.dumps(metadata))
            self.assertEqual(load_uni3d_text(path)[0].shape, (197, 1024))
            for key, value in [('text_model', 'ViT-bigG-14'), ('text_revision', 'wrong'),
                               ('prompt', 'a photo of a {}'), ('text_checkpoint_sha256', 'wrong'),
                               ('vocabulary', metadata['vocabulary'][::-1])]:
                with self.subTest(key=key):
                    np.savez(path, embeddings=embeddings, metadata=json.dumps({**metadata, key: value}))
                    with self.assertRaises(ValueError):
                        load_uni3d_text(path)
            for bad in [np.zeros((197, 1024)), np.ones((197, 512)),
                        np.full((197, 1024), np.nan)]:
                np.savez(path, embeddings=bad, metadata=json.dumps(metadata))
                with self.assertRaises(ValueError):
                    load_uni3d_text(path)

    def test_extracts_only_official_point_encoder_state(self):
        for prefix in ['', 'module.']:
            state = {prefix+'point_encoder.encoder.weight': 'weight', prefix+'logit_scale': 1.}
            self.assertEqual(point_encoder_state({'module': state}), {'encoder.weight': 'weight'})
        for state in [{'logit_scale': 1.}, {'point_encoder.x': 1., 'text.weight': 2.}]:
            with self.assertRaises(ValueError):
                point_encoder_state({'module': state})

    def test_shared_evaluator_accepts_injected_classifier_and_records_provenance(self):
        from tests.legacy.test_v82_pointcloud import V82Tests
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            args = V82Tests().fixture(directory)
            args.backend = 'uni3d'
            classifier = Mock()
            classifier.metadata = {'backend': 'uni3d', 'test': True}
            classifier.predict.return_value = {'status': 'completed', 'predicted_object': 'book',
                                               'top5': [{'category': 'book'}]}
            factory = Mock(return_value=classifier)
            with patch('seeground.experiments.run_categories_v82.PointCloudClassifier') as default:
                self.assertEqual(run(args, classifier_factory=factory,
                                     extra_config={'source_revision': 'test'}), 0)
                default.assert_not_called()
            self.assertEqual(classifier.predict.call_count, 2)
            config = json.loads((args.output/'config.json').read_text())
            self.assertEqual(config['extension'], {'source_revision': 'test'})
            summary = json.loads((args.output/'results.json').read_text())['summary']
            self.assertEqual((summary['completed'], summary['errors'], summary['correct']), (2, 0, 1))


if __name__ == '__main__':
    unittest.main()
