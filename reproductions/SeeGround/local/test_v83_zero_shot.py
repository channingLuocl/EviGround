"""Training-free score fusion and label/head isolation; no CUDA required."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from category_common_v82 import ROOT, file_hash
from pointcloud_zero_shot_v83 import LOCAL_WEIGHT, fuse_zero_shot
from run_categories_v83 import run


def features(global_vector, locals):
    geometry = np.zeros((len(locals), 7), dtype=np.float32)
    geometry[:, 0] = np.arange(len(locals))*.1
    geometry[:, -1] = .25
    return {'global_features': np.asarray(global_vector, dtype=np.float32),
            'local_features': np.asarray(locals, dtype=np.float32), 'geometry': geometry}


class ZeroShotTests(unittest.TestCase):
    def test_stable_positive_support_can_rerank_without_any_annotation(self):
        first = features([.72, .69, 0], [[0, 1, 0]]*8)
        result = fuse_zero_shot(first, first, np.eye(3))
        self.assertEqual(result['baseline_scores'].argmax(), 0)
        self.assertEqual(result['scores'].argmax(), 1)
        self.assertTrue((result['local_support'] >= 0).all())
        np.testing.assert_allclose(result['scores'] - result['global_scores'],
                                   LOCAL_WEIGHT*result['local_support'])

    def test_disagreement_and_missing_support_do_not_create_negative_evidence(self):
        first = features([.72, .69, 0], [[1, 0, 0]]*8)
        second = features([.72, .69, 0], [[0, 1, 0]]*8)
        result = fuse_zero_shot(first, second, np.eye(3))
        np.testing.assert_array_equal(result['local_support'], [0, 0, 0])
        np.testing.assert_array_equal(result['scores'], result['global_scores'])
        self.assertFalse(result['patch_agreement'].any())

    def test_duplicates_full_object_crops_and_single_vote_are_not_multiple_supports(self):
        first = features([.72, .69, 0], [[0, 1, 0]]*8)
        first['geometry'][:] = first['geometry'][0]
        result = fuse_zero_shot(first, first, np.eye(3))
        self.assertEqual(result['support_counts'][1], 1)
        self.assertEqual(result['local_support'][1], 0)
        first['geometry'][:, -1] = 1.
        result = fuse_zero_shot(first, first, np.eye(3))
        self.assertEqual(result['support_counts'].sum(), 0)

    def test_locals_cannot_promote_a_class_outside_global_candidates(self):
        first = features([7, 6, 5, 4, 3, 2, 1], [[0, 0, 0, 0, 0, 0, 1]]*8)
        result = fuse_zero_shot(first, first, np.eye(7))
        self.assertEqual(result['support_counts'][6], 8)
        np.testing.assert_array_equal(result['local_support'], np.zeros(7))
        np.testing.assert_array_equal(result['scores'], result['global_scores'])

    def test_sampling_changes_do_not_change_patch_geometry(self):
        # Exercise the actual paired extractor with only its CUDA encoder replaced.
        import torch
        from pointcloud_classifier_v83 import FrozenFeatures
        base = argparse.Namespace(torch=torch, metadata={'test': True},
            encoder=lambda x: x.mean(1).repeat(1, 128))
        xyz = np.random.RandomState(18).normal(size=(300, 3))
        with patch('pointcloud_classifier_v83.PointCloudClassifier', return_value=base), \
             patch.object(torch.Tensor, 'cuda', lambda self: self):
            extractor = FrozenFeatures('unused', 'unused')
            first = extractor.extract(xyz, 42)
            second = extractor.extract(xyz, 42, sampling_offset=104729)
        np.testing.assert_array_equal(first['geometry'], second['geometry'])
        self.assertFalse(np.array_equal(first['global_features'], second['global_features']))

    def test_default_runner_needs_no_head_annotations_or_label_files_and_resumes(self):
        with tempfile.TemporaryDirectory() as td, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(td)
            cloud, checkpoint = root/'cloud.npz', root/'backbone.pth'
            checkpoint.write_bytes(b'test fixture')
            np.savez(cloud, xyz=np.random.RandomState(8).normal(size=(100, 3)), ids=np.zeros(100, dtype=int),
                     category_labels=np.array([{'label': 'forbidden'}], dtype=object))
            args = argparse.Namespace(method='zero-shot', cloud=cloud, manifest=None, scene_root=root,
                bbox_id=[0], seed=42, limit=0, head_checkpoint=None, checkpoint=checkpoint,
                text_features=ROOT/'weights/v82_clip_text_features.npz', output=root/'run',
                prepare_only=False, retry_errors=False, allow_development_evaluation=False)
            with patch('run_categories_v83.POINTNEXT_SHA256', file_hash(checkpoint)), \
                 patch('run_categories_v83.load_head', side_effect=AssertionError('must not load supervised head')), \
                 patch('run_categories_v83.ZeroShotClassifierV83') as factory:
                factory.return_value.metadata = {'learning_regime': 'training_free'}
                factory.return_value.predict.return_value = {'status': 'completed', 'predicted_object': 'chair',
                    'top5': [{'category': 'chair'}], 'scores': [0.]*197}
                self.assertEqual(run(args), 0)
                points, colors, seed = factory.return_value.predict.call_args.args
                self.assertEqual(points.shape, (100, 3))
                self.assertIsNone(colors)
                self.assertIsInstance(seed, int)
                path = root/'run/cloud/object_0000/prediction.json'
                saved = path.read_bytes()
                self.assertNotIn('target', json.loads(saved))
                self.assertNotIn(b'forbidden', saved)
                self.assertFalse((root/'run/results.json').exists())
                self.assertEqual(json.loads((root/'run/config.json').read_text())['head_sha256'], None)
                factory.reset_mock()
                self.assertEqual(run(args), 0)
                factory.assert_not_called()
                self.assertEqual(saved, path.read_bytes())
                args.head_checkpoint = root/'do_not_use.pt'
                with self.assertRaisesRegex(ValueError, 'does not use'):
                    run(args)


if __name__ == '__main__':
    unittest.main()
