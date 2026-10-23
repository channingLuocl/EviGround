"""CPU verification of actual head training plus input/supervision isolation."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from eviground.categories.category_common_v82 import ROOT, file_hash
from eviground.classifiers.evidence_data_v83 import (VERSION, check_disjoint, crop_view, load_part_masks,
                               local_patches, state_targets)
from eviground.classifiers.evidence_head_v83 import EvidenceHead, training_loss
from eviground.classifiers.pointcloud_classifier_v82 import CLIP_REVISION, prepare_points
from eviground.classifiers.pointcloud_classifier_v83 import FrozenFeatures, load_head
from eviground.experiments.run_categories_v83 import run
from eviground.experiments.train_categories_v83 import cache_features, make_manifest, train


class FakeFrozenFeatures:
    """Replace only unavailable CUDA encoder; the trainable heads run real PyTorch."""
    def __init__(self, checkpoint, category_text):
        self.metadata = {'test_encoder': True}

    def extract(self, xyz, seed):
        rng = np.random.RandomState(seed)
        return {'global_features': rng.normal(size=512).astype(np.float32),
                'local_features': rng.normal(size=(8, 512)).astype(np.float32),
                'geometry': local_patches(xyz, seed)[1]}


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_unlabelled_and_unknown_are_distinct_and_crop_never_invents_absence(self):
        np.testing.assert_array_equal(state_targets({'seat': 'present', 'leg': 'unknown'},
                                                    ['seat', 'leg', 'backrest']), [0, 2, -1])
        xyz = np.random.RandomState(4).normal(size=(100, 3))
        states = np.array([0, 1, 2, -1])
        masks = np.zeros((100, 4), dtype=bool)
        # Choose observed support exclusively among points that this crop will remove.
        keep, _ = crop_view(xyz, states, masks, 3)
        masks[np.setdiff1d(np.arange(100), keep), 0] = True
        keep2, targets = crop_view(xyz, states, masks, 3)
        np.testing.assert_array_equal(keep, keep2)
        np.testing.assert_array_equal(targets, [2, 2, 2, -1])

    def test_part_masks_do_not_turn_missing_annotation_into_absence(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'parts.npz'
            np.savez(path, parts=np.array(['seat']), masks=np.ones((3, 1), dtype=bool))
            with self.assertRaisesRegex(ValueError, 'present annotation'):
                load_part_masks(path, 3, ['seat'], np.array([-1]))

    def test_patch_extraction_is_deterministic_and_scale_translation_invariant(self):
        xyz = np.random.RandomState(1).normal(size=(400, 3))
        indices, geometry = local_patches(xyz, 9)
        indices2, geometry2 = local_patches(xyz*4 + [10, -4, 8], 9)
        for first, second in zip(indices, indices2):
            np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(geometry, geometry2, atol=1e-6)

    def test_unknown_neutral_and_signed_evidence(self):
        model = EvidenceHead(torch.randn(5, 512), torch.randn(1, 512))
        with torch.no_grad():
            model.part_relation.weight.fill_(2.)
        base = torch.randn(1, 5)
        torch.testing.assert_close(model.combine_evidence(base, torch.tensor([[[0., 0., 1.]]])), base)
        torch.testing.assert_close(model.combine_evidence(base, torch.tensor([[[1., 0., 0.]]])), base+2)
        torch.testing.assert_close(model.combine_evidence(base, torch.tensor([[[0., 1., 0.]]])), base-2)

    def test_frozen_extraction_preserves_global_preprocessing_and_encodes_eight_crops(self):
        class CPUEncoder:
            def __init__(self):
                self.inputs = []
            def __call__(self, value):
                self.inputs.append(value.clone())
                return value.mean(dim=1).repeat(1, 128)
        encoder = CPUEncoder()
        base = argparse.Namespace(torch=torch, encoder=encoder, metadata={'CPU_test_only': True})
        xyz = np.random.RandomState(7).normal(size=(300, 3))
        with patch('eviground.classifiers.pointcloud_classifier_v83.PointCloudClassifier', return_value=base), \
             patch.object(torch.Tensor, 'cuda', lambda self: self):
            extractor = FrozenFeatures('unused.pth', 'unused.npz')
            features = extractor.extract(xyz, 42)
        self.assertEqual(len(encoder.inputs), 9)
        np.testing.assert_array_equal(encoder.inputs[0][0].numpy(), prepare_points(xyz, None, 'pointnext', 42))
        self.assertEqual(features['local_features'].shape, (8, 512))
        self.assertTrue(np.isfinite(features['local_features']).all())
        np.testing.assert_allclose(np.linalg.norm(features['local_features'], axis=1), 1., atol=1e-6)

    def test_all_ablations_backprop_and_masked_annotation_loss(self):
        features = (torch.randn(4, 512), torch.randn(4, 8, 512), torch.randn(4, 8, 7))
        for mode in ('global', 'attention', 'voting', 'evidence'):
            model = EvidenceHead(torch.randn(5, 512), torch.randn(2, 512), mode)
            optimizer = torch.optim.Adam(model.parameters(), lr=.005)
            labels, states = torch.tensor([0, 1, 2, 3]), torch.full((4, 2), -1)
            before = training_loss(model(*features), labels, states)[0].item()
            for _ in range(5):
                optimizer.zero_grad()
                loss, _, state_loss = training_loss(model(*features), labels, states)
                self.assertEqual(state_loss.item(), 0)
                loss.backward()
                optimizer.step()
            after = training_loss(model(*features), labels, states)[0].item()
            self.assertLess(after, before, mode)
            if mode != 'global':
                explicit_unknown = torch.full((4, 2), 2)
                _, _, auxiliary = training_loss(model(*features), labels, explicit_unknown)
                self.assertGreater(auxiliary.item(), 0)

    def fixture(self, root):
        schema = root/'parts.json'
        schema.write_text(json.dumps({'format': 'v8.3-parts-v1', 'parts': ['backrest'],
                                      'prompt': 'the {} of an object'}))
        text = root/'part_text.npz'
        np.savez(text, embeddings=np.random.RandomState(7).normal(size=(1, 512)).astype(np.float32),
                 metadata=json.dumps({'parts': ['backrest'], 'prompt': 'the {} of an object',
                                      'text_model': 'openai/clip-vit-base-patch16', 'text_revision': CLIP_REVISION}))
        backbone = root/'backbone.pth'
        backbone.write_bytes(b'fake CUDA backbone for test only')
        manifests = []
        for split in ('train', 'dev'):
            rows = []
            for i, state in enumerate(('present', 'absent', 'unknown')):
                scene = f'{split}_scene_{i}'
                directory = root/scene
                directory.mkdir()
                xyz = np.random.RandomState(i + (0 if split == 'train' else 10)).normal(size=(80, 3))
                np.savez(directory/'scene.npz', xyz=xyz, ids=np.zeros(80, dtype=np.int64),
                         forbidden=np.array([{'target': 'do not load'}], dtype=object))
                rows.append({'scene': scene, 'bbox_id': 0, 'cloud': f'{scene}/scene.npz',
                             'category': 'chair', 'part_states': {'backrest': state}})
            manifest = root/f'{split}.json'
            manifest.write_text(json.dumps({'format': 'v8.3-training-manifest-v1', 'split': split,
                                           'annotation_protocol': 'Synthetic test only; not real annotations.',
                                           'samples': rows}))
            manifests.append(manifest)
        return schema, text, backbone, manifests

    def test_cache_train_reload_and_label_free_inference_resume(self):
        with tempfile.TemporaryDirectory() as td, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(td)
            schema, part_text, backbone, manifests = self.fixture(root)
            for split, manifest in zip(('train', 'dev'), manifests):
                args = argparse.Namespace(parts=schema, manifest=manifest, crop_views=0,
                                          checkpoint=backbone, text_features=ROOT/'weights/v82_clip_text_features.npz',
                                          output=root/split, seed=42)
                with patch('eviground.experiments.train_categories_v83.FrozenFeatures', FakeFrozenFeatures):
                    cache_features(args)
            args = argparse.Namespace(train_cache=root/'train/features.npz', dev_cache=root/'dev/features.npz',
                                      text_features=ROOT/'weights/v82_clip_text_features.npz',
                                      part_text_features=part_text, mode='evidence', seed=42, epochs=3,
                                      batch_size=2, lr=.01, output=root/'model')
            train(args)
            # Refuse an evidence head with no verified absence/unknown supervision.
            with np.load(args.train_cache, allow_pickle=False) as data:
                incomplete = {k: data[k] for k in data.files}
            incomplete['states'] = np.zeros_like(incomplete['states'])
            np.savez(root/'incomplete.npz', **incomplete)
            invalid = argparse.Namespace(**vars(args))
            invalid.train_cache, invalid.output = root/'incomplete.npz', root/'invalid_head'
            with self.assertRaisesRegex(ValueError, 'state coverage'):
                train(invalid)
            self.assertFalse(invalid.output.exists())
            head, record = load_head(root/'model/head.pt')
            self.assertTrue(record['trained'])
            self.assertTrue(all(torch.isfinite(v).all() for v in head.state_dict().values()))
            # One split's scene cannot become an independent test merely through a new file path.
            overlap_args = argparse.Namespace(method='supervised-experimental', cloud=root/'dev_scene_0/scene.npz', bbox_id=[0], limit=0,
                manifest=None, scene_root=root, seed=42, output=root/'overlap', prepare_only=True,
                head_checkpoint=root/'model/head.pt', checkpoint=backbone,
                text_features=ROOT/'weights/v82_clip_text_features.npz', retry_errors=False,
                allow_development_evaluation=False)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                run(overlap_args)
            directory = root/'heldout_scene'
            directory.mkdir()
            np.savez(directory/'scene.npz', xyz=np.random.RandomState(99).normal(size=(90, 3)),
                     ids=np.zeros(90, dtype=np.int64))
            overlap_args.cloud, overlap_args.output, overlap_args.prepare_only = directory/'scene.npz', root/'test_run', False
            with patch('eviground.classifiers.pointcloud_classifier_v83.FrozenFeatures', FakeFrozenFeatures), \
                 patch('eviground.classifiers.pointcloud_classifier_v83.POINTNEXT_SHA256', file_hash(backbone)):
                self.assertEqual(run(overlap_args), 0)
            prediction = overlap_args.output/'heldout_scene/object_0000/prediction.json'
            saved = prediction.read_bytes()
            value = json.loads(saved)
            self.assertEqual(len(value['top5']), 5)
            self.assertEqual(len(value['scores']), 197)
            self.assertIn('unknown', value['part_evidence']['backrest'])
            self.assertNotIn('target', value)
            self.assertFalse((overlap_args.output/'results.json').exists())
            with patch('eviground.experiments.run_categories_v83.PointCloudClassifierV83') as factory:
                self.assertEqual(run(overlap_args), 0)
                factory.assert_not_called()
            self.assertEqual(saved, prediction.read_bytes())
            # Error retries archive the old error and never retry a successful wrong answer.
            overlap_args.output = root/'retry_run'
            with patch('eviground.experiments.run_categories_v83.PointCloudClassifierV83') as factory:
                factory.return_value.metadata = {'test': True}
                factory.return_value.predict.side_effect = ValueError('test data failure')
                self.assertEqual(run(overlap_args), 1)
            error_path = overlap_args.output/'heldout_scene/object_0000/prediction.json'
            error = error_path.read_bytes()
            overlap_args.retry_errors = True
            with patch('eviground.experiments.run_categories_v83.PointCloudClassifierV83') as factory:
                factory.return_value.metadata = {'test': True}
                factory.return_value.predict.return_value = {'status': 'completed', 'predicted_object': 'chair'}
                self.assertEqual(run(overlap_args), 0)
                self.assertEqual(factory.return_value.predict.call_count, 1)
            self.assertEqual(next(error_path.parent.glob('attempts/*/prediction.json')).read_bytes(), error)
            with patch('eviground.experiments.run_categories_v83.PointCloudClassifierV83') as factory:
                self.assertEqual(run(overlap_args), 0)
                factory.assert_not_called()
            (overlap_args.output/'FROZEN.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'read-only'):
                run(overlap_args)
            overlap_args.output = root/'test_run'
            backbone.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                run(overlap_args)

    def test_overlap_detects_renamed_duplicate_clouds_and_untrained_head_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            check_disjoint([{'scene': 'a', 'cloud_sha256': 'same'}],
                           [{'scene': 'b', 'cloud_sha256': 'same'}])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            check_disjoint([{'scene': 'scene0001_00', 'cloud_sha256': 'first'}],
                           [{'scene': 'scene0001_01', 'cloud_sha256': 'second'}])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'head.pt'
            torch.save({'version': VERSION, 'trained': False}, path)
            with self.assertRaisesRegex(ValueError, 'trained'):
                load_head(path)

    def test_manifest_builder_uses_designated_supervision_not_author_predictions(self):
        with tempfile.TemporaryDirectory() as td, redirect_stdout(io.StringIO()):
            root = Path(td)
            _, _, _, _ = self.fixture(root)
            directory = root/'train_scene_0'
            (directory/'objects.json').write_text(json.dumps([{'bbox_id': 0, 'target': 'do not use'}]))
            (directory/'objects_gt.json').write_text(json.dumps([{'bbox_id': 0, 'target': 'chair'}]))
            scenes = root/'scenes.json'
            scenes.write_text(json.dumps({'rows': [{'scene': 'train_scene_0'}]}))
            output = root/'manifest.json'
            make_manifest(argparse.Namespace(scenes=scenes, scene_root=root, split='train', output=output))
            row = json.loads(output.read_text())['samples'][0]
            self.assertEqual(row['category'], 'chair')
            self.assertEqual(row['part_states'], {})


if __name__ == '__main__':
    unittest.main()
