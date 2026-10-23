"""CPU checks for learning, room isolation and cache/train/predict interoperability."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from category_common_v82 import ROOT
from category_v10.data import (assert_disjoint, convert_scene, make_manifest, prepare_manifests,
                              split_training)
from category_v10.model import CategoryHead, adaptation_loss
from category_v10.pipeline import download
from category_v10.run import cache, code_hashes, metrics, predict, read_cache, train
from run_storage import write_json


class FakeExtractor:
    """Only CUDA encoding is replaced; the head, optimizer, files and scoring are real."""
    def __init__(self, *_):
        self.metadata = {'synthetic_test_encoder': True}

    def extract(self, points, seed):
        rng = np.random.RandomState(seed)
        return {'global_features': rng.normal(size=512).astype(np.float32),
                'local_features': rng.normal(size=(8, 512)).astype(np.float32),
                'geometry': rng.normal(size=(8, 7)).astype(np.float32)}


class CategoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_physical_rooms_and_official_validation_never_enter_training(self):
        scenes = ['scene0001_00', 'scene0001_01', 'scene0002_00', 'scene0003_00']
        first = split_training(scenes, .4, 42)
        self.assertEqual(first, split_training(list(reversed(scenes)), .4, 42))
        assert_disjoint(*first)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            assert_disjoint(['scene0001_00'], ['scene0001_01'])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/'train.txt').write_text('scene0001_00\nscene0002_00\n')
            (root/'val.txt').write_text('scene0001_01\n')
            args = argparse.Namespace(train_scenes=root/'train.txt', validation_scenes=root/'val.txt',
                eval_scenes=root/'val.txt', dev_fraction=.1, seed=42, scene_root=root, output=root/'out')
            with self.assertRaisesRegex(ValueError, 'overlap'):
                prepare_manifests(args)
            self.assertFalse(args.output.exists())

    def test_conditioning_changes_patch_selection_and_controls_match_parameter_budget(self):
        text = torch.randn(9, 512)
        ordinary = CategoryHead(text, 'attention')
        conditioned = CategoryHead(text, 'conditioned')
        self.assertEqual(sum(p.numel() for p in ordinary.parameters()),
                         sum(p.numel() for p in conditioned.parameters()))
        features = (torch.randn(3, 512), torch.randn(3, 8, 512), torch.randn(3, 8, 7))
        # Ordinary attention has a shared spatial distribution; conditioned queries differ by class.
        weights = ordinary(*features)['patch_attention']
        self.assertEqual(weights.shape, (3, 8, 1))
        out = conditioned(*features)
        self.assertFalse(torch.allclose(out['patch_attention'][..., 0], out['patch_attention'][..., 1]))
        mask = torch.tensor([[True, False, False, False, False, False, False, False]]).expand(3, -1)
        masked = conditioned(*features, patch_mask=mask)
        self.assertEqual(int(torch.count_nonzero(masked['patch_attention'][:, 1:])), 0)
        # Even the initial globally last-ranked class can receive a local correction.
        last = out['global_logits'].argmin(-1)
        correction = out['logits'] - out['global_logits']
        self.assertTrue((correction.gather(1, last[:, None]).abs() > 0).all())

    def test_real_training_reduces_loss_and_local_branch_gets_gradients(self):
        features = (torch.randn(12, 512), torch.randn(12, 8, 512), torch.randn(12, 8, 7))
        labels = torch.arange(12) % 6
        for mode in ('global', 'attention', 'conditioned'):
            model = CategoryHead(torch.randn(6, 512), mode)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            initial = float(adaptation_loss(model(*features), model(*features), labels).detach())
            for _ in range(25):
                optimizer.zero_grad()
                loss = adaptation_loss(model(*features), model(*features), labels)
                loss.backward()
                if mode != 'global':
                    self.assertGreater(float(model.key.weight.grad.abs().sum()), 0)
                optimizer.step()
            final = float(adaptation_loss(model(*features), model(*features), labels).detach())
            self.assertLess(final, initial*.5, mode)

    def fixture(self, root):
        manifests = []
        for index, split in enumerate(('train', 'dev', 'eval')):
            scene = f'scene{index+1:04d}_00'
            directory = root/scene
            directory.mkdir()
            xyz = np.random.RandomState(index).normal(size=(120, 3)).astype(np.float32)
            # An object array would fail if the extractor tried to read unrelated GT members.
            np.savez(directory/'scene.npz', xyz=xyz, ids=np.repeat([0, 1, 2], 40),
                     forbidden=np.array([{'target': 'must_not_read'}], dtype=object))
            write_json(directory/'objects_gt.json', [{'bbox_id': 0, 'target': 'chair'},
                       {'bbox_id': 1, 'target': 'table'}, {'bbox_id': 2, 'target': 'unlisted test category'}])
            value = make_manifest(root, [scene], split)
            path = root/f'{split}.json'
            write_json(path, value)
            manifests.append(path)
        return manifests

    def test_instance_dedup_and_oov_accounting(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = self.fixture(root)
            training = json.loads(paths[0].read_text())
            self.assertEqual(len(training['samples']), 2)
            self.assertEqual(len(training['excluded_training_instances']), 1)
            evaluation = json.loads(paths[2].read_text())
            self.assertEqual(len(evaluation['samples']), 3)
            labels = torch.tensor([0, 1, -1])
            result = metrics(torch.eye(3), labels)
            self.assertAlmostEqual(result['accuracy'], 2/3, places=6)
            self.assertEqual(result['out_of_vocabulary'], 1)
            annotation = root/'scene0001_00/objects_gt.json'
            rows = json.loads(annotation.read_text())
            write_json(annotation, rows + [rows[0]])
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                make_manifest(root, ['scene0001_00'], 'train')

    def test_cache_training_reload_prediction_and_leakage_rejection(self):
        with tempfile.TemporaryDirectory() as td, redirect_stdout(io.StringIO()):
            root = Path(td)
            paths = self.fixture(root)
            for split, manifest in zip(('train', 'dev', 'eval'), paths):
                args = argparse.Namespace(manifest=manifest, checkpoint='unused',
                    text_features=ROOT/'weights/v82_clip_text_features.npz', seed=42, output=root/split)
                with patch('pointcloud_classifier_v83.FrozenFeatures', FakeExtractor):
                    cache(args)
                before = (args.output/'features.npz').read_bytes()
                with patch('pointcloud_classifier_v83.FrozenFeatures', FakeExtractor), \
                     patch.object(FakeExtractor, 'extract', side_effect=AssertionError('must reuse all cached objects')):
                    cache(args)
                self.assertEqual(before, (args.output/'features.npz').read_bytes())
            with np.load(root/'train/features.npz', allow_pickle=False) as values:
                broken = {k: values[k] for k in values.files}
            broken['local_features'][0, 0, 0, 0] = np.nan
            np.savez(root/'corrupt.npz', **broken)
            with self.assertRaisesRegex(ValueError, 'numeric'):
                read_cache(root/'corrupt.npz')
            fitting = argparse.Namespace(train_cache=root/'train/features.npz', dev_cache=root/'dev/features.npz',
                text_features=ROOT/'weights/v82_clip_text_features.npz', mode='conditioned', seed=42,
                epochs=3, batch_size=2, lr=.01, consistency_weight=.1, threads=1, device='cpu', output=root/'model')
            self.assertEqual(train(fitting), 0)
            checkpoint = torch.load(fitting.output/'head.pt', weights_only=True)
            self.assertEqual(checkpoint['config']['code_sha256'], code_hashes())
            self.assertGreater(float(checkpoint['state_dict']['global_adapter.weight'].abs().sum()), 0)
            inference = argparse.Namespace(cache=root/'eval/features.npz', head=fitting.output/'head.pt',
                                           batch_size=2, threads=1, device='cpu', output=root/'predictions')
            predict(inference)
            predictions = json.loads((inference.output/'predictions.json').read_text())
            self.assertEqual(predictions['summary']['completed'], 3)
            for row in predictions['results']:
                self.assertNotIn('target', row)
                self.assertNotIn('category', row)
                self.assertAlmostEqual(sum(row['scores']), 1., places=6)
            data, meta = read_cache(inference.cache)
            head = CategoryHead(checkpoint['state_dict']['category_text'], 'conditioned')
            head.load_state_dict(checkpoint['state_dict'])
            expected = head(*(data[k][:, 0] for k in ('global_features', 'local_features', 'geometry')))
            np.testing.assert_allclose(predictions['results'][0]['scores'],
                                       expected['logits'].softmax(-1)[0].detach().numpy(), atol=1e-7)
            with self.assertRaisesRegex(ValueError, 'Frozen run'):
                predict(inference)
            # A copied evaluation file cannot conceal a training scene or identical cloud.
            with np.load(inference.cache, allow_pickle=False) as values:
                copy = {k: values[k] for k in values.files}
            meta['sources'][0]['scene'] = 'scene0001_01'
            for row in meta['records']:
                row['scene'] = 'scene0001_01'
            copy['metadata'] = json.dumps(meta)
            np.savez(root/'overlap.npz', **copy)
            inference.cache, inference.output = root/'overlap.npz', root/'bad_predictions'
            with self.assertRaisesRegex(ValueError, 'overlap'):
                predict(inference)
            self.assertFalse(inference.output.exists())
            fitting.dev_cache, fitting.output = root/'eval/features.npz', root/'bad_model'
            with self.assertRaisesRegex(ValueError, 'never validation'):
                train(fitting)

    def test_raw_conversion_needs_no_author_category_table(self):
        from plyfile import PlyData, PlyElement
        with tempfile.TemporaryDirectory() as td:
            root, scene = Path(td), 'scene0001_00'
            raw = root/'raw'/scene
            raw.mkdir(parents=True)
            points = np.zeros(6, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                      ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
            points['x'], points['y'], points['z'] = np.arange(6), np.arange(6)%2, np.arange(6)%3
            PlyData([PlyElement.describe(points, 'vertex')]).write(raw/f'{scene}_vh_clean_2.ply')
            (raw/f'{scene}.txt').write_text('axisAlignment = '+' '.join(map(str, np.eye(4).flatten())))
            write_json(raw/f'{scene}_vh_clean_2.0.010000.segs.json', {'segIndices': [0, 0, 0, 1, 1, 1]})
            write_json(raw/f'{scene}.aggregation.json', {'segGroups': [
                {'objectId': 8, 'segments': [0], 'label': 'chair'},
                {'objectId': 12, 'segments': [1], 'label': 'table'}]})
            output = root/'converted'/scene
            convert_scene(root/'raw', scene, output)
            with np.load(output/'scene.npz', allow_pickle=False) as cloud:
                np.testing.assert_array_equal(cloud['ids'], [8, 8, 8, 12, 12, 12])
            self.assertEqual(len(make_manifest(output.parent, [scene], 'train')['samples']), 2)
            self.assertFalse((output/'objects.json').exists())

    def test_download_is_atomic_and_reuses_only_completed_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root/'source'
            source.write_bytes(b'local-download-fixture'*100)
            destination = root/'download/scene.ply'
            download(source.as_uri(), destination, reserve_bytes=0)
            self.assertEqual(source.read_bytes(), destination.read_bytes())
            self.assertFalse(destination.with_suffix('.ply.part').exists())
            # Successful resume must not need the source endpoint again.
            download('https://invalid.example/not-used', destination, reserve_bytes=0)
            with patch('category_v10.pipeline.shutil.disk_usage', return_value=argparse.Namespace(free=0)):
                with self.assertRaisesRegex(ValueError, 'disk space'):
                    download(source.as_uri(), root/'no-space.ply', reserve_bytes=5*1024**3)
            self.assertFalse((root/'no-space.ply').exists())


if __name__ == '__main__':
    unittest.main()
