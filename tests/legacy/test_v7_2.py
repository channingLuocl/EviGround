"""Category-source switching, provenance, and GT isolation; no API calls."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from eviground.grounding.real_scene import structural_objects
from eviground.experiments.run_size_validation import GT_V71_EQUIVALENT_HASHES, comparable_settings, main, run_case
from eviground.grounding.v4_policy import settings


class V72Tests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'EVIGROUND_V4': '1', 'EVIGROUND_V7': '1',
            'EVIGROUND_V7_2': '1', 'EVIGROUND_API_PROVIDER': 'openai',
            'EVIGROUND_REACT_BACKEND': 'openai', 'EVIGROUND_MODEL': 'offline-mock',
            'EVIGROUND_BASE_URL': 'https://example.invalid/v1', 'EVIGROUND_API_KEY': 'offline-only',
            'EVIGROUND_API_SLOT_DIR': '', 'EVIGROUND_SYNTAX_BACKEND': 'none'})
        env.start(); self.addCleanup(env.stop)

    def fixture(self, root):
        scene = root / 'scenes/scene0010_00'; scene.mkdir(parents=True)
        for name in ('scene.npz', 'provenance.json', 'scene_instances.json'):
            (scene / name).write_text('{}')
        (scene / 'objects.json').write_text('[{"bbox_id":13,"target":"shelf"},{"bbox_id":5,"target":"shelf"}]')
        (scene / 'objects_gt.json').write_text('[{"bbox_id":13,"target":"chair"},{"bbox_id":5,"target":"chair"}]')
        row = {'csv_row': 10, 'scene': scene.name, 'query': 'the chair',
               'target_id': 13, 'stimulus_candidate_ids': [13, 5]}
        manifest = root / 'selection.json'
        manifest.write_text(json.dumps({'rows': [row], 'csv_sha256': 'offline'}))
        return ['run_size_validation.py', '--manifest', str(manifest), '--scene-root', str(root / 'scenes'),
                '--output', str(root / 'output'), '--workers', '1'], scene, row

    def call(self, root, argv):
        def worker(command, **kwargs):
            table = 'gt' if '--gt=true' in command else 'pred'
            self.assertIn('--gt=' + str(table == 'gt').lower(), command)
            self.assertNotIn('--target-id', command)
            directory = Path(command[command.index('--output') + 1])
            (directory / 'prediction.json').write_text(json.dumps({'query': 'the chair',
                'mode': 'active_diagnostic', 'object_table': table, 'predicted_id': 13}))
            return SimpleNamespace(returncode=0)
        with patch('sys.argv', argv), patch('eviground.experiments.run_size_validation.ROOT', root), \
                patch('eviground.experiments.run_size_validation.subprocess.run', side_effect=worker) as process, \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            main()
        return process

    def test_default_pred_uses_predicted_file_without_requiring_gt_labels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); argv, scene, _ = self.fixture(root)
            (scene / 'objects_gt.json').unlink()
            process = self.call(root, argv)
            self.assertEqual(process.call_count, 1)
            manifest = json.loads((root / 'output/manifest.json').read_text())
            self.assertEqual(manifest['object_table'], 'pred')
            self.assertEqual(manifest['v4']['version'], 'relative-size-v7.2')
            inputs = manifest['input_sha256'][scene.name]
            self.assertIn('objects.json', inputs); self.assertNotIn('objects_gt.json', inputs)
            result = json.loads((root / 'output/q00010/evaluation.json').read_text())
            self.assertTrue(result['correct']); self.assertEqual(result['object_table'], 'pred')
            self.assertEqual(self.call(root, argv).call_count, 0)

    def test_gt_flag_and_boolean_forms_propagate_to_worker(self):
        for flag, table in (('--gt', 'gt'), ('--gt=true', 'gt'), ('--gt=false', 'pred')):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as td:
                root = Path(td); argv, _, _ = self.fixture(root)
                self.call(root, argv + [flag])
                self.assertEqual(json.loads((root / 'output/manifest.json').read_text())['object_table'], table)

    def test_same_directory_cannot_mix_gt_and_predicted_results(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); argv, _, _ = self.fixture(root)
            self.call(root, argv + ['--gt'])
            original = (root / 'output/results.json').read_text()
            with self.assertRaisesRegex(ValueError, 'different code/settings'):
                self.call(root, argv)
            self.assertEqual((root / 'output/results.json').read_text(), original)

    def test_sample_cache_rejects_historical_gt_under_predicted_mode(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); _, _, row = self.fixture(root)
            directory = root / 'q00010'; directory.mkdir()
            (directory / 'evaluation.json').write_text(json.dumps({**row, 'predicted_id': 13, 'correct': True}))
            with patch('eviground.experiments.run_size_validation.subprocess.run') as process:
                with self.assertRaisesRegex(ValueError, 'different object-table'):
                    run_case(row, root / 'scenes', root, 3, object_table='pred')
                process.assert_not_called()

    def test_verified_v72_edits_are_equivalent_only_for_v71_gt(self):
        baseline = {'object_table': 'gt', 'v4': {'version': 'relative-size-v7.1'},
                    'code_sha256': {name: hashes[1] for name, hashes in GT_V71_EQUIVALENT_HASHES.items()}}
        current = {**baseline, 'code_sha256': {
            name: hashes[0] for name, hashes in GT_V71_EQUIVALENT_HASHES.items()}}
        self.assertEqual(comparable_settings(baseline), comparable_settings(current))
        self.assertEqual(current['code_sha256']['real_scene.py'], GT_V71_EQUIVALENT_HASHES['real_scene.py'][0])
        for change in ({'object_table': 'pred'}, {'v4': {'version': 'relative-size-v7.2'}},
                       {'code_sha256': {**current['code_sha256'], 'real_scene.py': 'unverified-edit'}}):
            with self.subTest(change=change):
                self.assertNotEqual(comparable_settings(baseline), comparable_settings({**current, **change}))

    def test_worker_default_and_gt_switch_reach_real_scene(self):
        for flags, table in (([], 'pred'), (['--gt'], 'gt')):
            argv = ['run_size_validation.py', '--worker', '--scene-dir', '/fixture',
                    '--query', 'the chair', '--output', '/output', '--csv-row', '10', *flags]
            with self.subTest(table=table), patch('sys.argv', argv), patch('eviground.grounding.real_scene.run_query') as query:
                main()
                self.assertEqual(query.call_args.kwargs['object_table'], table)

    def test_structural_references_do_not_supply_gt_categories_in_predicted_mode(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'scene_instances.json').write_text(json.dumps([
                {'bbox_id': 13, 'target': 'chair', 'bbox_3d': [0, 0, 0, 1, 1, 1]},
                {'bbox_id': 90, 'target': 'floor', 'bbox_3d': [0, 0, 0, 5, 5, .1]}]))
            objects = [{'bbox_id': 13}]
            predicted = structural_objects(root, objects, 'pred')
            self.assertEqual(predicted[0]['target'], 'object')
            self.assertEqual(structural_objects(root, objects, 'gt')[0]['target'], 'floor')
            self.assertEqual(predicted[0]['bbox_id'], 90)
            self.assertEqual(predicted[0]['bbox_3d'], [0, 0, 0, 5, 5, .1])
            self.assertEqual(settings()['version'], 'relative-size-v7.2')


if __name__ == '__main__':
    unittest.main()
