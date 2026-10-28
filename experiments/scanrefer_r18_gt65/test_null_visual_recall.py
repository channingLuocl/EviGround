"""Offline gate, actual dispatch, and completed-run checks; network is mocked."""
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from null_visual_recall import null_reason, recover_candidates
import retry_null_visual as entry


def object_row(oid):
    return {'bbox_id': oid, 'bbox_3d': [float(oid), 0., 1., 1., 1., 1.],
            'target': 'chair', 'category_status': 'uncertain',
            'category_hypotheses': [{'category': 'chair', 'rank': 1, 'score': 2.5, 'gap_from_top1': 0.}]}


def load_source(directory, name, source):
    path = directory / (name + '.py')
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VisualRecallTests(unittest.TestCase):
    def test_empty_pool_uses_only_supplied_classifier_objects_without_mutation(self):
        objects = [object_row(4), object_row(19)]
        before = deepcopy(objects)
        recovered, audit = recover_candidates([], objects)
        self.assertEqual([obj['bbox_id'] for obj in recovered], [4, 19])
        self.assertEqual(objects, before)
        self.assertTrue(audit['fallback_applied'])
        self.assertFalse(audit['ground_truth_used_for_recall'])
        self.assertIs(recovered[0], objects[0])

    def test_nonempty_pool_does_not_expand_or_change_category_scores(self):
        objects = [object_row(4), object_row(19)]
        recovered, audit = recover_candidates([objects[1]], objects)
        self.assertEqual(recovered, [objects[1]])
        self.assertFalse(audit['fallback_applied'])
        self.assertEqual(audit['original_candidate_ids'], audit['effective_candidate_ids'])

    def test_no_invented_candidate_when_input_objects_are_empty(self):
        recovered, audit = recover_candidates([], [])
        self.assertEqual(recovered, [])
        self.assertFalse(audit['fallback_applied'])

    def test_null_console_reasons_distinguish_gate_model_and_error(self):
        self.assertEqual(null_reason({'active_trace': {'stop_reason': 'category_gate_abstention'}}), 'empty_candidate_pool')
        self.assertEqual(null_reason({'predicted_id': None}), 'model_abstention')
        self.assertEqual(null_reason({'predicted_id': 4, 'error': 'provider failed'}), 'runtime_or_api_error')
        self.assertIsNone(null_reason({'predicted_id': 0}))

    def test_unknown_frozen_source_anchor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            entry.replace_once('different code', 'expected anchor', 'replacement')

    def test_all_generated_runtime_files_compile(self):
        for name, source in entry.runtime_sources().items():
            compile(source, name, 'exec')

    def test_actual_react_schema_and_prompt_expose_the_recovered_pool(self):
        class ReachedPlanner(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            react = load_source(root, 'visual_test_react', entry.runtime_sources()['r18_react.py'])
            objects = [object_row(4), object_row(19)]
            candidates, audit = recover_candidates([], objects)
            parsed = {'Target': ['unmatched'], 'Anchor': [],
                      'candidate_generation': {'visual_recall': audit}}

            def fake_chat(messages, **kwargs):
                state = json.loads(messages[1]['content'][0]['text'])
                self.assertEqual(state['allowed_target_ids'], [4, 19])
                self.assertTrue(state['candidate_generation']['fallback_applied'])
                self.assertIn('none is category-verified', state['category_policy'])
                self.assertNotIn('target_id', state)
                schema = kwargs['response_schema']['schema']
                self.assertEqual(schema['properties']['predicted_id']['enum'], [4, 19, None])
                self.assertEqual(state['remaining_tool_calls'], 3)
                raise ReachedPlanner()

            views = [{'image': 'mock_scene.png'}, {'image': 'mock_top.png'}]
            with patch('v74_grounding.scene_views', return_value=views), \
                 patch('v4_policy.v62_enabled', return_value=True), \
                 patch('qwen_api.api_provider', return_value='openai'), \
                 patch.object(react, 'precision_evidence', return_value={'objects': [{'id': 4}, {'id': 19}]}), \
                 patch.object(react, 'image_part', return_value={'type': 'mock_image'}), \
                 patch.object(react, 'chat', side_effect=fake_chat), \
                 self.assertRaises(ReachedPlanner):
                react.ground('find an unmatched object', parsed, objects, candidates,
                    None, None, None, root, [], {}, {'steps': []}, 3)

    def test_actual_run_query_dispatches_empty_pool_to_react(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = load_source(root, 'visual_test_scene', entry.runtime_sources()['real_scene.py'])
            table = root / 'objects_top5.json'
            table.write_text(json.dumps({'version': 'fixture', 'gate': {},
                                        'source_predictions_sha256': 'fixture', 'cloud_sha256': 'fixture'}))
            np.savez(root / 'scene.npz', xyz=np.array([[0., 0., 0.], [1., 1., 1.]]),
                     rgb=np.zeros((2, 3), dtype=np.uint8), ids=np.array([4, 19]))
            objects = [object_row(4), object_row(19)]

            def fake_ground(query, parsed, objs, geometry, candidates, xyz, rgb, ids,
                            out, calls, trace, limit, trusted, **kwargs):
                self.assertEqual([obj['bbox_id'] for obj in candidates], [4, 19])
                self.assertEqual(parsed['candidate_generation']['status'], 'unknown')
                self.assertTrue(parsed['candidate_generation']['visual_recall']['fallback_applied'])
                self.assertFalse(trusted)
                self.assertEqual(limit, 3)
                trace.update(decision_backend='openai', planner_input='text_and_scene_images',
                             steps=[{'mocked_react': True}])
                return {'predicted_id': 4, 'bbox_3d': objects[0]['bbox_3d'],
                        'diagnosis': {'candidate_ids': [4, 19]}, 'active_trace': trace,
                        'decision_status': 'tentative', 'prediction_source': 'offline_mock',
                        'implementation_version': 'r18'}

            with patch.object(scene, 'object_table_path', return_value=table), \
                 patch.object(scene, 'structural_objects', return_value=[]), \
                 patch.object(scene, 'api_config_summary', return_value={}), \
                 patch.object(scene, 'api_provider', return_value='openai'), \
                 patch.object(scene, 'parse_query', return_value={'Target': ['unmatched'], 'Anchor': []}), \
                 patch.object(scene, 'match_retrieval_categories', return_value=[]), \
                 patch('categories_v9.load_category_objects', return_value=objects), \
                 patch('wall_geometry.attach_wall_geometry', side_effect=lambda objs, *args: objs), \
                 patch('v4_policy.v5_enabled', return_value=False), \
                 patch('react_decision.validate_react_config', return_value='openai'), \
                 patch('r8_grounding.generic_target_pool', return_value=False), \
                 patch('active_diagnostic.active_ground', side_effect=fake_ground) as ground, \
                 redirect_stdout(io.StringIO()):
                prediction = scene.run_query(root, 'find an unmatched object', root / 'episode',
                    mode='active_diagnostic', object_table='v9_top5', review_limit=3)
            ground.assert_called_once()
            self.assertEqual(prediction['predicted_id'], 4)
            self.assertEqual(prediction['category_target_ids'], [])
            self.assertEqual(prediction['candidate_gate']['effective_candidate_ids'], [4, 19])
            saved = json.loads((root / 'episode/prediction.json').read_text())
            self.assertEqual(saved['candidate_gate'], prediction['candidate_gate'])

    def test_preflight_and_completed_run_do_not_launch_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            entry.save(output / 'manifest.json', {'rows': []})
            entry.save(output / 'results.json', {'results': []})
            entry.save(output / 'completion.json', {'results_sha256': entry.digest(output / 'results.json')})
            protocol = {'selected_ids': [1], 'baseline_correct25': 5834}
            for action in ('preflight', 'run'):
                with patch.object(entry, 'OUTPUT', output), \
                     patch.object(entry, 'verify', return_value=({}, output / 'manifest.json')), \
                     patch.object(entry, 'checked', return_value=(protocol, [])), \
                     patch.object(entry, 'validate_runtime_settings'), \
                     patch.object(entry, 'validate'), \
                     patch.object(entry, 'run_visual') as launch, \
                     patch.object(entry, 'report_visual'), \
                     patch.dict(entry.os.environ, {}, clear=True), \
                     patch('sys.argv', ['retry_null_visual.py', action]), \
                     redirect_stdout(io.StringIO()):
                    entry.main()
                    launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
