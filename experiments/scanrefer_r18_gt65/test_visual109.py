"""Offline policy/actual-dispatch tests; no live model calls."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import retry_visual109 as entry
import visual109_policy as policy
from retry_null import merge
from test_null_visual_recall import load_source, object_row


class Visual109Tests(unittest.TestCase):
    def test_nonempty_category_pool_expands_without_mutating_scores(self):
        objects = [object_row(19), object_row(4)]
        before = deepcopy(objects)
        candidates, gate = policy.recover_candidates([objects[0]], objects)
        self.assertEqual([o['bbox_id'] for o in candidates], [4, 19])
        self.assertTrue(gate['fallback_applied'])
        self.assertEqual(objects, before)
        self.assertFalse(gate['ground_truth_used_for_recall'])

    def test_empty_pool_and_supplemental_geometry(self):
        supplemental = {'bbox_id': 100, 'bbox_3d': [0, 0, 0, 1, 1, 1], 'target': 'unknown'}
        candidates, gate = policy.recover_candidates([], [supplemental, object_row(4)])
        self.assertEqual([o['bbox_id'] for o in candidates], [4])
        self.assertEqual(gate['effective_candidate_ids'], [4])
        self.assertEqual(policy.recover_candidates([], [supplemental])[0], [])

    def test_generated_overrides_compile(self):
        self.assertEqual(set(entry.runtime_sources()), {'r18_react.py', 'real_scene.py', 'run_grounding_v9.py'})
        for name, source in entry.runtime_sources().items():
            compile(source, name, 'exec')

    def test_atlas_covers_every_id_in_stable_pages_using_actual_renderer(self):
        with tempfile.TemporaryDirectory() as directory:
            objects = [object_row(i) for i in reversed(range(13))]
            rng = np.random.default_rng(7)
            ids = np.repeat(np.arange(13), 30)
            xyz = rng.uniform(-.25, .25, (len(ids), 3)) + np.column_stack((ids, ids*0, ids*0+1))
            rgb = np.full((len(ids), 3), [120, 170, 210], dtype=np.uint8)
            views = policy.candidate_atlas(objects, xyz, rgb, ids, directory)
            self.assertEqual([v['focus_ids'] for v in views], [list(range(12)), [12]])
            self.assertEqual([i for v in views for i in v['marked_ids']], list(range(13)))
            self.assertEqual(sum(v['views_rendered'] for v in views), 13)
            self.assertTrue(all(Path(v['image']).exists() for v in views))

    def test_missing_instance_points_are_explicit_not_invented_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            views = policy.candidate_atlas([object_row(4)], np.zeros((1, 3)),
                np.zeros((1, 3), dtype=np.uint8), np.array([9]), directory)
            self.assertEqual(views[0]['focus_ids'], [4])
            self.assertEqual(views[0]['marked_ids'], [])
            self.assertEqual(views[0]['views_rendered'], 0)

    def test_actual_react_gets_all_views_bounded_schema_and_no_answers(self):
        class ReachedPlanner(Exception):
            pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            react = load_source(root, 'test_v109_react', entry.runtime_sources()['r18_react.py'])
            objects = [object_row(4), object_row(19)]
            candidates, gate = policy.recover_candidates([objects[0]], objects)
            parsed = {'Target': ['chair'], 'Anchor': [], 'candidate_generation': {'visual_recall': gate}}
            overview = [{'image': f'overview{i}.png'} for i in range(3)]
            atlas = [{'image': 'atlas.png', 'focus_ids': [4, 19], 'marked_ids': [4, 19], 'views_rendered': 2}]
            def fake_chat(messages, **kwargs):
                state = json.loads(messages[1]['content'][0]['text'])
                self.assertEqual(state['allowed_target_ids'], [4, 19])
                self.assertEqual(len(messages[1]['content'])-1, 4)
                self.assertEqual(state['remaining_tool_calls'], 3)
                self.assertEqual(kwargs['response_schema']['schema']['properties']['focus_ids']['maxItems'], 12)
                self.assertEqual(kwargs['response_schema']['schema']['type'], 'object')
                self.assertNotIn('target_id', state)
                self.assertNotIn('iou_3d', state)
                self.assertNotIn('correct25', state)
                self.assertNotIn('previous_result', state)
                self.assertIn('support', messages[0]['content'])
                raise ReachedPlanner()
            trace = {'steps': [], 'internal_usage': {'render': 0}}
            with patch('v74_grounding.scene_views', return_value=overview), \
                 patch('v4_policy.v62_enabled', return_value=True), \
                 patch('qwen_api.api_provider', return_value='openai'), \
                 patch.object(react, 'candidate_atlas', return_value=atlas), \
                 patch.object(react, 'precision_evidence', return_value={'objects': [{'id': 4}, {'id': 19}]}), \
                 patch.object(react, 'image_part', side_effect=lambda image: {'type': 'image', 'image': image}), \
                 patch.object(react, 'chat', side_effect=fake_chat), self.assertRaises(ReachedPlanner):
                react.ground('find the chair', parsed, objects, candidates, None, None, None,
                             root, [], {}, trace, 3)
            self.assertEqual(trace['r18_initial_resources']['atlas_instances'], 2)
            self.assertEqual(trace['internal_usage']['render'], 5)

    def test_actual_react_defers_uninspected_finish_then_accepts_inspected_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            react = load_source(root, 'test_v109_finish', entry.runtime_sources()['r18_react.py'])
            objects = [object_row(4), object_row(19)]
            parsed = {'Target': ['chair'], 'Anchor': [], '_v4_budget': {}}
            finish = {'action': 'finish', 'predicted_id': 4, 'reason': 'Chair 4 matches the description; rival 19 differs.',
                'evidence': [{'source_quote': 'find the chair', 'object_ids': [4, 19], 'observation': 'focused shape comparison'}], 'unresolved': []}
            inspect = {'action': 'inspect', 'predicted_id': None, 'focus_ids': [4, 19], 'context': 'isolated',
                       'azimuth_deg': 315, 'elevation_deg': 30}
            responses = [finish, inspect, finish]
            trace = {'steps': [], 'internal_usage': {'render': 0, 'visual_check': 0}}
            with patch('v74_grounding.scene_views', return_value=[{'image': 'overview.png'}]*3), \
                 patch.object(react, 'candidate_atlas', return_value=[]), \
                 patch.object(react, 'precision_evidence', return_value={'objects': [{'id': 4}, {'id': 19}]}), \
                 patch.object(react, 'image_part', return_value={'type': 'mock_image'}), \
                 patch('active_diagnostic.render_action', return_value={'image': 'focus.png', 'marked_ids': [4, 19], 'views_rendered': 2}), \
                 patch.object(react, 'settings', return_value={'version': 'test-r18'}), \
                 patch.object(react, 'chat', side_effect=[{'_content': json.dumps(a), '_finish_reason': 'stop'} for a in responses]):
                result = react.ground('find the chair', parsed, objects, objects, None, None, None,
                                      root, [], {}, trace, 3)
            self.assertEqual(result['predicted_id'], 4)
            self.assertEqual(trace['r18_observations'][0]['tool'], 'finish_check')
            self.assertEqual(trace['r18_observations'][1]['tool'], 'render')
            self.assertEqual(trace['internal_usage']['render'], 5)

    def test_malformed_array_response_is_rejected(self):
        with self.assertRaises((ValueError, TypeError)):
            policy.parse_action('[{"id":4,"context":"isolated"}]')
        self.assertEqual(policy.parse_action('{"action":"finish","predicted_id":null}')['action'], 'finish')

    def test_actual_scene_dispatch_expands_nonempty_pool_without_gt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = load_source(root, 'test_v109_scene', entry.runtime_sources()['real_scene.py'])
            table = root/'objects_top5.json'
            table.write_text(json.dumps({'version': 'fixture', 'gate': {},
                'source_predictions_sha256': 'fixture', 'cloud_sha256': 'fixture'}))
            np.savez(root/'scene.npz', xyz=np.array([[0., 0., 0.], [1., 1., 1.]]),
                rgb=np.zeros((2, 3), dtype=np.uint8), ids=np.array([4, 19]))
            objects = [object_row(4), object_row(19)]
            def fake_ground(query, parsed, objs, geometry, candidates, xyz, rgb, ids,
                            out, calls, trace, limit, trusted, **kwargs):
                self.assertEqual([o['bbox_id'] for o in candidates], [4, 19])
                self.assertEqual(parsed['candidate_generation']['visual_recall']['original_candidate_ids'], [4])
                self.assertEqual(parsed['candidate_generation']['status'], 'unknown')
                self.assertFalse(trusted)
                self.assertEqual(limit, 3)
                trace.update(decision_backend='openai', planner_input='text_and_scene_images', steps=[{'offline': True}])
                return {'predicted_id': 19, 'bbox_3d': objects[1]['bbox_3d'],
                    'diagnosis': {'candidate_ids': [4, 19]}, 'active_trace': trace,
                    'decision_status': 'tentative', 'prediction_source': 'offline_mock', 'implementation_version': 'r18'}
            with patch.object(scene, 'object_table_path', return_value=table), \
                 patch.object(scene, 'structural_objects', return_value=[]), \
                 patch.object(scene, 'api_config_summary', return_value={}), \
                 patch.object(scene, 'api_provider', return_value='openai'), \
                 patch.object(scene, 'parse_query', return_value={'Target': ['chair'], 'Anchor': []}), \
                 patch.object(scene, 'match_retrieval_categories', return_value=[objects[0]]), \
                 patch('categories_v9.load_category_objects', return_value=objects), \
                 patch('wall_geometry.attach_wall_geometry', side_effect=lambda objs, *args: objs), \
                 patch('v4_policy.v5_enabled', return_value=False), \
                 patch('react_decision.validate_react_config', return_value='openai'), \
                 patch('r8_grounding.generic_target_pool', return_value=False), \
                 patch('active_diagnostic.active_ground', side_effect=fake_ground) as ground, \
                 redirect_stdout(io.StringIO()):
                prediction = scene.run_query(root, 'find the chair', root/'episode',
                    mode='active_diagnostic', object_table='v9_top5', review_limit=3)
            ground.assert_called_once()
            self.assertEqual(prediction['category_target_ids'], [4])
            self.assertEqual(prediction['candidate_gate']['effective_candidate_ids'], [4, 19])
            self.assertEqual(prediction['predicted_id'], 19)

    def test_selection_is_all_failed_outcomes_and_merge_is_not_best_of(self):
        rows = [{'csv_row': 3, 'predicted_id': 4, 'iou_3d': 1},
                {'csv_row': 1, 'predicted_id': 4, 'iou_3d': 0},
                {'csv_row': 2, 'predicted_id': None, 'iou_3d': 0}]
        self.assertEqual(entry.selected_ids(rows), [1, 2])
        replacements = [{'csv_row': 1, 'predicted_id': None, 'iou_3d': 0, 'error': 'new failure'},
                        {'csv_row': 2, 'predicted_id': 19, 'iou_3d': 0}]
        result = merge(rows, replacements, [1, 2])
        self.assertEqual(result[0], rows[0])
        self.assertEqual(result[1:], replacements)
        with self.assertRaises(ValueError):
            merge(rows, replacements[:1], [1, 2])

    def test_worker_arguments_exclude_ground_truth_and_prior_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = load_source(root, 'test_v109_worker', entry.runtime_sources()['run_grounding_v9.py'])
            row = {'csv_row': 100, 'scene': 'scene_fixture', 'query': 'find the object', 'target_id': 991}
            scene = root / row['scene']
            scene.mkdir()
            (scene / 'objects_gt.json').write_text(json.dumps([{'bbox_id': 991, 'bbox_3d': [0, 0, 0, 1, 1, 1]}]))
            def fake_run(command, **kwargs):
                self.assertNotIn('--manifest', command)
                self.assertNotIn('--target-id', command)
                self.assertNotIn('991', command)
                self.assertNotIn('--baseline', command)
                self.assertEqual(command[command.index('--query')+1], row['query'])
                out = Path(command[command.index('--output')+1])
                (out / 'prediction.json').write_text(json.dumps({'predicted_id': 4, 'category_target_ids': [4], 'bbox_3d': [0, 0, 0, 1, 1, 1], 'active_trace': {}}))
                return SimpleNamespace(returncode=0)
            with patch.object(runner.subprocess, 'run', side_effect=fake_run) as worker, \
                 patch('validation_reporting.evaluate', return_value={**row, 'predicted_id': 4, 'object_table': 'v9_top5'}), \
                 patch('metrics.bbox_iou_3d', return_value=0):
                runner.run_case(row, root, root/'round', 5, 3)
                worker.assert_called_once()

    def test_all_cached_outcomes_are_reused_equally(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = load_source(root, 'test_v109_cache', entry.runtime_sources()['run_grounding_v9.py'])
            for i, outcome in enumerate(({'predicted_id': 4, 'iou_3d': 1}, {'predicted_id': 19, 'iou_3d': 0},
                                        {'predicted_id': None}, {'predicted_id': None, 'error': 'api'})):
                row = {'csv_row': i, 'scene': 'fixture', 'query': 'query', 'target_id': 4}
                saved = {**row, 'object_table': 'v9_top5', **outcome}
                out = root / f'q{i:05d}'
                out.mkdir()
                (out/'evaluation.json').write_text(json.dumps(saved))
                with patch.object(runner.subprocess, 'run') as worker:
                    self.assertEqual(runner.run_case(row, root, root, 5, 3), saved)
                    worker.assert_not_called()

    def test_preflight_and_completed_run_never_launch_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            entry.save(output/'manifest.json', {'rows': []})
            entry.save(output/'results.json', {'results': []})
            entry.save(output/'completion.json', {'results_sha256': entry.digest(output/'results.json')})
            protocol = {'selected_ids': [1], 'baseline_correct25': 6108}
            for action in ('prepare', 'preflight', 'run'):
                entry.save(output/'protocol.json', protocol)
                with patch.object(entry, 'OUTPUT', output), \
                     patch.object(entry, 'verify', return_value=({}, output/'manifest.json')), \
                     patch.object(entry, 'checked', return_value=(protocol, [])), \
                     patch.object(entry, 'validate_runtime_settings'), patch.object(entry, 'validate'), \
                     patch.object(entry, 'run_visual') as launch, patch.object(entry, 'report_visual'), \
                     patch.dict(entry.os.environ, {}, clear=True), patch('sys.argv', ['retry_visual109.py', action]), \
                     redirect_stdout(io.StringIO()):
                    entry.main()
                    launch.assert_not_called()

    def test_interrupted_episode_is_not_silently_restarted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'q00001').mkdir()
            with patch.object(entry, 'ROUND', root), patch.object(entry.subprocess, 'run') as launch:
                with self.assertRaisesRegex(ValueError, 'Interrupted trajectory'):
                    entry.run_visual({'selected_ids': [1]}, [])
                launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
