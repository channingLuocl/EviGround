"""Thresholds, alternate-category retrieval, payloads and isolated paired evaluation."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from seeground.categories.categories_v9 import (VERSION, gate_categories, load_category_objects, prepare_tables, sha256)
from seeground.experiments.compare_grounding_v9 import compare
from seeground.grounding.evidence import match_retrieval_categories
from seeground.grounding.real_scene import run_query
from seeground.grounding.react_decision import jev_state
from seeground.cli.run_grounding_v9 import save_manifest, run_case
from tests.legacy.test_v7_evidence import obj, parse


def top5(names=('clothes', 'chair', 'box', 'bag', 'sofa'), scores=(.34, .32, .30, .29, .28)):
    return [{'category': name, 'score': score} for name, score in zip(names, scores)]


def prepared(oid, predictions):
    accepted, audit = gate_categories(predictions)
    return {**obj(oid, predictions[0]['category']), 'top5': predictions,
            'category_hypotheses': accepted, 'category_gate_audit': audit,
            'category_status': 'single_hypothesis' if len(accepted) == 1 else 'uncertain'}


def write_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'version': VERSION, 'gate': {'top_k': 5, 'min_score': .2, 'max_gap': .03},
        'source_predictions_sha256': 'test', 'cloud_sha256': 'test', 'objects': rows}))


class GateTests(unittest.TestCase):
    def test_absolute_and_gap_boundaries_and_top1_control(self):
        rows = top5(scores=(.23, .20, .199, .18, .1))
        retained, audit = gate_categories(rows)
        self.assertEqual([r['rank'] for r in retained], [1, 2])
        self.assertIn('below_min_score', audit[2]['reasons'])
        self.assertIn('gap_exceeds_max', audit[3]['reasons'])
        self.assertEqual(len(gate_categories(rows, top_k=1)[0]), 1)
        self.assertEqual(gate_categories(top5(scores=(.19, .18, .17, .16, .15)))[0], [])
        for changes in ({'min_score': float('nan')}, {'max_gap': -.1}, {'top_k': 3}):
            with self.assertRaises(ValueError):
                gate_categories(rows, **changes)
        with self.assertRaises(ValueError):
            gate_categories(top5(scores=(.2, .3, .1, .09, .08)))

    def test_alternatives_match_targets_and_references_once_without_rejected_fallback(self):
        chair = prepared(1, top5())
        desk = prepared(2, top5(('table', 'desk', 'box', 'bag', 'sofa')))
        rejected = prepared(3, top5(('chair', 'stool', 'box', 'bag', 'sofa'), (.19, .18, .17, .16, .15)))
        rows = [chair, desk, rejected]
        self.assertEqual([r['bbox_id'] for r in match_retrieval_categories(['chair'], rows)], [1])
        self.assertEqual([r['bbox_id'] for r in match_retrieval_categories(['desk'], rows)], [2])
        self.assertEqual(len(match_retrieval_categories(['chair', 'clothes'], rows)), 1)
        self.assertEqual(match_retrieval_categories(['box'], rows), [])
        self.assertEqual(len(match_retrieval_categories(['chair'], [obj(8, 'chair')])), 1)

    def test_prepared_table_revalidates_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'table.json'
            row = prepared(1, top5())
            write_table(p, [row])
            self.assertEqual(load_category_objects(p)[0]['category_hypotheses'][1]['category'], 'chair')
            row['category_hypotheses'].append({'category': 'box'})
            write_table(p, [row])
            with self.assertRaises(ValueError):
                load_category_objects(p)

    def test_prepare_uses_frozen_predictions_and_cloud_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source'; source.mkdir()
            scene = root/'scenes'/'test'; scene.mkdir(parents=True)
            np.savez(scene/'scene.npz', ids=np.array([1]))
            row = {**prepared(1, top5()), 'scene': 'test', 'status': 'completed', 'predicted_object': 'clothes'}
            (source/'predictions.json').write_text(json.dumps({'summary': {'complete': True, 'errors': 0,
                'version': 'object-category-v8.3-zero-shot-r1'}, 'results': [row]}))
            (source/'config.json').write_text(json.dumps({'jobs': [{'scene': 'test', 'bbox_id': 1}],
                'sources': [{'scene': 'test', 'cloud_sha256': sha256(scene/'scene.npz')}]}))
            (source/'FROZEN.json').write_text(json.dumps({'files': {n: sha256(source/n) for n in
                ('predictions.json', 'config.json')}}))
            result = prepare_tables(source, root/'scenes', root/'out', ['test'])
            self.assertEqual(result['instances'], 1)
            self.assertEqual(result['mean_retained'], 2)
            # There are no GT/author OLT files anywhere in this fixture.
            with self.assertRaises(ValueError):
                prepare_tables(source, root/'scenes', root/'out', ['test'], min_score=.35)
            (source/'predictions.json').write_text('{}')
            with self.assertRaises(ValueError):
                prepare_tables(source, root/'scenes', root/'other', ['test'])

    def test_compact_planner_keeps_hypotheses(self):
        row = prepared(1, top5())
        data = jev_state({'query': 'chair', 'target_table': [row]})
        self.assertEqual(data['target_candidates'][0]['category_hypotheses'], row['category_hypotheses'])
        self.assertIn('mutually exclusive', data['category_policy'])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scene = self.root/'scene'; self.scene.mkdir()
        self.tables = self.root/'tables'
        env = patch.dict(os.environ, {'SEEGROUND_V4': '1', 'SEEGROUND_V7': '1', 'SEEGROUND_V7_4': '1',
            'SEEGROUND_V9': '1', 'SEEGROUND_REACT_BACKEND': 'openai', 'SEEGROUND_API_PROVIDER': 'openai',
            'SEEGROUND_V9_CATEGORY_ROOT': str(self.tables)})
        env.start(); self.addCleanup(env.stop)
        self.rows = [prepared(1, top5()), prepared(2, top5(('table', 'desk', 'box', 'bag', 'sofa')))]
        write_table(self.tables/'scene'/'objects_top5.json', self.rows)
        (self.scene/'scene_instances.json').write_text(json.dumps([obj(99, 'ANNOTATION_MUST_NOT_LEAK')]))
        np.savez(self.scene/'scene.npz', xyz=np.array([[0., 0., 0.], [1., 1., 1.]]),
                 rgb=np.zeros((2, 3), dtype=np.uint8), ids=np.array([1, 2]))

    def test_true_second_rank_target_and_reference_reach_grounder_without_gt(self):
        query = 'chair beside desk'
        parsed = parse(query, 'chair', [], anchors=['desk'])
        def ground(query, parsed, objects, geometry, candidates, xyz, rgb, ids, out, calls, trace, *args, **kwargs):
            self.assertEqual([o['bbox_id'] for o in candidates], [1])
            self.assertEqual(geometry[-1]['target'], 'object')
            self.assertEqual([o['bbox_id'] for o in match_retrieval_categories(['desk'], geometry)], [2])
            self.assertNotIn('ANNOTATION_MUST_NOT_LEAK', json.dumps(geometry))
            self.assertFalse(args[1])  # trusted_categories
            trace.update(decision_backend='openai', planner_input='text_only', steps=[])
            return {'query': query, 'predicted_id': 1, 'bbox_3d': objects[0]['bbox_3d'],
                'implementation_version': VERSION, 'prediction_source': 'mock',
                'decision_status': 'tentative', 'diagnosis': {}, 'active_trace': trace}
        with patch('seeground.grounding.real_scene.parse_query', return_value=parsed) as parser, \
                patch('seeground.grounding.active_diagnostic.active_ground', side_effect=ground), \
                patch('seeground.geometry.wall_geometry.attach_wall_geometry', side_effect=lambda objects, *args: objects):
            result = run_query(self.scene, query, self.root/'out', mode='active_diagnostic', object_table='v9_top5')
        self.assertIn('chair', parser.call_args.kwargs['scene_categories'])
        self.assertEqual(result['category_target_ids'], [1])
        self.assertEqual(result['category_anchor_ids'], [2])
        self.assertEqual(result['category_bindings']['targets'][0]['matched_hypotheses'][0]['rank'], 2)
        self.assertFalse(result['gt_categories_supplied_to_model'])

    def test_missing_target_abstains_without_full_scene_fallback(self):
        with patch('seeground.grounding.real_scene.parse_query', return_value=parse('the laptop', 'laptop', [])), \
                patch('seeground.grounding.active_diagnostic.active_ground') as ground, \
                patch('seeground.geometry.wall_geometry.attach_wall_geometry', side_effect=lambda objects, *args: objects):
            result = run_query(self.scene, 'the laptop', self.root/'out', mode='active_diagnostic', object_table='v9_top5')
        ground.assert_not_called()
        self.assertIsNone(result['predicted_id'])
        self.assertEqual(result['active_trace']['stop_reason'], 'category_gate_abstention')
        self.assertEqual(result['category_target_ids'], [])

    def test_generic_planner_receives_hypothesis_policy(self):
        from seeground.grounding.active_diagnostic import active_ground
        query = 'the chair'
        parsed = parse(query, 'chair', [])
        (self.root/'out').mkdir()
        points = np.array([[0., 0., 0.], [1., 1., 1.]])
        stopped = {'_content': json.dumps({'action': 'stop', 'reason': 'Need visual evidence'}),
                   '_finish_reason': 'stop', '_baseline': {}}
        answer = {'_content': json.dumps({'predicted_id': None, 'reason': 'Insufficient category evidence',
            'evidence': [], 'unresolved': ['uncertain category']}), '_finish_reason': 'stop', '_baseline': {}}
        with patch.dict(os.environ, {'SEEGROUND_V4_FINAL_REVIEW': '0'}), \
                patch('seeground.grounding.active_diagnostic.chat', return_value=stopped) as planner, \
                patch('seeground.grounding.v74_grounding.scene_views', return_value=[{'image': 'test.png'}]), \
                patch('seeground.grounding.v74_grounding.image_part', return_value={}), \
                patch('seeground.grounding.v74_grounding.chat', return_value=answer):
            active_ground(query, parsed, self.rows, self.rows, self.rows[:1], points, points,
                          np.array([1, 2]), self.root/'out', [], {}, 3, False)
        messages = planner.call_args.args[0]
        self.assertIn('mutually exclusive', messages[0]['content'])
        payload = json.loads(messages[1]['content'][0]['text'])
        self.assertIn('category_hypotheses', payload['target_table'][0])

    def test_scene_model_receives_alternatives_as_hypotheses(self):
        from seeground.grounding.active_diagnostic import active_ground
        query = 'the chair at the bottom of the stack'
        parsed = parse(query, 'chair', [], unhandled=['at the bottom of the stack'])
        (self.root/'out').mkdir()
        reply = {'_content': json.dumps({'predicted_id': 1, 'reason': 'Observed chair at stack base',
            'evidence': [{'source_quote': query, 'object_ids': [1], 'observation': 'Chair at base'}],
            'unresolved': ['category remains tentative']}), '_finish_reason': 'stop', '_baseline': {}}
        points = np.array([[0., 0., 0.], [1., 1., 1.]])
        with patch('seeground.grounding.v74_grounding.scene_views', return_value=[{'image': 'test.png'}]), \
                patch('seeground.grounding.v74_grounding.image_part', return_value={}), \
                patch('seeground.grounding.v74_grounding.chat', return_value=reply) as model:
            result = active_ground(query, parsed, self.rows, self.rows, self.rows[:1], points, points,
                                  np.array([1, 2]), self.root/'out', [], {}, 3, False)
        messages = model.call_args.args[0]
        payload = json.loads(messages[1]['content'][0]['text'])
        self.assertEqual(payload['allowed_target_ids'], [1])
        self.assertEqual(payload['objects'][0]['category_hypotheses'][1]['category'], 'chair')
        self.assertIn('mutually exclusive', messages[0]['content'])
        self.assertEqual(result['implementation_version'], VERSION)


class EvaluationTests(unittest.TestCase):
    def test_resume_rejects_changed_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            save_manifest(path, {'gate': .2})
            save_manifest(path, {'gate': .2})
            with self.assertRaises(ValueError):
                save_manifest(path, {'gate': .21})

    def test_worker_gets_query_only_and_resume_keeps_incorrect_results(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); scene = root/'scene'; scene.mkdir()
            (scene/'objects_gt.json').write_text(json.dumps([obj(7, 'SECRET_GT_CATEGORY')]))
            output = root/'run'; output.mkdir()
            row = {'csv_row': 42, 'scene': 'scene', 'query': 'the chair', 'target_id': 7,
                   'stimulus_candidate_ids': [7, 8]}
            def worker(command, **kwargs):
                self.assertIn('--query', command)
                for forbidden in ('--target-id', '--stimulus-candidate-ids', '--gt'):
                    self.assertNotIn(forbidden, command)
                self.assertNotIn('SECRET_GT_CATEGORY', ' '.join(command))
                path = output/'q00042'
                (path/'prediction.json').write_text(json.dumps({'query': row['query'],
                    'object_table': 'v9_top5', 'mode': 'active_diagnostic', 'predicted_id': None,
                    'category_target_ids': [], 'bbox_3d': None}))
                return SimpleNamespace(returncode=0)
            with patch('seeground.cli.run_grounding_v9.subprocess.run', side_effect=worker) as process:
                result = run_case(row, root, output, 5, 3)
                self.assertFalse(result['correct'])
                self.assertNotIn('error', result)
                self.assertEqual(run_case(row, root, output, 5, 3), result)
                self.assertEqual(process.call_count, 1)

    def test_paired_comparison_requires_identical_settings_and_full_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            dirs = [Path(directory)/name for name in ('top1', 'top5')]
            manifest = {'rows': [{'csv_row': 1}, {'csv_row': 2}], 'models': 'same', 'categories': {
                'gate': {'top_k': 1, 'min_score': .2, 'max_gap': .03}, 'files': {},
                'retained_count_histogram': {}, 'mean_retained': 1}, 'api': {'api_slot_dir': 'one'}}
            for path, k, correct in zip(dirs, (1, 5), ((True, False), (False, True))):
                path.mkdir(); value = deepcopy(manifest); value['categories']['gate']['top_k'] = k
                (path/'manifest.json').write_text(json.dumps(value))
                (path/'results.json').write_text(json.dumps({'summary': {}, 'results': [
                    {'csv_row': i, 'correct': c} for i, c in enumerate(correct, 1)]}))
            result = compare(*dirs)
            self.assertEqual((result['gained'], result['lost']), (1, 1))
            value['models'] = 'different'
            (dirs[1]/'manifest.json').write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                compare(*dirs)


if __name__ == '__main__':
    unittest.main()
