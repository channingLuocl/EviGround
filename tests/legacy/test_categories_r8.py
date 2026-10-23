"""r8 score semantics, verified input identity, alternate bindings and model payloads."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from seeground.categories.categories_r8 import EXPORT_VERSION, load_export
from seeground.categories.categories_v9 import (R8_SCORE_TYPE, R8_VERSION, category_fields, gate_categories,
                           load_category_objects, prepare_tables, sha256)
from seeground.grounding.evidence import match_retrieval_categories
from seeground.grounding.real_scene import run_query
from tests.legacy.test_v7_evidence import obj, parse


class R8Tests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root/'export'; self.source.mkdir()
        self.scene = self.root/'scenes'/'test'; self.scene.mkdir(parents=True)
        self.tables = self.root/'tables'
        np.savez(self.scene/'scene.npz', xyz=np.array([[0., 0., 0.], [1., 1., 1.]]),
                 rgb=np.zeros((2, 3), dtype=np.uint8), ids=np.array([1, 2]))
        self.rows = []
        for oid, names in ((1, ['clothes', 'chair', 'box', 'bag', 'sofa']),
                           (2, ['table', 'desk', 'box', 'bag', 'sofa'])):
            self.rows.append({**obj(oid, names[0]), 'scene': 'test', 'status': 'completed',
                'predicted_object': names[0], 'score_type': R8_SCORE_TYPE,
                'top5': [{'category': n, 'score': s, 'rank': k} for k, (n, s) in
                         enumerate(zip(names, (3.4, 2.6, 1.5, .1, -.2)), 1)]})
        self.write_source()
        env = patch.dict(os.environ, {'SEEGROUND_V4': '1', 'SEEGROUND_V7': '1',
            'SEEGROUND_V7_4': '1', 'SEEGROUND_V9': '1', 'SEEGROUND_V9_SOURCE': 'r8',
            'SEEGROUND_REACT_BACKEND': 'openai', 'SEEGROUND_API_PROVIDER': 'openai',
            'SEEGROUND_V9_CATEGORY_ROOT': str(self.tables)})
        env.start(); self.addCleanup(env.stop)

    def write_source(self):
        (self.source/'predictions.json').write_text(json.dumps({'version': EXPORT_VERSION,
            'summary': {'complete': True, 'errors': 0, 'count': 2, 'scenes': 1,
                        'score_type': R8_SCORE_TYPE}, 'results': self.rows}))
        (self.source/'config.json').write_text(json.dumps({'version': EXPORT_VERSION,
            'score_type': R8_SCORE_TYPE, 'sources': [
                {'scene': 'test', 'cloud_sha256': sha256(self.scene/'scene.npz')}]}))
        (self.source/'EXPORT_VERIFIED.json').write_text(json.dumps({'status': 'passed',
            'files': {n: sha256(self.source/n) for n in ('predictions.json', 'config.json')}}))

    def prepare(self, k=5):
        return prepare_tables(self.source, self.root/'scenes', self.tables, ['test'], top_k=k)

    def test_rank_only_retains_raw_fusion_scores_and_negative_alternative(self):
        result = self.prepare()
        objects = load_category_objects(self.tables/'test'/'objects_top5.json')
        self.assertEqual((result['version'], result['mean_retained']), (R8_VERSION, 5))
        hypotheses = objects[0]['category_hypotheses']
        self.assertEqual([h['score'] for h in hypotheses], [3.4, 2.6, 1.5, .1, -.2])
        self.assertEqual(hypotheses[-1]['gap_from_top1'], 3.6)
        self.assertEqual(category_fields(objects[0])['category_score_type'], R8_SCORE_TYPE)
        self.assertEqual([o['bbox_id'] for o in match_retrieval_categories(['chair'], objects)], [1])
        self.assertEqual([o['bbox_id'] for o in match_retrieval_categories(['desk'], objects)], [2])
        self.prepare(1)
        control = load_category_objects(self.tables/'test'/'objects_top1.json')
        self.assertEqual(len(control[0]['category_hypotheses']), 1)
        self.assertEqual(match_retrieval_categories(['chair'], control), [])
        with self.assertRaisesRegex(ValueError, 'omit cosine'):
            prepare_tables(self.source, self.root/'scenes', self.tables, ['test'], min_score=.2)

    def test_zip_and_directory_have_identical_identity_and_reject_modified_sources(self):
        path = self.root/'export.zip'
        with ZipFile(path, 'w') as archive:
            for p in self.source.iterdir():
                archive.write(p, 'export/'+p.name)
        zipped = prepare_tables(path, self.root/'scenes', self.root/'zip_tables', ['test'])
        self.assertEqual(zipped, self.prepare())
        (self.source/'predictions.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'file changed'):
            load_export(self.source)

    def test_reject_inconsistent_top1_rank_duplicate_ids_and_changed_cloud(self):
        for change in ('top1', 'rank', 'duplicate'):
            with self.subTest(change=change):
                original = json.loads(json.dumps(self.rows))
                if change == 'top1':
                    self.rows[0]['predicted_object'] = 'chair'
                elif change == 'rank':
                    self.rows[0]['top5'][1]['rank'] = 5
                else:
                    self.rows[1]['bbox_id'] = 1
                self.write_source()
                with self.assertRaises(ValueError):
                    load_export(self.source)
                self.rows = original
        self.write_source()
        np.savez(self.scene/'scene.npz', ids=np.array([999]))
        with self.assertRaisesRegex(ValueError, 'Point cloud differs'):
            self.prepare()

    def test_second_rank_target_and_reference_reach_scene_decision_without_gt(self):
        self.prepare()
        # Only the structural geometry file exists; no GT/author object table.
        (self.scene/'scene_instances.json').write_text(json.dumps([obj(99, 'SECRET_GT_LABEL')]))
        query = 'the chair at the bottom of the stack beside the desk'
        parsed = parse(query, 'chair', [], anchors=['desk'], unhandled=['at the bottom of the stack'])
        reply = {'_content': json.dumps({'predicted_id': 1, 'reason': 'Visible chair beside desk',
            'evidence': [{'source_quote': query, 'object_ids': [1, 2],
                          'observation': 'Chair at base beside desk'}], 'unresolved': []}),
            '_finish_reason': 'stop', '_baseline': {}, 'usage': {}}
        with patch('seeground.grounding.real_scene.parse_query', return_value=parsed), \
                patch('seeground.geometry.wall_geometry.attach_wall_geometry', side_effect=lambda objects, *args: objects), \
                patch('seeground.grounding.v74_grounding.scene_views', return_value=[{'image': 'test.png'}]), \
                patch('seeground.grounding.v74_grounding.image_part', return_value={}), \
                patch('seeground.grounding.v74_grounding.chat', return_value=reply) as model:
            result = run_query(self.scene, query, self.root/'run', mode='active_diagnostic',
                               object_table='v9_top5')
        self.assertEqual(result['predicted_id'], 1)
        self.assertEqual(result['implementation_version'], R8_VERSION)
        self.assertEqual(result['category_anchor_ids'], [2])
        self.assertEqual(result['category_bindings']['targets'][0]['matched_hypotheses'][0]['rank'], 2)
        payload = json.loads(model.call_args.args[0][1]['content'][0]['text'])
        self.assertEqual(payload['allowed_target_ids'], [1])
        self.assertEqual(payload['objects'][0]['category_score_type'], R8_SCORE_TYPE)
        self.assertEqual(len(payload['objects'][0]['category_hypotheses']), 5)
        self.assertNotIn('SECRET_GT_LABEL', json.dumps(payload))
        self.assertFalse(result['gt_categories_supplied_to_model'])


if __name__ == '__main__':
    unittest.main()
