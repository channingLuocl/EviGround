"""Native scores and overlapping masks must stay bound to the same instances."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from eviground.categories.export import NATIVE_EXPORT_VERSION, prepare_tables
from eviground.categories.hypotheses import (
    MASK3D_SCORE_TYPE, gate_categories, load_category_objects, sha256,
    SEEGROUND_SCORE_TYPE,
)
from eviground.categories.retrieval import match_retrieval_categories
from scripts.import_mask3d_categories import import_scene
from scripts.import_seeground_nr3d_categories import import_predictions
from baselines.seeground.upstream.data.scannet200_constants import CLASS_LABELS_200


class NativeCategoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data, self.pred = self.root / 'data', self.root / 'pred'
        self.output, self.alignment = self.root / 'prepared', self.root / 'raw'
        self.scene = 'synthetic'
        (self.data / 'inputs').mkdir(parents=True)
        (self.pred / 'scenes').mkdir(parents=True)
        (self.alignment / self.scene).mkdir(parents=True)
        self.xyz = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                               [1, 1, 1], [2, 2, 2], [3, 2, 2]], dtype=np.float32)
        self.rgb = np.arange(21, dtype=np.uint8).reshape(-1, 3)
        cloud = self.data / 'inputs' / (self.scene + '.npz')
        np.savez(cloud, xyz=self.xyz, rgb=self.rgb)
        masks = np.zeros((7, 2), dtype=bool)
        masks[:5, 0] = True
        masks[2:6, 1] = True
        self.masks = masks
        self.names = [CLASS_LABELS_200[1], *CLASS_LABELS_200[3:]]
        self.prob = np.zeros((2, 200), dtype=np.float32)
        for j, classes in enumerate([['chair', 'table', 'couch', 'bed', 'sink'],
                                    ['table', 'chair', 'couch', 'bed', 'sink']]):
            for name, score in zip(classes, [.55, .25, .1, .05, .025]):
                self.prob[j, self.names.index(name)] = score
        indices = np.argsort(-self.prob[:, :198], axis=1, kind='stable')[:, :5]
        self.arrays = self.pred / 'scenes' / (self.scene + '.npz')
        np.savez(self.arrays, vertices=7, packed_masks=np.packbits(masks, axis=0),
                 class_probabilities=self.prob, top5_class_indices=indices,
                 scores=np.asarray([.8, .7]), query_ids=np.asarray([4, 7]))
        self.proposal_path = self.arrays.with_suffix('.json')
        self.proposals = [dict(proposal_id=j, query_id=q, score=s,
            top1=self.names[int(indices[j, 0])],
            top5=[self.names[int(k)] for k in indices[j]],
            top5_scores=self.prob[j, indices[j]].tolist())
            for j, (q, s) in enumerate([(4, .8), (7, .7)])]
        self.proposal_path.write_text(json.dumps(dict(scene_id=self.scene, proposals=self.proposals,
                                                     arrays_sha256=sha256(self.arrays))))
        align = self.alignment / self.scene / (self.scene + '.txt')
        align.write_text('axisAlignment = 1 0 0 10 0 1 0 20 0 0 1 30 0 0 0 1\n')
        self.record = dict(input_sha256=sha256(cloud))
        self.raw_record = dict(raw_files=[dict(file=align.name, sha256=sha256(align))])

    def convert(self):
        return import_scene(self.scene, self.data, self.pred, self.alignment,
                            self.output, self.record, self.raw_record)

    def export(self):
        rows, source = self.convert()
        export = self.root / 'export'
        export.mkdir()
        config = dict(version=NATIVE_EXPORT_VERSION, category_source='seeground_mask3d_scannet200',
            score_type=MASK3D_SCORE_TYPE, sources=[source], frontend_protocol_sha256='frozen-protocol')
        data = dict(version=NATIVE_EXPORT_VERSION, summary=dict(complete=True, errors=0,
            count=len(rows), scenes=1, score_type=MASK3D_SCORE_TYPE), results=rows)
        self.save_export(export, config, data)
        return export, rows

    def save_export(self, path, config, data):
        (path / 'config.json').write_text(json.dumps(config))
        (path / 'predictions.json').write_text(json.dumps(data))
        (path / 'EXPORT_VERIFIED.json').write_text(json.dumps(dict(status='passed',
            files={name: sha256(path / name) for name in ('config.json', 'predictions.json')})))

    def test_original_scores_all_instances_and_overlap_survive_import(self):
        export, rows = self.export()
        scene = self.output / 'scenes' / self.scene
        with np.load(scene / 'scene.npz') as saved:
            self.assertEqual(len(saved['xyz']), 10)  # 1 background +5 +4; preserve overlap.
            for j in range(2):
                np.testing.assert_array_equal(saved['xyz'][saved['ids'] == j],
                                               self.xyz[self.masks[:, j]] + [10, 20, 30])
                np.testing.assert_array_equal(saved['rgb'][saved['ids'] == j], self.rgb[self.masks[:, j]])
        tables = self.root / 'tables'
        manifest = prepare_tables(export, self.output / 'scenes', tables, [self.scene], 5)
        self.assertEqual(manifest['instances'], 2)
        objects = load_category_objects(tables / self.scene / 'objects_top5.json')
        self.assertEqual(objects[0]['category_source'], 'seeground_mask3d_scannet200')
        self.assertEqual(objects[0]['category_score_type'], MASK3D_SCORE_TYPE)
        self.assertEqual([r['score'] for r in objects[0]['category_hypotheses']],
                         [r['score'] for r in rows[0]['top5']])
        # Both target and reference matching see all proposals, including lower ranks.
        self.assertEqual([r['bbox_id'] for r in match_retrieval_categories(['chair'], objects)], [0, 1])
        self.assertEqual([r['bbox_id'] for r in match_retrieval_categories(['table'], objects)], [0, 1])
        self.assertNotIn('gt_', json.dumps(rows))

    def test_reassigned_proposal_ids_are_rejected(self):
        value = json.loads(self.proposal_path.read_text())
        value['proposals'][1]['proposal_id'] = 0
        self.proposal_path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'ID mismatch'):
            self.convert()

    def test_changed_prediction_arrays_are_rejected(self):
        with self.arrays.open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.convert()

    def test_wrong_top5_scores_are_rejected_without_reranking(self):
        value = json.loads(self.proposal_path.read_text())
        value['proposals'][0]['top5_scores'][1] = .2
        self.proposal_path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'ID mismatch'):
            self.convert()

    def test_native_cache_cannot_omit_unreferenced_instance(self):
        export, _ = self.export()
        config = json.loads((export / 'config.json').read_text())
        data = json.loads((export / 'predictions.json').read_text())
        data['results'].pop()
        data['summary']['count'] -= 1
        self.save_export(export, config, data)
        with self.assertRaisesRegex(ValueError, 'every aligned instance'):
            prepare_tables(export, self.output / 'scenes', self.root / 'tables', [self.scene], 5)

    def test_native_cache_rejects_old_geometry_and_proposal_policy(self):
        export, _ = self.export()
        path = self.output / 'scenes' / self.scene / 'scene_instances.json'
        value = json.loads(path.read_text())
        value[0]['bbox_3d'][0] += 1
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'geometry changed'):
            prepare_tables(export, self.output / 'scenes', self.root / 'tables', [self.scene], 5)
        policy = self.root / 'old-policy.json'
        policy.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'old proposal policy'):
            prepare_tables(export, self.output / 'scenes', self.root / 'tables', [self.scene], 5,
                           proposal_policy=policy)

    def test_foreground_scores_are_not_renormalized_and_unknown_type_fails(self):
        scores = [dict(category=str(i), score=s) for i, s in enumerate([.3, .2, .1, .05, .01])]
        retained, _ = gate_categories(scores, score_type=MASK3D_SCORE_TYPE)
        self.assertEqual([r['score'] for r in retained], [.3, .2, .1, .05, .01])
        for score_type in ['probability', 'guessed_mask3d_scores']:
            with self.assertRaises(ValueError):
                gate_categories(scores, score_type=score_type)
        invalid = deepcopy(scores)
        invalid[0]['score'] = 1.1
        with self.assertRaises(ValueError):
            gate_categories(invalid, score_type=MASK3D_SCORE_TYPE)


class SeeGroundNr3DTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.scenes = self.root / 'source', self.root / 'scenes'
        self.scene = self.scenes / 'synthetic'
        self.scene.mkdir(parents=True)
        for name in ('raw_scores', 'provenance', 'scenes', 'author_top1_lookup_tables'):
            (self.source / name).mkdir(parents=True)
        self.vocab = ['chair', 'table', 'bed', 'couch', 'sink'] + [f'label{i}' for i in range(192)]
        self.matrix = np.full((2, 197), -5, dtype=np.float32)
        self.matrix[0, :5] = [4, 3, 2, 1, 0]
        self.matrix[1, :5] = [3, 4, 2, 1, 0]
        matrix = self.source / 'raw_scores/native_logits_3413x197.npy'
        np.save(matrix, self.matrix)
        (self.source / 'vocabulary.json').write_text(json.dumps(self.vocab))
        (self.source / 'provenance/RESULTS_FROZEN.json').write_text(
            json.dumps(dict(files={'eval/V0.npy': sha256(matrix)})))
        self.objects = [dict(scene_id='synthetic', object_id=i + 1,
            top1=self.vocab[int(np.argmax(self.matrix[i]))],
            top5=[dict(category=self.vocab[int(k)], score=float(self.matrix[i, k]), rank=j)
                  for j, k in enumerate(np.argsort(-self.matrix[i], kind='stable')[:5], 1)],
            bbox_3d=[i, 0, .5, 1, 1, 1]) for i in range(2)]
        self.predictions = dict(score_type=SEEGROUND_SCORE_TYPE,
                               source_scores_sha256=sha256(matrix), objects=self.objects)
        self.save_predictions()
        geometry = [dict(bbox_id=o['object_id'], bbox_3d=o['bbox_3d'], target='author-label')
                    for o in self.objects]
        (self.source / 'author_top1_lookup_tables/synthetic.json').write_text(json.dumps(geometry))
        (self.scene / 'objects.json').write_text(json.dumps(geometry))
        geometry.append(dict(bbox_id=99, bbox_3d=[3, 0, .5, 1, 1, 1], target='SECRET_GT_SHELL'))
        (self.scene / 'scene_instances.json').write_text(json.dumps(geometry))
        np.savez(self.scene / 'scene.npz', xyz=np.zeros((3, 3)), rgb=np.zeros((3, 3)),
                 ids=np.asarray([1, 2, 99]))
        # Evaluation-only data must not be consulted to form a candidate set.
        (self.source / 'evaluation_only').mkdir()
        (self.source / 'evaluation_only/forbidden.json').write_text('invalid JSON: SECRET_GT')

    def save_predictions(self):
        (self.source / 'predictions_top5.json').write_text(json.dumps(self.predictions))
        (self.source / 'scenes/synthetic.json').write_text(json.dumps(dict(
            scene_id='synthetic', score_type=SEEGROUND_SCORE_TYPE, objects=self.objects)))

    def test_native_scores_original_candidates_and_geometry_are_preserved(self):
        output = self.root / 'export'
        summary = import_predictions(self.source, self.scenes, output)
        manifest = prepare_tables(output, self.scenes, self.root / 'shared', ['synthetic'], 5)
        table = self.root / 'shared/synthetic/objects_top5.json'
        objects = load_category_objects(table)
        self.assertFalse(summary['evaluation_only_read'])
        self.assertFalse(summary['model_rerun'])
        self.assertEqual(manifest['category_source'], 'seeground_pointnext_clip_reproduction')
        self.assertEqual([o['bbox_id'] for o in objects], [1, 2])
        self.assertEqual(objects[0]['category_score_type'], SEEGROUND_SCORE_TYPE)
        self.assertEqual([h['score'] for h in objects[0]['category_hypotheses']], [4, 3, 2, 1, 0])
        self.assertEqual([o['bbox_id'] for o in match_retrieval_categories(['chair'], objects)], [1, 2])
        self.assertNotIn('SECRET_GT', table.read_text())
        self.assertEqual(objects[1]['bbox_3d'], self.objects[1]['bbox_3d'])

    def test_reordered_raw_rows_and_fabricated_scores_fail(self):
        for variant in ('reorder', 'score'):
            with self.subTest(variant=variant):
                if variant == 'reorder':
                    self.objects.reverse()
                else:
                    self.objects[0]['top5'][1]['score'] += .1
                self.save_predictions()
                with self.assertRaisesRegex(ValueError, 'score/rank alignment'):
                    import_predictions(self.source, self.scenes, self.root / variant)
                if variant == 'reorder':
                    self.objects.reverse()

    def test_new_category_cannot_be_attached_to_different_geometry(self):
        self.objects[0]['bbox_3d'][0] += 1
        self.save_predictions()
        with self.assertRaisesRegex(ValueError, 'geometry differ'):
            import_predictions(self.source, self.scenes, self.root / 'export')

    def test_omitting_non_target_candidate_fails_the_original_candidate_set(self):
        self.objects.pop()
        self.matrix = self.matrix[:1]
        matrix = self.source / 'raw_scores/native_logits_3413x197.npy'
        np.save(matrix, self.matrix)
        self.predictions['source_scores_sha256'] = sha256(matrix)
        (self.source / 'provenance/RESULTS_FROZEN.json').write_text(
            json.dumps(dict(files={'eval/V0.npy': sha256(matrix)})))
        self.save_predictions()
        with self.assertRaisesRegex(ValueError, 'candidate IDs or geometry differ'):
            import_predictions(self.source, self.scenes, self.root / 'export')


if __name__ == '__main__':
    unittest.main()
