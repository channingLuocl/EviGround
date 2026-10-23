"""Verify point-cloud-only inference and evaluation-label independence."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import eviground.experiments.predict_pointcloud_v8 as inference
import eviground.experiments.run_object_classification as batch
from eviground.api.qwen_api import QwenAPIError
from tests.legacy.test_object_classification_v8 import ENV, fixture, response


class PointCloudInputTests(unittest.TestCase):
    def inputs(self, root):
        _, xyz, ids = fixture(root)
        path = root/'input.npz'
        np.savez(path, xyz=xyz, rgb=np.full(xyz.shape, 100, dtype=np.uint8), ids=ids)
        return argparse.Namespace(cloud=path, output=root/'predictions', bbox_id=[3],
                                  workers=1, retry_errors=False, prepare_only=False)

    def test_real_inference_runs_without_gt_author_manifest_or_mesh(self):
        with tempfile.TemporaryDirectory() as td, patch.dict('os.environ', ENV), \
                redirect_stdout(StringIO()) as stdout, redirect_stderr(StringIO()):
            root = Path(td)
            args = self.inputs(root)
            # Remove every scene/label fixture; only the point cloud remains.
            import shutil
            shutil.rmtree(root/'scenes')
            (root/'manifest.json').unlink()
            with patch('eviground.classifiers.object_classifier_v8.chat', return_value=response()) as chat:
                self.assertEqual(inference.predict(args), 0)
            result = json.loads((args.output/'predictions.json').read_text())
            self.assertEqual(result['results'][0]['predicted_object'], 'chair')
            self.assertEqual(set(json.loads(stdout.getvalue())), {'bbox_id','predicted_object','status'})
            self.assertTrue(all(v['surface_mode'] == 'point_splats'
                                for v in result['results'][0]['geometry']['views']))
            request = json.dumps(chat.call_args.args[0])
            for forbidden in ('author-wrong', 'secret-gt', 'secret query'):
                self.assertNotIn(forbidden, request)
            self.assertNotIn('target', result['results'][0])
            with patch('eviground.experiments.predict_pointcloud_v8.classify_instance') as classify:
                inference.predict(args)
                classify.assert_not_called()

    def test_api_error_is_null_and_retry_archives_it_without_label_fallback(self):
        with tempfile.TemporaryDirectory() as td, patch.dict('os.environ', ENV), \
                patch('eviground.experiments.predict_pointcloud_v8.classify_instance') as classify, \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            args = self.inputs(Path(td))
            classify.side_effect = QwenAPIError('network',diagnostic={'failure_kind':'network'})
            self.assertEqual(inference.predict(args),1)
            saved = json.loads((args.output/'predictions.json').read_text())
            self.assertIsNone(saved['results'][0]['predicted_object'])
            self.assertEqual(saved['errors'],1)
            args.retry_errors=True
            classify.side_effect=None
            classify.return_value={'bbox_id':3,'predicted_object':'stool','status':'completed'}
            self.assertEqual(inference.predict(args),0)
            self.assertEqual(len(list(args.output.glob('object_*/attempts/*/prediction.json'))),1)
            with patch.dict('os.environ',{'EVIGROUND_MODEL':'changed'}), \
                    self.assertRaisesRegex(ValueError,'changed'):
                inference.predict(args)

    def test_changing_reference_labels_only_changes_scores(self):
        with tempfile.TemporaryDirectory() as td, patch.dict('os.environ', ENV), \
                patch('eviground.experiments.run_object_classification.classify_instance') as classify, \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            root=Path(td)
            args,_,_=fixture(root)
            received=[]
            def predict(xyz,rgb,ids,oid,out,vocab):
                received.append((xyz.copy(),rgb.copy(),ids.copy(),oid,list(vocab)))
                return {'bbox_id':oid,'predicted_object':'chair','status':'completed'}
            classify.side_effect=predict
            batch.run(args)
            before=json.loads((args.output/'results.json').read_text())['results'][0]
            scene=args.scene_root/'scene0001_00'
            for filename,label in [('objects.json','reference-changed'),('objects_gt.json','stool')]:
                path=scene/filename
                table=json.loads(path.read_text())
                for row in table: row['target']=label
                path.write_text(json.dumps(table))
            args.output=root/'changed-label-output'
            batch.run(args)
            after=json.loads((args.output/'results.json').read_text())['results'][0]
            self.assertEqual(before['predicted_object'],after['predicted_object'])
            self.assertTrue(before['correct'])
            self.assertFalse(after['correct'])
            for a,b in zip(received[0][:3],received[1][:3]): np.testing.assert_array_equal(a,b)
            self.assertEqual(received[0][3:],received[1][3:])

    def test_invalid_cloud_and_instance_ids_fail_before_requests(self):
        with tempfile.TemporaryDirectory() as td, patch.dict('os.environ', ENV), \
                patch('eviground.experiments.predict_pointcloud_v8.classify_instance') as classify:
            args=self.inputs(Path(td))
            args.bbox_id=[99]
            with self.assertRaisesRegex(ValueError,'present'):
                inference.predict(args)
            np.savez(args.cloud,xyz=np.zeros((2,3)),rgb=np.zeros((2,3)),ids=np.array([.5,1.]))
            with self.assertRaisesRegex(ValueError,'integer instance'):
                inference.predict(args)
            classify.assert_not_called()


if __name__=='__main__':
    unittest.main()
