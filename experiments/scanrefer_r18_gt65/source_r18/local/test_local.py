import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
import run
from qwen_api import QwenAPIError, parse_predicted_id

class LocalChecks(unittest.TestCase):
    def test_depth_keeps_front_object(self):
        with tempfile.TemporaryDirectory() as td:
            xyz=np.array([[0.,0.,0.],[0.,1.,0.]])
            meta=run.render(xyz,np.array([[255,0,0],[0,0,255]],dtype=np.uint8),Path(td)/'x.png',eye=[0,-3,0],at=[0,0,0],ids=np.array([1,2]))
            self.assertEqual(meta['marked_ids'],[1])
    def test_point_disks_keep_nearest_color_and_instance_at_each_pixel(self):
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/'disks.png'
            xyz=np.array([[0.,0.,0.],[0.,1.,0.]])
            colors=np.array([[10,150,30],[0,0,255]],dtype=np.uint8)
            meta=run.render(xyz,colors,output,eye=[0,-3,0],at=[0,0,0],
                            ids=np.array([1,2]),size=100,point_radius=2)
            self.assertEqual(meta['marked_ids'],[1])
            self.assertEqual(meta['visible_instance_pixels'],{'1':13})
            self.assertEqual(meta['visible_pixels'],13)
            run.render(xyz,colors,output,eye=[0,-3,0],at=[0,0,0],
                       size=100,point_radius=2)
            pixels=np.asarray(Image.open(output))
            self.assertEqual(int(np.sum(np.all(pixels==colors[0],axis=2))),13)
            self.assertFalse(np.any(np.all(pixels==colors[1],axis=2)))
    def test_point_disks_clip_to_frame_edges(self):
        with tempfile.TemporaryDirectory() as td:
            xyz=np.array([[0.,0.,0.]])
            with patch('run.project',return_value=(np.array([[0.,0.]]),np.array([1.]))):
                meta=run.render(xyz,np.array([[20,30,40]],dtype=np.uint8),Path(td)/'edge.png',
                                ids=np.array([3]),size=100,point_radius=2)
            self.assertEqual(meta['visible_pixels'],6)
            self.assertEqual(meta['visible_instance_pixels'],{'3':6})
    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'o.json'
            p.write_text(json.dumps([{'bbox_id':1,'target':'chair','bbox_3d':[0,0,0,1,1,1]}]*2))
            with self.assertRaises(ValueError):run.load_objects(p)
    def test_dry_run_never_calls_network(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);run.demo(p)
            args=SimpleNamespace(image=str(p/'scene.png'),objects=str(p/'objects.json'),query='red chair',dry_run=True)
            with patch('run.chat') as request:
                run.ground(args);request.assert_not_called()
    def test_mock_api_maps_id_to_bbox(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);run.demo(p)
            args=SimpleNamespace(image=str(p/'scene.png'),objects=str(p/'objects.json'),query='laptop',dry_run=False,output=str(p/'result.json'))
            api_data={'choices':[{'message':{'content':'Predicted ID: 2\nExplanation: test fixture'}}], 'usage': {'total_tokens': 3}}
            with patch.dict('os.environ',{'SEEGROUND_BASE_URL':'https://example.invalid/v1','SEEGROUND_MODEL':'mock-model','SEEGROUND_API_KEY':'test-only'}),patch('run.chat',return_value={'_content':'Predicted ID: 2\nExplanation: test fixture','_baseline':{'model':'mock-model','latency_s':0.01},'usage':api_data['usage']}) as request:
                run.ground(args)
                request.assert_called_once()
            data=json.loads((p/'result.json').read_text())
            self.assertEqual(data['predicted_id'],2)
            self.assertEqual(data['bbox_3d'],[-.3,0.,.7,.5,.35,.1])
            self.assertNotIn('diagnosis_policy',data)
            self.assertNotIn('Diagnosis decision policy',request.call_args.args[0][0]['content'][-1]['text'])

    def test_unresolved_diagnoses_allow_prediction_outside_audit_candidates(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);run.demo(p)
            for status in ('missing','ambiguous','conflict'):
                with self.subTest(status=status):
                    evidence={'diagnosis':{'status':status,'candidate_ids':[1],
                                          'stable_candidate_ids':[1],
                                          'hypotheses':[{'candidate_ids':[1]}]}}
                    args=SimpleNamespace(image=str(p/'scene.png'),objects=str(p/'objects.json'),
                                         query='laptop',dry_run=False,output=str(p/'result.json'),evidence=evidence)
                    with patch('run.chat',return_value={'_content':'Predicted ID: 2',
                               '_baseline':{'model':'mock','latency_s':.01}}) as request:
                        result=run.ground(args)
                    prompt=request.call_args.args[0][0]['content'][-1]['text']
                    self.assertIn('Diagnosis decision policy: audit_only',prompt)
                    self.assertIn('Do not use these sets to exclude, prioritize or break ties',prompt)
                    self.assertIn('full supplied object table and inspection images',prompt)
                    self.assertNotIn('Use the unique stable_candidate_ids entry as strong supporting evidence',prompt)
                    self.assertEqual(result['predicted_id'],2)
                    self.assertEqual(result['diagnosis_policy'],'audit_only')
                    self.assertEqual(json.loads((p/'result.json').read_text())['diagnosis_policy'],'audit_only')

    def test_id_parser_accepts_common_qwen_formats(self):
        self.assertEqual(parse_predicted_id('Object ID: 3', {1, 3}), 3)
        with self.assertRaises(QwenAPIError):
            parse_predicted_id('Predicted ID: 9', {1, 3})

if __name__=='__main__':unittest.main()
