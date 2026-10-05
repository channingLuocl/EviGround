import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import run
from qwen_api import QwenAPIError, parse_predicted_id

class LocalChecks(unittest.TestCase):
    def test_depth_keeps_front_object(self):
        with tempfile.TemporaryDirectory() as td:
            xyz=np.array([[0.,0.,0.],[0.,1.,0.]])
            meta=run.render(xyz,np.array([[255,0,0],[0,0,255]],dtype=np.uint8),Path(td)/'x.png',eye=[0,-3,0],at=[0,0,0],ids=np.array([1,2]))
            self.assertEqual(meta['marked_ids'],[1])
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

    def test_id_parser_accepts_common_qwen_formats(self):
        self.assertEqual(parse_predicted_id('Object ID: 3', {1, 3}), 3)
        with self.assertRaises(QwenAPIError):
            parse_predicted_id('Predicted ID: 9', {1, 3})

if __name__=='__main__':unittest.main()
