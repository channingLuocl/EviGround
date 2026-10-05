import csv
import tempfile
import unittest
from pathlib import Path
from batch_nr3d import select_rows, summarize
from batch_nr3d import experiment_settings
from unittest.mock import patch

class BatchTests(unittest.TestCase):
    def test_errors_remain_in_denominator(self):
        stats=summarize([{'predicted_id':2,'correct':True},{'predicted_id':3,'correct':False},{'error':'timeout','correct':False}])
        self.assertEqual(stats,{'total':3,'completed':2,'correct':1,'accuracy':1/3,'iou_evaluated':0,'acc25':None,'acc50':None,'mean_iou':None,'mean_latency_s':None,'mean_pipeline_latency_s':None})

    def test_failed_requests_count_in_iou_denominator(self):
        stats=summarize([{'iou_3d':1.,'pipeline_latency_s':5.},{'error':'timeout'}])
        self.assertEqual(stats['acc25'],.5)
        self.assertEqual(stats['acc50'],.5)
        self.assertEqual(stats['mean_iou'],.5)
        self.assertEqual(stats['mean_pipeline_latency_s'],5.)

    def test_diagnostic_acceptance_does_not_change_accuracy_denominator(self):
        stats=summarize([{'decision_status':'resolved','accepted':True,'correct':True},
                         {'decision_status':'missing','accepted':False,'correct':False},
                         {'error':'timeout','correct':False}])
        self.assertEqual(stats['accuracy'],1/3)
        self.assertEqual(stats['accepted_count'],1)
        self.assertEqual(stats['accepted_accuracy'],1.)
        self.assertEqual(stats['decision_status_counts']['missing'],1)

    def test_diagnostic_manifest_tracks_rgb_and_budget(self):
        import json
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            scene=Path(td)
            (scene/'scene.npz').write_bytes(b'fixture');(scene/'objects_gt.json').write_text('[]')
            (scene/'objects.json').write_text('[{"bbox_id":1,"target":"tv","bbox_3d":[0,0,0,1,1,1]}]')
            first=experiment_settings(scene,'diagnostic',2)
            self.assertNotEqual(first,experiment_settings(scene,'diagnostic',6))
            Image.new('RGB',(40,40),'white').save(scene/'crop.png')
            (scene/'rgb_crops.json').write_text(json.dumps({'scene':scene.name,'crops':[
                {'id':1,'image':'crop.png','binding':'projected_instance','source_frame':'001.jpg'}]}))
            self.assertNotEqual(first,experiment_settings(scene,'diagnostic',2))

    def test_manifest_tracks_mode_and_model_without_key(self):
        with tempfile.TemporaryDirectory() as d:
            scene=Path(d)
            (scene/'scene.npz').write_bytes(b'fixture')
            (scene/'objects_gt.json').write_text('[]')
            with patch.dict('os.environ',{'SEEGROUND_MODEL':'model-A','SEEGROUND_API_KEY':'secret-fixture'}):
                first=experiment_settings(scene,'geometry')
                second=experiment_settings(scene,'visual_review')
            self.assertNotEqual(first,second)
            self.assertEqual(first['model'],'model-A')
            self.assertNotIn('secret-fixture',str(first))
            self.assertNotEqual(experiment_settings(scene,'category_review',2),
                                experiment_settings(scene,'category_review',6))

    def test_selection_filters_before_limiting_and_preserves_utterance(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.csv'
            with p.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['scan_id','correct_guess','utterance','target_id','stimulus_id']);w.writeheader()
                for scene,correct,query in [('other','True','skip'),('scene','False','skip'),('scene','True','The left window.'),('scene','True','Another.')]:
                    w.writerow(dict(scan_id=scene,correct_guess=correct,utterance=query,target_id=24,stimulus_id='scene-window-2-24-23'))
            rows,n=select_rows(p,'scene',1)
            self.assertEqual(n,2);self.assertEqual(rows[0]['query'],'The left window.');self.assertEqual(rows[0]['csv_row'],4)

if __name__=='__main__':unittest.main()
