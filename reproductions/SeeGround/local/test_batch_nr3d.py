import csv
import tempfile
import unittest
from pathlib import Path
from batch_nr3d import select_rows, summarize
from batch_nr3d import experiment_settings
from unittest.mock import patch
import io
import json
from contextlib import redirect_stdout
from batch_nr3d import main

class BatchTests(unittest.TestCase):
    def test_gt_batch_binds_selected_table_and_rejects_pred_cache_reuse(self):
        import hashlib
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);scene=root/'local_outputs'/'scene';scene.mkdir(parents=True)
            gt=[{'bbox_id':1,'target':'trash can','bbox_3d':[0,0,0,1,1,1]}]
            pred=[{**gt[0],'target':'water cooler'}]
            (scene/'objects_gt.json').write_text(json.dumps(gt))
            (scene/'objects.json').write_text(json.dumps(pred))
            (scene/'scene.npz').write_bytes(b'fixture')
            data=root/'queries.csv'
            data.write_text('scan_id,correct_guess,utterance,target_id,stimulus_id\nscene,True,trash can,1,fixture\n')
            output=root/'experiment'
            def mock_run(scene_dir,query,out,*args,**kwargs):
                self.assertEqual(kwargs['object_table'],'gt')
                self.assertNotIn('target_id',kwargs)
                out.mkdir(parents=True)
                (out/'prediction.json').write_text(json.dumps({'query':query,'object_table':'gt',
                    'predicted_id':1,'bbox_3d':gt[0]['bbox_3d'],'response':'mock','model':'mock'}))
            argv=['batch_nr3d.py','--csv',str(data),'--scene','scene','--count','1',
                  '--output',str(output),'--object-table','gt']
            with patch('sys.argv',argv),patch('batch_nr3d.ROOT',root), \
                 patch('batch_nr3d.run_query',side_effect=mock_run),redirect_stdout(io.StringIO()):
                main()
            result=json.loads((output/'results.json').read_text())
            self.assertEqual(result['manifest']['settings']['object_table'],'gt')
            self.assertEqual(result['manifest']['objects_sha256'],hashlib.sha256((scene/'objects_gt.json').read_bytes()).hexdigest())
            self.assertIn('oracle OLT',(output/'report.html').read_text())
            argv[-1]='pred'
            with patch('sys.argv',argv),patch('batch_nr3d.ROOT',root), \
                 patch('batch_nr3d.run_query') as api:
                with self.assertRaisesRegex(ValueError,'不同实验'):main()
                api.assert_not_called()

    def test_batch_failure_prints_reason_and_keeps_stage_and_latency(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);scene=root/'local_outputs'/'scene';scene.mkdir(parents=True)
            objects=[{'bbox_id':1,'target':'window','bbox_3d':[0,0,0,1,1,1]}]
            for name in ('objects.json','objects_gt.json'):
                (scene/name).write_text(json.dumps(objects))
            (scene/'scene.npz').write_bytes(b'fixture')
            data=root/'queries.csv'
            data.write_text('scan_id,correct_guess,utterance,target_id,stimulus_id\nscene,True,window,1,fixture\n')
            output=root/'experiment'
            def failed_run(scene_dir,query,out,*args,**kwargs):
                out.mkdir(parents=True)
                (out/'run.json').write_text(json.dumps({'query':query,'error_stage':'parse_query',
                                                       'pipeline_latency_s':.25,'calls':[]}))
                raise ValueError('fixture parser failure')
            console=io.StringIO()
            with patch('sys.argv',['batch_nr3d.py','--csv',str(data),'--scene','scene',
                                   '--count','1','--output',str(output)]), \
                 patch('batch_nr3d.ROOT',root),patch('batch_nr3d.run_query',side_effect=failed_run), \
                 redirect_stdout(console):
                main()
            self.assertIn('ERROR [parse_query/ValueError]: fixture parser failure',console.getvalue())
            result=json.loads((output/'results.json').read_text())
            self.assertEqual(result['results'][0]['error_stage'],'parse_query')
            self.assertEqual(result['summary']['mean_pipeline_latency_s'],.25)
            self.assertEqual(result['summary']['completed'],0)

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
            gt=experiment_settings(scene,'geometry',object_table='gt')
            self.assertEqual(gt['objects_file'],'objects_gt.json')
            self.assertNotEqual(first,gt)

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
