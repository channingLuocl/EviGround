import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
from seeground.experiments.paired_nr3d import select_plan, check_plan, run_paired, compare_pairs


class PairedTests(unittest.TestCase):
    def inputs(self, root):
        data=root/'queries.csv';olt=root/'olt.zip'
        with data.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['scan_id','correct_guess','dataset','utterance','target_id','stimulus_id'])
            writer.writeheader()
            for scene in ('scene0011_00','scene0100_00','scene0101_00','scene0102_00'):
                for i in range(4):
                    writer.writerow({'scan_id':scene,'correct_guess':i!=3,'dataset':'nr3d',
                        'utterance':f'{scene} chair {i}','target_id':i%2+1,'stimulus_id':'evaluation-only'})
        with ZipFile(olt,'w') as z:
            for scene in ('scene0011_00','scene0100_00','scene0101_00'):
                z.writestr(f'seeground_object_lookup_table/nr3d/gt/{scene}.json','[]')
        return data,olt

    def prepare(self, root, plan):
        for entry in plan['scenes']:
            scene=root/'local_outputs'/entry['scene'];scene.mkdir(parents=True)
            objects=[{'bbox_id':i,'target':'chair','bbox_3d':[i,0,0,1,1,1]} for i in (1,2)]
            (scene/'objects_gt.json').write_text(json.dumps(objects))
            (scene/'scene.npz').write_bytes(b'mocked cloud')

    def test_sampling_is_fixed_excludes_development_and_requires_olt(self):
        with tempfile.TemporaryDirectory() as td:
            data,olt=self.inputs(Path(td))
            plan=select_plan(data,olt,2,2,2026,['scene0011_00'])
            self.assertEqual(plan,select_plan(data,olt,2,2,2026,['scene0011_00']))
            self.assertEqual([s['scene'] for s in plan['scenes']],['scene0100_00','scene0101_00'])
            self.assertEqual(plan['total'],4)
            self.assertTrue(all(s['available']==3 and len(s['rows'])==2 for s in plan['scenes']))
            check_plan(plan,data)
            with self.assertRaisesRegex(ValueError,'场景不足'):
                select_plan(data,olt,3,2,2026,['scene0011_00'])

    def test_rejects_tampered_rows_and_changed_csv(self):
        with tempfile.TemporaryDirectory() as td:
            data,olt=self.inputs(Path(td));plan=select_plan(data,olt,2,2,2026,['scene0011_00'])
            changed=json.loads(json.dumps(plan));changed['scenes'][0]['rows'][0]['target_id']=999
            with self.assertRaisesRegex(ValueError,'CSV 不一致'):check_plan(changed,data)
            data.write_text(data.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'哈希不一致'):check_plan(plan,data)

    def test_missing_later_scene_stops_before_any_api_call(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);data,olt=self.inputs(root);plan=select_plan(data,olt,2,2,2026,['scene0011_00'])
            self.prepare(root,plan)
            (root/'local_outputs'/plan['scenes'][-1]['scene']/'scene.npz').unlink()
            with patch('seeground.experiments.paired_nr3d.ROOT',root),patch('seeground.evaluation.batch_nr3d.run_query') as api:
                with self.assertRaisesRegex(ValueError,'尚未调用 API'):
                    run_paired(plan,data,root/'experiment')
                api.assert_not_called()
            self.assertFalse((root/'experiment').exists())

    def test_pair_report_matches_by_scene_and_row_and_counts_failures(self):
        def result(scene,row,correct,pred):
            return {'scene':scene,'csv_row':row,'query':'chair','target_id':1,
                    'stimulus_id':'fixture','correct':correct,'predicted_id':pred}
        a=[result('scene0100_00',2,True,1),result('scene0101_00',3,False,None)]
        b=[result('scene0101_00',3,True,1),result('scene0100_00',2,False,2)]
        summary,pairs=compare_pairs(a,b)
        self.assertEqual(summary['baseline']['accuracy'],.5)
        self.assertEqual(summary['diagnostic']['accuracy'],.5)
        self.assertEqual(summary['completed_pairs'],1)
        self.assertEqual(summary['changes']['improved'],1)
        self.assertEqual(summary['changes']['regressed'],1)
        self.assertEqual(pairs[0]['diagnostic_id'],2)
        with self.assertRaisesRegex(ValueError,'两组题目不一致'):compare_pairs(a,b[:1])

    def test_paired_run_uses_same_queries_resumes_and_binds_settings(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);data,olt=self.inputs(root);plan=select_plan(data,olt,2,2,2026,['scene0011_00'])
            self.prepare(root,plan);output=root/'experiment';calls=[]
            def mock_run(scene_dir,query,out,source,**kwargs):
                self.assertEqual(kwargs['object_table'],'gt')
                self.assertNotIn('target_id',kwargs)
                self.assertNotIn('evaluation-only',source)
                calls.append((scene_dir.name,query,kwargs['mode']))
                out.mkdir(parents=True,exist_ok=True)
                pred=1 if kwargs['mode']=='baseline' else 2
                (out/'prediction.json').write_text(json.dumps({'query':query,'object_table':'gt',
                    'predicted_id':pred,'bbox_3d':[pred,0,0,1,1,1],'response':'mock','model':'mock'}))
            with patch('seeground.experiments.paired_nr3d.ROOT',root),patch('seeground.evaluation.batch_nr3d.ROOT',root), \
                 patch('seeground.evaluation.batch_nr3d.run_query',side_effect=mock_run),redirect_stdout(io.StringIO()):
                summary=run_paired(plan,data,output)
                self.assertEqual(len(calls),8)
                self.assertEqual({(s,q) for s,q,m in calls if m=='baseline'},
                                 {(s,q) for s,q,m in calls if m=='diagnostic'})
                self.assertEqual(summary['planned_total'],4)
                self.assertEqual(summary['completed_pairs'],4)
                report=json.loads((output/'comparison.json').read_text())
                self.assertEqual(len(report['per_scene']),2)
                self.assertEqual(len(report['pairs']),4)
                run_paired(plan,data,output)
                self.assertEqual(len(calls),8)
                with self.assertRaisesRegex(ValueError,'不同实验'):
                    run_paired(plan,data,output,review_limit=3)
                self.assertEqual(len(calls),8)
            with (output/'comparison.csv').open(newline='') as f:
                self.assertEqual(len(list(csv.DictReader(f))),4)


if __name__=='__main__':unittest.main()
