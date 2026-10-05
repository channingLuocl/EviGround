import csv
import tempfile
import unittest
from pathlib import Path
from batch_nr3d import select_rows, summarize

class BatchTests(unittest.TestCase):
    def test_errors_remain_in_denominator(self):
        stats=summarize([{'predicted_id':2,'correct':True},{'predicted_id':3,'correct':False},{'error':'timeout','correct':False}])
        self.assertEqual(stats,{'total':3,'completed':2,'correct':1,'accuracy':1/3,'iou_evaluated':0,'acc25':None,'acc50':None,'mean_iou':None,'mean_latency_s':None})

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
