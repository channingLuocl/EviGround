import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from category_review import check_categories


def reply(content, finish='stop'):
    return {'_content':content,'_baseline':{'model':'mock','latency_s':.01},
            '_finish_reason':finish,'usage':{}}


class CategoryReviewTests(unittest.TestCase):
    def check(self, answer):
        with tempfile.TemporaryDirectory() as td:
            obj={'bbox_id':43,'target':'wrong-label','bbox_3d':[0,0,0,1,1,1]}
            xyz=np.random.default_rng(2).uniform(-.5,.5,(100,3))
            calls=[]
            with patch('category_review.chat',return_value=answer) as api:
                result=check_categories(['trash can'],[obj],xyz,np.full((100,3),100,dtype=np.uint8),
                                        np.full(100,43),Path(td),calls)
            prompt=api.call_args.args[0][0]['content'][1]['text']
            self.assertNotIn('wrong-label',prompt)
            self.assertNotIn('TargetIDs',prompt)
            self.assertEqual(len(calls),1)
            return result

    def test_exact_instance_verdict(self):
        result=self.check(reply('{"id":43,"verdict":"yes","reason":"container shape"}'))
        self.assertEqual(result['checks'][0]['verdict'],'yes')

    def test_wrong_id_or_truncated_output_becomes_uncertain(self):
        for data in [reply('{"id":1,"verdict":"yes","reason":"test"}'),
                     reply('{"id":43,"verdict":"yes","reason":"test"}','length'),
                     reply('{"TargetIDs":[43]}')]:
            result=self.check(data)
            self.assertEqual(result['checks'][0]['verdict'],'uncertain')
            self.assertIn('error',result['checks'][0])

    def test_missing_points_and_multiple_categories_make_no_request(self):
        with tempfile.TemporaryDirectory() as td, patch('category_review.chat') as api:
            args=([{'bbox_id':43,'target':'wrong-label'}],np.empty((0,3)),
                  np.empty((0,3),dtype=np.uint8),np.array([],dtype=int),Path(td),[])
            result=check_categories(['trash can'],*args)
            self.assertEqual(result['checks'][0]['verdict'],'uncertain')
            result=check_categories(['chair','table'],*args)
            self.assertTrue(result['status'].startswith('skipped'))
            api.assert_not_called()


if __name__=='__main__':unittest.main()
