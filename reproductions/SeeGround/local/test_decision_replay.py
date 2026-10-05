import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch
import requests
from decision_api import decide, validate_answers
from decision_replay import build_input, digest, freeze, rank, run_replay, summarize


def answers(questions, choice='supported'):
    return {key:{'type':'choice','choice':choice,'confidence':1.,
                 'probabilities':{option:float(option==choice) for option in q['criteria']}}
            for key,q in questions.items()}


def fixture():
    objects=[{'bbox_id':1,'target':'cabinet','bbox_3d':[0,0,0,1,1,1]},
             {'bbox_id':2,'target':'cabinet','bbox_3d':[2,0,0,1,1,1]}]
    run={'query':'the smallest cabinet','object_table':'gt','parsed_query':{'Target':['cabinet']},
         'calls':[{'stage':'ground','response':'ANSWER_MUST_NOT_LEAK'}],
         'evidence':{'geometry':{'pairs':[]},'diagnosis':{'status':'missing','missing':['size'],
             'hypotheses':[],'scope':'fixture','category_source':'annotated','checks':[
                 {'id':1,'category':'cabinet','attributes':['smallest'],'verdict':'uncertain',
                  'reason':'no image','crops':[{'image':'PRIVATE_PATH'}]}]}}}
    return run,objects


class DecisionTests(unittest.TestCase):
    def test_jev_request_and_response_are_typed_without_evaluation(self):
        run,objects=fixture();state,questions=build_input(run,objects)
        payload={'model':'jev-fixture','answers':answers(questions),'usage':{'input_tokens':4}}
        with patch.dict('os.environ',{'TYPESAFE_API_KEY':'secret-fixture'}), \
             patch('decision_api.requests.post',return_value=Mock(ok=True,json=lambda:payload)) as post:
            result=decide('jev',state,questions)
        kwargs=post.call_args.kwargs
        self.assertEqual(post.call_args.args[0],'https://api.typesafe.ai/v1/systemone')
        self.assertEqual(kwargs['json']['state'],state)
        self.assertEqual(kwargs['json']['questions'],questions)
        self.assertNotIn('evaluation',str(kwargs['json']))
        self.assertEqual(result['model'],'jev-fixture')
        self.assertNotIn('secret-fixture',json.dumps(result))

    def test_qwen_receives_identical_text_state_and_questions(self):
        run,objects=fixture();state,questions=build_input(run,objects)
        raw={'_content':json.dumps({'answers':answers(questions)}),
             '_baseline':{'model':'mock'},'_finish_reason':'stop'}
        with patch('decision_api.chat',return_value=raw) as chat:
            result=decide('qwen',state,questions)
        self.assertEqual(json.loads(chat.call_args.args[0][1]['content']),{'state':state,'questions':questions})
        self.assertEqual(result['model'],'mock')
        raw['_finish_reason']='length'
        with patch('decision_api.chat',return_value=raw):
            with self.assertRaisesRegex(ValueError,'truncated'):decide('qwen',state,questions)

    def test_invalid_answers_are_rejected(self):
        run,objects=fixture();_,questions=build_input(run,objects)
        for value in (None,{}, {'extra':{}}):
            with self.assertRaises(ValueError):validate_answers(value,questions)
        key=next(iter(questions))
        for bad in (float('nan'),1.1,-.1,True):
            value=answers(questions);value[key]['confidence']=bad
            with self.assertRaises(ValueError):validate_answers(value,questions)
        value=answers(questions);value[key]['probabilities']['unknown']=.4
        with self.assertRaises(ValueError):validate_answers(value,questions)
        value=answers(questions);value[key]['choice']='unknown'
        with self.assertRaises(ValueError):validate_answers(value,questions)

    def test_jev_errors_do_not_echo_key_or_response_body(self):
        run,objects=fixture();state,questions=build_input(run,objects)
        with patch.dict('os.environ',{'TYPESAFE_API_KEY':'secret-fixture'}):
            with patch('decision_api.requests.post',return_value=Mock(ok=False,status_code=401)):
                with self.assertRaisesRegex(ValueError,'HTTP 401'):decide('jev',state,questions)
            with patch('decision_api.requests.post',side_effect=requests.Timeout('secret-fixture')):
                with self.assertRaises(ValueError) as error:decide('jev',state,questions)
                self.assertNotIn('secret-fixture',str(error.exception))

    def test_state_has_no_old_prediction_or_image_paths(self):
        run,objects=fixture();state,questions=build_input(run,objects)
        self.assertNotIn('ANSWER_MUST_NOT_LEAK',json.dumps(state))
        self.assertNotIn('PRIVATE_PATH',json.dumps(state))
        self.assertEqual(len(questions),6)
        self.assertEqual(state['visual_checks'][0]['verdict'],'uncertain')

    def test_and_ranking_and_exact_tie_are_deterministic(self):
        run,objects=fixture();state,questions=build_input(run,list(reversed(objects)))
        value=answers(questions)
        self.assertEqual(rank(state,value)['predicted_id'],1)
        self.assertTrue(rank(state,value)['top_tied'])
        value['1_appearance']={'type':'choice','choice':'unknown','confidence':1.,
                              'probabilities':{'supported':0.,'contradicted':0.,'unknown':1.}}
        result=rank(state,value)
        self.assertEqual(result['predicted_id'],2)
        self.assertFalse(result['top_tied'])
        self.assertTrue(result['best_effort_only'])

    def test_errors_count_in_accuracy_and_do_not_look_stable(self):
        cases=[{'scene':'s','csv_row':2}]
        rows=[{'scene':'s','csv_row':2,'backend':'jev','predicted_id':1,'correct':True,'latency_s':.1},
              {'scene':'s','csv_row':2,'backend':'jev','predicted_id':1,'correct':True,'latency_s':.1},
              {'scene':'s','csv_row':2,'backend':'jev','error':'failed','correct':False}]
        stats=summarize(rows,cases,3)['jev']
        self.assertEqual(stats['accuracy'],2/3)
        self.assertEqual(stats['cases'][0]['pair_agreement'],1.)
        self.assertEqual(stats['all_repeats_identical_rate'],0.)
        self.assertEqual(stats['errors'],1)

    def test_freeze_checks_source_hash_and_separates_answer(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);scene=root/'local_outputs'/'s';scene.mkdir(parents=True)
            run,objects=fixture();obj=scene/'objects_gt.json';obj.write_text(json.dumps(objects))
            source=root/'batch';(source/'q001').mkdir(parents=True)
            (source/'q001/run.json').write_text(json.dumps(run))
            manifest={'scene':'s','settings':{'mode':'diagnostic','object_table':'gt'},'objects_sha256':digest(obj)}
            (source/'results.json').write_text(json.dumps({'manifest':manifest,'results':[
                {'csv_row':2,'query':run['query'],'target_id':2,'predicted_id':1}]}))
            with patch('decision_replay.ROOT',root),redirect_stdout(io.StringIO()):
                plan=freeze(source,root/'plan.json',[2])
                self.assertEqual(plan['cases'][0]['evaluation'],{'target_id':2})
                self.assertNotIn('target_id',plan['cases'][0]['state'])
                obj.write_text('changed')
                with self.assertRaisesRegex(ValueError,'哈希'):freeze(source,root/'other.json',[2])

    def test_replay_resumes_without_calls_and_rejects_changed_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run,objects=fixture();state,questions=build_input(run,objects)
            case={'scene':'s','csv_row':2,'state':state,'questions':questions,'evaluation':{'target_id':1}}
            plan=root/'plan.json';plan.write_text(json.dumps({'version':1,'cases':[case]}))
            reply={'model':'mock','answers':answers(questions),'latency_s':.1}
            env={'TYPESAFE_API_KEY':'SECRET','SEEGROUND_API_KEY':'SECRET',
                 'SEEGROUND_MODEL':'mock','SEEGROUND_BASE_URL':'https://example.invalid/v1'}
            with patch.dict('os.environ',env),patch('decision_replay.decide',return_value=reply) as api, \
                 redirect_stdout(io.StringIO()):
                run_replay(plan,root/'dry',2,dry_run=True);api.assert_not_called()
                self.assertFalse((root/'dry').exists())
                stats=run_replay(plan,root/'out',2)
                self.assertEqual(api.call_count,4)
                self.assertEqual(stats['jev']['accuracy'],1.)
                for call in api.call_args_list:
                    self.assertEqual(call.args[1],state)
                    self.assertNotIn('evaluation',str(call.args))
                self.assertNotIn('SECRET',(root/'out/manifest.json').read_text())
                api.reset_mock();run_replay(plan,root/'out',2);api.assert_not_called()
                case['state']['query']='changed';plan.write_text(json.dumps({'version':1,'cases':[case]}))
                with self.assertRaisesRegex(ValueError,'不同实验'):run_replay(plan,root/'out',2)
                api.assert_not_called()


if __name__=='__main__':unittest.main()
