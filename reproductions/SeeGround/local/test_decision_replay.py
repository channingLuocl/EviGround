import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch
import requests
from decision_api import DecisionAPIError, decide, validate_answers
from decision_replay import build_input, digest, freeze, rank, recover_record, recover_replay, run_replay, summarize


def answers(questions, choice='1'):
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
        value=answers(questions);value[key]['probabilities']={'1':0.,'2':0.}
        with self.assertRaises(ValueError):validate_answers(value,questions)
        value=answers(questions);value[key]['choice']='not_an_option'
        with self.assertRaises(ValueError):validate_answers(value,questions)

    def test_distribution_mismatch_is_recorded_without_changing_choice(self):
        run,objects=fixture();state,questions=build_input(run,objects)
        value=answers(questions,choice='2')
        value['target']['probabilities']={'1':.51,'2':.48}
        warnings=validate_answers(value,questions)
        self.assertEqual({w['issue'] for w in warnings},{'probability_sum','choice_not_probability_max'})
        self.assertEqual(rank(state,value)['predicted_id'],2)

    def test_rejected_response_retains_original_answers(self):
        run,objects=fixture();state,questions=build_input(run,objects)
        payload={'model':'jev-fixture','answers':{}}
        with patch.dict('os.environ',{'TYPESAFE_API_KEY':'secret-fixture'}), \
             patch('decision_api.requests.post',return_value=Mock(ok=True,json=lambda:payload)):
            with self.assertRaises(ValueError) as error:decide('jev',state,questions)
        self.assertEqual(error.exception.response_data,payload)

    def test_qwen_partial_probability_response_is_usable_with_warning(self):
        run,objects=fixture();state,questions=build_input(run,objects)
        response={'answers':{'target':{
            'type':'choice','choice':'2','probabilities':{'2':.125},'confidence':.125}},
            'model':'qwen','response':'original raw response','usage':{}}
        record={'backend':'qwen','scene':'s','csv_row':2,'repeat':1,'error_response':response}
        recovered=recover_record(record,{'state':state,'questions':questions,'evaluation':{'target_id':2},
                                         'scene':'s','csv_row':2})
        self.assertEqual(recovered['predicted_id'],2)
        self.assertEqual(recovered['validation_warnings'][0]['issue'],'incomplete_probability_options')
        self.assertTrue(recovered['correct'])
        self.assertIsNone(recovered['latency_s'])
        self.assertIsNone(recovered['top_tied'])
        self.assertEqual(recovered['answers']['target']['probabilities'],{'2':.125})
        with self.assertRaises(ValueError):validate_answers(response['answers'],questions)

    def test_server_sparse_probabilities_and_unknown_options(self):
        run,objects=fixture();_,questions=build_input(run,objects)
        value=answers(questions,choice='2');value['target']['probabilities']={'1':.1}
        warnings=validate_answers(value,questions,allow_partial_probabilities=True)
        self.assertIn('choice_probability_missing',{w['issue'] for w in warnings})
        value['target']['probabilities']={'invalid':.1}
        with self.assertRaises(ValueError):validate_answers(value,questions,allow_partial_probabilities=True)

    def test_live_qwen_retains_sparse_choice_but_jev_rejects_sparse_distribution(self):
        run,objects=fixture();state,questions=build_input(run,objects)
        value=answers(questions,choice='2');value['target']['probabilities']={'2':.125}
        payload={'answers':value,'model':'mock'}
        raw={'_content':json.dumps(payload),'_baseline':{'model':'mock'},'_finish_reason':'stop'}
        with patch('decision_api.chat',return_value=raw):
            result=decide('qwen',state,questions)
        self.assertEqual(result['answers']['target']['choice'],'2')
        self.assertEqual(len(result['validation_warnings']),2)
        with patch.dict('os.environ',{'TYPESAFE_API_KEY':'secret-fixture'}), \
             patch('decision_api.requests.post',return_value=Mock(ok=True,json=lambda:payload)):
            with self.assertRaises(ValueError):decide('jev',state,questions)

    def test_offline_recovery_never_calls_api_or_changes_source(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run,objects=fixture();state,questions=build_input(run,objects)
            case={'scene':'s','csv_row':2,'state':state,'questions':questions,'evaluation':{'target_id':2}}
            plan=root/'plan.json';plan.write_text(json.dumps({'version':2,'cases':[case]}))
            source=root/'source';(source/'q001').mkdir(parents=True)
            (source/'manifest.json').write_text(json.dumps({'plan_sha256':digest(plan),'repeats':2}))
            sparse=answers(questions,choice='2');sparse['target']['probabilities']={'2':.125}
            for repeat in (1,2):
                qwen={'scene':'s','csv_row':2,'backend':'qwen','repeat':repeat,'error':'protocol',
                      'error_response':{'answers':sparse,'model':'mock','response':'raw'}}
                jev={'scene':'s','csv_row':2,'backend':'jev','repeat':repeat,'predicted_id':2,
                     'correct':True,'latency_s':.1}
                (source/f'q001/qwen_{repeat:03d}.json').write_text(json.dumps(qwen))
                (source/f'q001/jev_{repeat:03d}.json').write_text(json.dumps(jev))
            hashes={str(p):digest(p) for p in source.rglob('*.json')}
            with patch('decision_replay.decide') as api,patch('decision_api.requests.post') as http, \
                 redirect_stdout(io.StringIO()):
                stats=recover_replay(plan,source,root/'recovered')
                api.assert_not_called();http.assert_not_called()
                with self.assertRaisesRegex(ValueError,'已存在'):recover_replay(plan,source,root/'recovered')
            self.assertEqual(stats['qwen']['accuracy'],1.)
            self.assertEqual(stats['qwen']['all_repeats_identical_rate'],1.)
            self.assertIsNone(stats['qwen']['mean_latency_s'])
            self.assertEqual(stats['qwen']['latency_evaluated'],0)
            self.assertEqual(hashes,{str(p):digest(p) for p in source.rglob('*.json')})

    def test_jev_errors_do_not_echo_key_or_response_body(self):
        run,objects=fixture();state,questions=build_input(run,objects)
        with patch.dict('os.environ',{'TYPESAFE_API_KEY':'secret-fixture'}):
            with patch('decision_api.requests.post',return_value=Mock(ok=False,status_code=401)):
                with self.assertRaisesRegex(ValueError,'HTTP 401'):decide('jev',state,questions)
            with patch('decision_api.requests.post',side_effect=requests.Timeout('secret-fixture')):
                with self.assertRaises(ValueError) as error:decide('jev',state,questions)
                self.assertNotIn('secret-fixture',str(error.exception))

    def test_jev_http_error_preserves_safe_service_diagnostic(self):
        run,objects=fixture();state,questions=build_input(run,objects)
        body={'error':'invalid question criteria'}
        with patch.dict('os.environ',{'TYPESAFE_API_KEY':'secret-fixture'}), \
             patch('decision_api.requests.post',return_value=Mock(ok=False,status_code=400,
                                                                   json=lambda:body)):
            with self.assertRaises(DecisionAPIError) as error:decide('jev',state,questions)
        self.assertIn('invalid question criteria',str(error.exception))
        self.assertEqual(error.exception.response_data,{'status_code':400,'body':body})
        self.assertNotIn('secret-fixture',str(error.exception))

    def test_jev_rejects_non_finite_state_before_network(self):
        state={'bad':float('nan')};questions={'q':{'type':'choice','instructions':'pick',
                                                   'criteria':{'a':'A','b':'B'}}}
        with patch.dict('os.environ',{'TYPESAFE_API_KEY':'secret-fixture'}), \
             patch('decision_api.requests.post') as post:
            with self.assertRaisesRegex(ValueError,'严格 JSON'):
                decide('jev',state,questions)
        post.assert_not_called()

    def test_state_has_no_old_prediction_or_image_paths(self):
        run,objects=fixture();state,questions=build_input(run,objects)
        self.assertNotIn('ANSWER_MUST_NOT_LEAK',json.dumps(state))
        self.assertNotIn('PRIVATE_PATH',json.dumps(state))
        self.assertEqual(len(questions),1)
        self.assertEqual(set(questions['target']['criteria']),{'1','2'})
        self.assertEqual(state['visual_checks'][0]['verdict'],'uncertain')

    def test_exact_tie_preserves_provider_choice_instead_of_smallest_id(self):
        run,objects=fixture();state,questions=build_input(run,list(reversed(objects)))
        value=answers(questions,choice='2')
        value['target']['probabilities']={'1':.5,'2':.5}
        self.assertEqual(rank(state,value)['predicted_id'],2)
        self.assertTrue(rank(state,value)['top_tied'])
        value['target']['probabilities']={'1':.2,'2':.8}
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
            plan=root/'plan.json';plan.write_text(json.dumps({'version':2,'cases':[case]}))
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
                api.side_effect=DecisionAPIError('missing answer',{'answers':{},'model':'mock'})
                failed=run_replay(plan,root/'failed',2)
                self.assertEqual(failed['jev']['errors'],2)
                self.assertEqual(failed['jev']['accuracy'],0.)
                error_record=json.loads((root/'failed/q001/jev_001.json').read_text())
                self.assertEqual(error_record['error_response'],{'answers':{},'model':'mock'})
                api.reset_mock()
                case['state']['query']='changed';plan.write_text(json.dumps({'version':2,'cases':[case]}))
                with self.assertRaisesRegex(ValueError,'不同实验'):run_replay(plan,root/'out',2)
                api.assert_not_called()

    def test_old_plan_fails_before_api(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);plan=root/'plan.json'
            plan.write_text(json.dumps({'version':1,'cases':[{}]}))
            with patch('decision_replay.decide') as api:
                with self.assertRaisesRegex(ValueError,'version=2'):run_replay(plan,root/'out',2)
                api.assert_not_called()


if __name__=='__main__':unittest.main()
