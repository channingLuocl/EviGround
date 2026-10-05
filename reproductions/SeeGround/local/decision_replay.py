"""Freeze existing diagnostic evidence; compare repeated text-only Qwen/Jev decisions."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from decision_api import decide, jev_model, validate_answers
from run import ROOT, load_objects

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2))


def build_input(run, objects):
    evidence=run['evidence'];diagnosis=evidence['diagnosis']
    # Explicit fields exclude source predictions, evaluation labels and image/file paths.
    state={'query':run['query'],'object_table':run['object_table'],'objects':objects,
           'parsed_query':run['parsed_query'],
           'geometry':evidence['geometry'],
           'supplementary_geometry':evidence.get('supplementary_geometry'),
           'diagnosis':{k:diagnosis[k] for k in ('status','missing','hypotheses','scope','category_source')},
           'visual_checks':[{k:c[k] for k in ('id','category','attributes','verdict','reason') if k in c}
                            for c in diagnosis.get('checks',[])],
           'policy':'No image is supplied. Unknown is not false. Incomplete diagnostic candidate sets '
                    'must not exclude or prioritize objects. Inspection camera is not observer viewpoint. '
                    'Annotated categories are given; predicted labels and visual checks are soft evidence.'}
    if not 1<=len(objects)<=255:raise ValueError('Direct Choice requires 1-255 object options')
    questions={'target':{'type':'choice','instructions':
        'Which instance best matches the original query using ONLY supplied evidence? '
        'Return a best-effort target ID, even if conditions remain unresolved. '
        'Respect annotated target categories. Check omitted conditions in the original query. '
        'Combine spatial conditions under the SAME observer/reference hypothesis. '
        'Near is not nearest; camera position is not speaker viewpoint. '
        'A bounding box alone does not prove appearance. Unknown visual evidence is not negative. '
        'Do not use incomplete diagnosis candidate sets to exclude or prioritize objects. '
        'If several objects remain plausible, reflect uncertainty in the probability distribution.',
        'criteria':{str(obj['bbox_id']):{'instance_id':obj['bbox_id'],'category':obj['target']}
                    for obj in objects}}}
    return state,questions


def freeze(source, output, csv_rows=None, count=3):
    if output.exists():raise ValueError('冻结文件已存在，请换一个路径')
    if count<1 or (csv_rows is not None and len(set(csv_rows))!=len(csv_rows)):
        raise ValueError('题数须为正，CSV 行号不能重复')
    cases=[]
    paths=[source] if source.is_file() else sorted(source.rglob('results.json'))
    for path in paths:
        batch=json.loads(path.read_text());manifest=batch['manifest']
        if manifest['settings']['mode']!='diagnostic':continue
        scene=manifest['scene'];table=manifest['settings']['object_table']
        object_path=ROOT/'local_outputs'/scene/('objects_gt.json' if table=='gt' else 'objects.json')
        selected=[(i,r) for i,r in enumerate(batch['results'])
                  if r.get('predicted_id') is not None and (csv_rows is None or r['csv_row'] in csv_rows)]
        if not selected:continue
        if digest(object_path)!=manifest['objects_sha256']:
            raise ValueError('场景物体表与原实验哈希不一致：'+scene)
        objects=load_objects(object_path)
        for index,row in selected:
            run_path=path.parent/f'q{index+1:03d}'/'run.json'
            run=json.loads(run_path.read_text())
            if run['query']!=row['query'] or run['object_table']!=table:
                raise ValueError('原实验查询/物体表不一致')
            state,questions=build_input(run,objects)
            if row['target_id'] not in {o['bbox_id'] for o in objects}:raise ValueError('评测目标不在物体表内')
            cases.append({'scene':scene,'csv_row':row['csv_row'],'state':state,'questions':questions,
                          'evaluation':{'target_id':row['target_id']},
                          'source_sha256':{'results':digest(path),'run':digest(run_path),'objects':digest(object_path)}})
    if csv_rows is not None:
        if {c['csv_row'] for c in cases}!=set(csv_rows):raise ValueError('指定 CSV 行缺少已完成 diagnostic 记录')
    else:cases=cases[:count]
    if not cases or len({(c['scene'],c['csv_row']) for c in cases})!=len(cases):
        raise ValueError('没有可冻结题目或题目重复')
    plan={'version':2,'comparison':'text-only fixed evidence; not original image-grounding accuracy',
          'selection':'direct target-ID Choice; preserve provider choice, never resolve ties in code',
          'cases':cases}
    write_json(output,plan)
    print(f'冻结 {len(cases)} 题：{output}；未调用 API')
    return plan


def rank(state, answers):
    answer=answers['target']
    oid=int(answer['choice'])
    if oid not in {obj['bbox_id'] for obj in state['objects']}:raise ValueError('Choice outside full object table')
    ranked=[{'id':int(key),'probability':value} for key,value in answer['probabilities'].items()]
    ranked.sort(key=lambda r:(-r['probability'],r['id']))
    complete=set(answer['probabilities'])=={str(obj['bbox_id']) for obj in state['objects']}
    return {'predicted_id':oid,'ranking':ranked,'probability_options_complete':complete,
            'top_tied':(len(ranked)>1 and ranked[0]['probability']==ranked[1]['probability']) if complete else None,
            'best_effort_only':True}


def recover_record(record, case):
    """Recover a Qwen choice from a saved protocol-error response without a network call."""
    if record.get('backend')!='qwen' or not isinstance(record.get('error_response'),dict):
        return None
    response=record['error_response']
    if not isinstance(response.get('answers'),dict):return None
    data=response
    warnings=validate_answers(data.get('answers'),case['questions'],allow_partial_probabilities=True)
    if not isinstance(data.get('model'),str):raise ValueError('Saved response model missing')
    result={**record,**data,'validation_warnings':warnings,'latency_s':record.get('latency_s'),
            'recovered_offline':True,'original_error':record.get('error')}
    result.pop('error',None)
    result.update(rank(case['state'],data['answers']))
    result['correct']=result['predicted_id']==case['evaluation']['target_id']
    return result


def summarize(records, cases, repeats):
    summary={}
    for backend in ('qwen','jev'):
        rows=[r for r in records if r['backend']==backend]
        groups=[]
        for case in cases:
            selected=[r for r in rows if (r['scene'],r['csv_row'])==(case['scene'],case['csv_row'])]
            ids=[r['predicted_id'] for r in selected if 'predicted_id' in r]
            counts=Counter(ids);pairs=len(ids)*(len(ids)-1)//2
            groups.append({'scene':case['scene'],'csv_row':case['csv_row'],'completed':len(ids),
                'id_counts':dict(counts),'plurality_agreement':max(counts.values())/len(ids) if ids else None,
                'pair_agreement':sum(n*(n-1)//2 for n in counts.values())/pairs if pairs else None,
                'all_repeats_identical':len(ids)==repeats and len(counts)==1})
        completed=[r for r in rows if 'predicted_id' in r]
        timed=[r for r in completed if isinstance(r.get('latency_s'),(int,float))]
        total=len(cases)*repeats
        summary[backend]={'planned_calls':total,'recorded_calls':len(rows),'completed':len(completed),
            'errors':sum('error' in r for r in rows),'correct':sum(r.get('correct') is True for r in rows),
            'accuracy':sum(r.get('correct') is True for r in rows)/total,
            'top_tied_count':sum(r.get('top_tied') is True for r in rows),
            'warning_calls':sum(bool(r.get('validation_warnings')) for r in rows),
            'latency_evaluated':len(timed),
            'all_repeats_identical_rate':sum(g['all_repeats_identical'] for g in groups)/len(cases),
            'mean_latency_s':sum(r['latency_s'] for r in timed)/len(timed) if timed else None,
            'cases':groups}
    return summary


def run_replay(plan_path, output, repeats=3, dry_run=False):
    if repeats<2:raise ValueError('稳定性实验至少重复两次')
    plan=json.loads(plan_path.read_text())
    if plan.get('version')!=2 or not plan.get('cases'):
        raise ValueError('需要 version=2 的直接 Choice 清单；请重新 freeze 到新文件，不覆盖旧实验')
    cases=plan['cases']
    if dry_run:
        print(json.dumps({'network_request':False,'cases':len(cases),'repeats':repeats,
                          'calls':len(cases)*repeats*2,'questions_per_case':[len(c['questions']) for c in cases]}))
        return
    if not all(os.getenv(k,'').strip() for k in ('TYPESAFE_API_KEY','SEEGROUND_BASE_URL','SEEGROUND_MODEL','SEEGROUND_API_KEY')):
        raise ValueError('请先配置 Qwen 和 TYPESAFE_API_KEY；尚未调用 API')
    manifest={'plan_sha256':digest(plan_path),'repeats':repeats,
              'qwen_model':os.getenv('SEEGROUND_MODEL').strip(),
              'qwen_base_url':os.getenv('SEEGROUND_BASE_URL').strip().rstrip('/'),
              'jev_model':jev_model(),
              'code_sha256':{name:digest(Path(__file__).with_name(name)) for name in
                             ('decision_api.py','decision_replay.py','qwen_api.py')}}
    mp=output/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('输出目录已有不同实验，请更换目录')
    write_json(mp,manifest)
    records=[]
    for case_index,case in enumerate(cases):
        for repeat in range(repeats):
            # Alternate order so one backend is not always measured first.
            for backend in (('qwen','jev') if repeat%2==0 else ('jev','qwen')):
                path=output/f'q{case_index+1:03d}'/f'{backend}_{repeat+1:03d}.json'
                if path.exists():record=json.loads(path.read_text())
                else:
                    record={'scene':case['scene'],'csv_row':case['csv_row'],'backend':backend,'repeat':repeat+1}
                    try:
                        answer=decide(backend,case['state'],case['questions'])
                        record.update(answer,**rank(case['state'],answer['answers']))
                        record['correct']=record['predicted_id']==case['evaluation']['target_id']
                    except (ValueError,OSError,KeyError,TypeError) as exc:
                        record.update(error=str(exc),correct=False)
                        if hasattr(exc,'response_data'):record['error_response']=exc.response_data
                    write_json(path,record)
                records.append(record)
                stats=summarize(records,cases,repeats)
                write_json(output/'comparison.json',{'manifest':manifest,'summary':stats,'records':records})
                print(f'{case["scene"]}/{case["csv_row"]} {backend} {repeat+1}/{repeats}: '
                      f'{record.get("predicted_id",record.get("error"))}',flush=True)
    print(json.dumps(stats,ensure_ascii=False))
    return stats


def recover_replay(plan_path, source, output):
    if output.exists():raise ValueError('恢复输出目录已存在，请更换目录')
    plan=json.loads(plan_path.read_text());original=json.loads((source/'manifest.json').read_text())
    if plan.get('version')!=2 or original['plan_sha256']!=digest(plan_path):
        raise ValueError('清单版本或原实验输入哈希不一致；未调用 API')
    repeats=original['repeats'];records=[];files=[]
    for index,case in enumerate(plan['cases']):
        for repeat in range(repeats):
            for backend in ('qwen','jev'):
                path=source/f'q{index+1:03d}'/f'{backend}_{repeat+1:03d}.json'
                record=json.loads(path.read_text())
                if (record.get('scene'),record.get('csv_row'),record.get('backend'),record.get('repeat'))!=(
                        case['scene'],case['csv_row'],backend,repeat+1):
                    raise ValueError('原重复记录与清单不一致')
                try:recovered=recover_record(record,case)
                except (ValueError,KeyError,TypeError) as exc:
                    recovered=None;record={**record,'recovery_error':str(exc)}
                records.append(recovered if recovered is not None else record)
                files.append({'path':str(path.relative_to(source)),'sha256':digest(path)})
    manifest={'operation':'offline protocol recovery; no API calls','original_manifest':original,
              'original_manifest_sha256':digest(source/'manifest.json'),'source_records':files,
              'recovery_code_sha256':{name:digest(Path(__file__).with_name(name))
                                      for name in ('decision_api.py','decision_replay.py')}}
    stats=summarize(records,plan['cases'],repeats)
    write_json(output/'comparison.json',{'manifest':manifest,'summary':stats,'records':records})
    print(json.dumps({'network_request':False,'recovered':sum(r.get('recovered_offline') is True for r in records),
                      'summary':stats},ensure_ascii=False))
    return stats


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('freeze');p.add_argument('--source',type=Path,required=True)
    p.add_argument('--csv-rows',type=int,nargs='+');p.add_argument('--count',type=int,default=3)
    p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('run');p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true')
    p=sub.add_parser('recover',help='离线恢复已保存回答，不调用 API，不修改原实验')
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.command=='recover':recover_replay(args.plan,args.source,args.output);return
    if args.command=='freeze':freeze(args.source,args.output,args.csv_rows,args.count)
    else:run_replay(args.plan,args.output,args.repeats,args.dry_run)


if __name__=='__main__':main()
