"""Fixed held-out-scene sampling and resumable baseline/diagnostic comparison."""
import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path
from zipfile import ZipFile
from eviground.evaluation.batch_nr3d import run_batch, summarize, experiment_settings
from eviground.grounding.real_scene import object_table_path, OBJECT_TABLES
from eviground.rendering.run import ROOT, load_objects

MODES=('baseline','diagnostic')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def csv_pools(path):
    pools={}
    with path.open(newline='') as f:
        for index,row in enumerate(csv.DictReader(f)):
            if row['correct_guess'].lower()!='true' or row.get('dataset','nr3d')!='nr3d':continue
            scene=row['scan_id']
            if not re.fullmatch(r'scene\d{4}_\d{2}',scene):raise ValueError('非法场景编号')
            pools.setdefault(scene,[]).append({'csv_row':index+2,'query':row['utterance'],
                'target_id':int(row['target_id']),'stimulus_id':row['stimulus_id']})
    return pools

def select_plan(csv_path, olt_path, scene_count, per_scene, seed, exclude, scenes=None):
    if scene_count<1 or per_scene<1:raise ValueError('场景数和每场景题数必须为正')
    pools=csv_pools(csv_path)
    with ZipFile(olt_path) as archive:
        supported={Path(name).stem for name in archive.namelist()
                   if name.startswith('seeground_object_lookup_table/nr3d/gt/') and name.endswith('.json')}
    eligible=sorted(s for s,rows in pools.items()
                    if s in supported and s not in exclude and len(rows)>=per_scene)
    rng=random.Random(seed)
    if scenes is None:
        if len(eligible)<scene_count:raise ValueError('符合条件的场景不足')
        scenes=sorted(rng.sample(eligible,scene_count))
    elif len(set(scenes))!=len(scenes) or any(s not in eligible for s in scenes):
        raise ValueError('指定场景重复、被排除、缺少 OLT 或有效描述不足')
    selected=[{'scene':s,'available':len(pools[s]),
               'rows':sorted(rng.sample(pools[s],per_scene),key=lambda r:r['csv_row'])} for s in scenes]
    return {'version':1,'csv_sha256':digest(csv_path),'seed':seed,'excluded_scenes':sorted(exclude),
            'selection':'seeded random scenes and correct_guess=True Nr3D rows; not official test split',
            'scenes':selected,'total':len(scenes)*per_scene}

def save_fixed(path, data):
    if path.exists() and json.loads(path.read_text())!=data:raise ValueError('已有不同实验，请更换输出路径')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2))

def check_plan(plan, csv_path):
    if plan.get('version')!=1 or plan.get('csv_sha256')!=digest(csv_path):
        raise ValueError('抽样清单版本或 CSV 哈希不一致')
    pools=csv_pools(csv_path);seen=set();seen_scenes=set()
    for entry in plan['scenes']:
        scene=entry['scene']
        if scene in seen_scenes or scene in plan['excluded_scenes']:raise ValueError('场景重复或属于排除场景')
        seen_scenes.add(scene)
        source={r['csv_row']:r for r in pools.get(scene,[])}
        if entry['available']!=len(source) or not entry['rows']:raise ValueError('场景有效题数不一致')
        for row in entry['rows']:
            if row!=source.get(row['csv_row']) or row['csv_row'] in seen:raise ValueError('抽样题目与 CSV 不一致或重复')
            seen.add(row['csv_row'])
    if not seen or len(seen)!=plan['total']:raise ValueError('抽样总数不一致')

def preflight(plan, csv_path, object_table, review_limit):
    check_plan(plan,csv_path)
    settings={};missing=[]
    for entry in plan['scenes']:
        scene_dir=ROOT/'local_outputs'/entry['scene']
        required={scene_dir/'scene.npz',scene_dir/'objects_gt.json',object_table_path(scene_dir,object_table)}
        missing.extend(str(p) for p in sorted(required) if not p.is_file())
    if missing:raise ValueError('缺少已准备场景文件；尚未调用 API：\n'+'\n'.join(missing))
    for entry in plan['scenes']:
        scene_dir=ROOT/'local_outputs'/entry['scene']
        gt_ids={o['bbox_id'] for o in load_objects(scene_dir/'objects_gt.json')}
        if any(row['target_id'] not in gt_ids for row in entry['rows']):raise ValueError('GT 表缺少所选目标')
        load_objects(object_table_path(scene_dir,object_table))
        settings[entry['scene']]={mode:{**experiment_settings(scene_dir,mode,review_limit,object_table),
            'objects_sha256':digest(object_table_path(scene_dir,object_table))} for mode in MODES}
    return settings

def compare_pairs(baseline, diagnostic):
    key=lambda r:(r['scene'],r['csv_row'])
    first={key(r):r for r in baseline};second={key(r):r for r in diagnostic}
    if len(first)!=len(baseline) or len(second)!=len(diagnostic) or first.keys()!=second.keys():
        raise ValueError('两组题目不一致或重复')
    pairs=[]
    for k,a in first.items():
        b=second[k]
        if any(a[field]!=b[field] for field in ('query','target_id','stimulus_id')):
            raise ValueError('同一题的查询或评测答案不一致')
        ac=a.get('correct') is True;bc=b.get('correct') is True
        change=('both_correct' if bc else 'regressed') if ac else ('improved' if bc else 'both_wrong')
        pairs.append({'scene':k[0],'csv_row':k[1],'query':a['query'],'target_id':a['target_id'],
            'baseline_id':a.get('predicted_id'),'diagnostic_id':b.get('predicted_id'),'change':change,
            'decision_status':b.get('decision_status'),'baseline_error':a.get('error'),
            'diagnostic_error':b.get('error')})
    stats={'baseline':summarize(baseline),'diagnostic':summarize(diagnostic),
           'changes':{c:sum(r['change']==c for r in pairs)
                      for c in ('improved','regressed','both_correct','both_wrong')}}
    stats['accuracy_delta']=stats['diagnostic']['accuracy']-stats['baseline']['accuracy'] if pairs else None
    stats['completed_pairs']=sum(a.get('predicted_id') is not None and second[k].get('predicted_id') is not None
                                 for k,a in first.items())
    return stats,pairs

def run_paired(plan, csv_path, output, object_table='gt', review_limit=6):
    if review_limit<1:raise ValueError('review-limit must be positive')
    settings=preflight(plan,csv_path,object_table,review_limit)
    manifest={'plan':plan,'object_table':object_table,'settings':settings,'runner_sha256':digest(Path(__file__))}
    save_fixed(output/'manifest.json',manifest)
    accumulated={m:[] for m in MODES};per_scene={}
    for entry in plan['scenes']:
        scene=entry['scene'];current={}
        for mode in MODES:
            print(f'场景 {scene} / {mode}',flush=True)
            results=run_batch(csv_path,scene,entry['rows'],entry['available'],output/scene/mode,
                              mode,review_limit,object_table,plan['selection'])
            current[mode]=[{**r,'scene':scene} for r in results]
            accumulated[mode].extend(current[mode])
        per_scene[scene]=compare_pairs(current['baseline'],current['diagnostic'])[0]
        summary,pairs=compare_pairs(accumulated['baseline'],accumulated['diagnostic'])
        summary.update(planned_total=plan['total'],scenes_evaluated=len(per_scene))
        (output/'comparison.json').write_text(json.dumps({'summary':summary,'per_scene':per_scene,
            'pairs':pairs,'manifest':manifest},ensure_ascii=False,indent=2))
        with (output/'comparison.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    plan_cmd=sub.add_parser('plan',help='仅生成固定题目清单，不下载、不调用 API')
    plan_cmd.add_argument('--olt',type=Path,default=ROOT/'local_outputs/downloads/olt.zip')
    plan_cmd.add_argument('--scene-count',type=int,default=5);plan_cmd.add_argument('--per-scene',type=int,default=20)
    plan_cmd.add_argument('--seed',type=int,default=2026);plan_cmd.add_argument('--scenes',nargs='+')
    plan_cmd.add_argument('--exclude-scenes',nargs='*',default=['scene0011_00'])
    run_cmd=sub.add_parser('run',help='核对全部场景后，运行 baseline 与 diagnostic，调用 API')
    run_cmd.add_argument('--object-table',choices=OBJECT_TABLES,default='gt')
    run_cmd.add_argument('--review-limit',type=int,default=6)
    run_cmd.add_argument('--output',type=Path,default=ROOT/'local_outputs/nr3d_paired_gt_01')
    for cmd in (plan_cmd,run_cmd):
        cmd.add_argument('--csv',type=Path,default=ROOT/'local_outputs/downloads/nr3d_official.csv')
        cmd.add_argument('--plan',type=Path,default=ROOT/'local_outputs/nr3d_heldout_plan_01.json')
    args=parser.parse_args()
    if args.command=='run':
        run_paired(json.loads(args.plan.read_text()),args.csv,args.output,args.object_table,args.review_limit)
    else:
        plan=select_plan(args.csv,args.olt,args.scene_count,args.per_scene,args.seed,args.exclude_scenes,args.scenes)
        save_fixed(args.plan,plan)
        print(f'固定清单：{args.plan}；{len(plan["scenes"])} 个场景，共 {plan["total"]} 题；未下载、未调用 API')
        for entry in plan['scenes']:
            scene=entry['scene'];scene_dir=ROOT/'local_outputs'/scene
            missing=[name for name in ('scene.npz','objects_gt.json') if not (scene_dir/name).is_file()]
            print(f'{scene}: 抽样 {len(entry["rows"])}/{entry["available"]}；缺少 {missing or "无"}')
            if missing:print(f'  原始文件备齐后：.venv/bin/python local/prepare_scene.py --scene {scene}')

if __name__=='__main__':main()
