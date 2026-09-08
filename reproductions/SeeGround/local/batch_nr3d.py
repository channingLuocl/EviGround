"""Small, resumable Nr3D trial. Ground truth is used only for evaluation."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
from real_scene import run_query
from run import ROOT, load_objects

SOURCE='https://drive.google.com/file/d/1qswKclq4BlnHSGMSgzLmUu8iqdUXD8ZC/view'

def select_rows(path, scene, count):
    rows=[]
    with path.open(newline='') as f:
        for index,row in enumerate(csv.DictReader(f)):
            if row['scan_id']==scene and row['correct_guess'].lower()=='true':
                rows.append({'csv_row':index+2,'query':row['utterance'],'target_id':int(row['target_id']),
                             'stimulus_id':row['stimulus_id']})
    if count<1 or len(rows)<count:raise ValueError('请求数量不合法或有效描述不足')
    return rows[:count],len(rows)

def summarize(results):
    return {'total':len(results),'completed':sum(r.get('predicted_id') is not None for r in results),
            'correct':sum(r.get('correct') is True for r in results),
            'accuracy':sum(r.get('correct') is True for r in results)/len(results) if results else None}

def report(out,results,manifest):
    stats=summarize(results)
    (out/'results.json').write_text(json.dumps({'summary':stats,'manifest':manifest,'results':results},ensure_ascii=False,indent=2))
    esc=lambda x:html.escape(str(x))
    cards=[]
    for i,r in enumerate(results):
        cards.append(f'<article><h2>{i+1}. {"正确" if r.get("correct") else "错误或未完成"}</h2><p>{esc(r["query"])}</p><p>正确编号：{r["target_id"]}；预测编号：{esc(r.get("predicted_id","未完成"))}</p><p>{esc(r.get("response",r.get("error","")))}</p><img loading="lazy" src="q{i+1:03d}/view.png"></article>')
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>SeeGround · Nr3D 小样本试跑</title><style>body{max-width:960px;margin:40px auto;font:18px/1.7 system-ui;padding:20px;background:#f3f5f8}article{background:white;padding:24px;margin:24px 0;border-radius:12px}img{max-width:100%}</style><h1>Nr3D 真实描述试跑</h1><p>'+f'完成 {stats["completed"]}/{stats["total"]}，正确 {stats["correct"]}/{stats["total"]}。'+'</p><p>单场景、按 CSV 顺序选取的有效描述；这不是完整基准成绩。使用 Qwen API、CPU 渲染和简化类别匹配，与论文原始实现有差异。失败请求计入总数。</p>'+''.join(cards))
    return stats

def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,default=ROOT/'local_outputs/downloads/nr3d_official.csv');p.add_argument('--scene',default='scene0011_00');p.add_argument('--count',type=int,default=10);p.add_argument('--output',type=Path,default=ROOT/'local_outputs/nr3d_trial_10');a=p.parse_args()
    rows,available=select_rows(a.csv,a.scene,a.count)
    scene_dir=ROOT/'local_outputs'/a.scene
    candidate_ids={o['bbox_id'] for o in load_objects(scene_dir/'objects.json')}
    if any(r['target_id'] not in candidate_ids for r in rows):raise ValueError('选定描述包含候选清单缺失的目标；请检查数据对齐')
    manifest={'source':SOURCE,'csv_sha256':hashlib.sha256(a.csv.read_bytes()).hexdigest(),'scene':a.scene,
              'selection':'first N rows in CSV with matching scan_id and correct_guess=True','available':available,'rows':rows,
              'objects_sha256':hashlib.sha256((scene_dir/'objects.json').read_bytes()).hexdigest()}
    a.output.mkdir(parents=True,exist_ok=True);mp=a.output/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('输出目录已有不同实验，请更换目录')
    mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    results=[]
    for i,row in enumerate(rows):
        out=a.output/f'q{i+1:03d}';pred=out/'prediction.json';result=dict(row)
        try:
            if not pred.exists():run_query(scene_dir,row['query'],out,'official Nr3D utterance; CSV row '+str(row['csv_row']))
            prediction=json.loads(pred.read_text())
            if prediction['query']!=row['query']:raise ValueError('缓存查询不一致')
            result.update(predicted_id=prediction['predicted_id'],response=prediction['response'],model=prediction['model'])
            result['correct']=result['predicted_id']==row['target_id']
        except (ValueError,OSError,KeyError,IndexError) as exc:
            result.update(error=str(exc),correct=False)
        results.append(result);stats=report(a.output,results,manifest)
        print(f'{i+1}/{len(rows)}: predicted={result.get("predicted_id")} expected={row["target_id"]} correct={result["correct"]}',flush=True)
    print(json.dumps(stats),flush=True)

if __name__=='__main__':main()
