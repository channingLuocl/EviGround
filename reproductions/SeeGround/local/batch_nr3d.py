"""Resumable Nr3D trial; optional annotated OLT, query answers are evaluation-only."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import html
import json
import os
from pathlib import Path
from real_scene import run_query, MODES, OBJECT_TABLES, object_table_path
from run import ROOT, load_objects
from metrics import bbox_iou_3d, threshold_accuracy

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
    ious=[r['iou_3d'] for r in results if isinstance(r.get('iou_3d'),(int,float))]
    latencies=[r['latency_s'] for r in results if isinstance(r.get('latency_s'),(int,float))]
    pipeline=[r['pipeline_latency_s'] for r in results if isinstance(r.get('pipeline_latency_s'),(int,float))]
    scoring_ious=[r.get('iou_3d',0.) for r in results]
    stats={'total':len(results),'completed':sum(r.get('predicted_id') is not None for r in results),
            'correct':sum(r.get('correct') is True for r in results),
            'accuracy':sum(r.get('correct') is True for r in results)/len(results) if results else None,
            'iou_evaluated':len(ious),'acc25':threshold_accuracy(scoring_ious,.25) if ious else None,
            'acc50':threshold_accuracy(scoring_ious,.5) if ious else None,
            'mean_iou':sum(ious)/len(results) if ious else None,
            'mean_latency_s':sum(latencies)/len(latencies) if latencies else None,
            'mean_pipeline_latency_s':sum(pipeline)/len(pipeline) if pipeline else None}
    answered = [r for r in results if r.get('predicted_id') is not None]
    stats.update(
        run_succeeded=sum(not r.get('error') for r in results),
        answer_coverage=len(answered)/len(results) if results else None,
        answered_accuracy=sum(r.get('correct') is True for r in answered)/len(answered) if answered else None,
        singleton_fallback_count=sum(r.get('prediction_source') == 'singleton_fallback' for r in results))
    if any('decision_status' in r for r in results):
        stats['decision_status_counts']={status:sum(r.get('decision_status')==status for r in results)
                                         for status in ('resolved','missing','ambiguous','conflict')}
        accepted=[r for r in results if r.get('accepted') is True]
        stats['accepted_count']=len(accepted)
        stats['accepted_coverage']=len(accepted)/len(results) if results else None
        stats['accepted_accuracy']=sum(r.get('correct') is True for r in accepted)/len(accepted) if accepted else None
    if any('final_review' in r for r in results):
        stats['backtrack_count'] = sum(r.get('backtrack_count', 0) for r in results)
        stats['review_status_counts'] = {s: sum((r.get('final_review') or {}).get('status') == s for r in results)
                                         for s in ('consistent', 'contradiction', 'unknown', 'not_applicable', 'disabled', 'review_failed')}
        changed = [r for r in results if r.get('review_predictions')]
        stats['review_wrong_to_correct'] = sum(r['review_predictions'][0]['correct'] is False and r.get('correct') is True for r in changed)
        stats['review_correct_to_wrong'] = sum(r['review_predictions'][0]['correct'] is True and r.get('correct') is False for r in changed)
        stats['review_correct_to_abstained'] = sum(r['review_predictions'][0]['correct'] is True and r.get('predicted_id') is None for r in changed)
        stats['review_correct_to_incorrect_answer'] = sum(r['review_predictions'][0]['correct'] is True and
            r.get('predicted_id') is not None and r.get('correct') is False for r in changed)
        stats['review_target_restored_but_abstained'] = sum(r.get('backtrack_target_restored') is True and
            r.get('predicted_id') is None for r in results)
        stats['review_confirms_exclusion'] = sum(f.get('finding') == 'confirms_exclusion'
            for r in results for f in r.get('review_findings', []))
        stats['historical_review_status_counts'] = {s: sum(v.get('status') == s for r in results
            for v in r.get('historical_reviews', [])) for s in ('consistent', 'contradiction', 'unknown', 'review_failed')}
        stats['review_errors'] = sum(r.get('review_failed') is True for r in results)
        stats['inference_errors'] = sum(bool(r.get('error')) for r in results)
        stats['actual_api_attempts'] = sum(r.get('api_attempts') or r.get('api_calls') or 0 for r in results)
        stats['internal_v4_calls'] = sum(sum((r.get('internal_budget_usage') or {}).values()) for r in results)
        stats['review_trigger_count'] = sum(r.get('review_triggered') is True for r in results)
        stats['v4_api_calls'] = sum(r.get('v4_api_calls', 0) for r in results)
        stats['v4_latency_s'] = sum(r.get('v4_latency_s', 0) for r in results)
    from v4_policy import v5_enabled
    if v5_enabled():
        stats['incorrect_answers'] = sum(r.get('predicted_id') is not None and not r.get('correct') for r in results)
        stats['abstained'] = sum(r.get('predicted_id') is None and not r.get('error') for r in results)
        stats['inference_errors'] = sum(bool(r.get('error')) for r in results)
        stats['review_errors'] = sum(r.get('review_failed') is True for r in results)
        stats['actual_api_attempts'] = sum(r.get('api_attempts') or r.get('api_calls') or 0 for r in results)
        stats['extra_api_attempts'] = sum(len(r.get('recovery_attempts', [])) for r in results)
        stats['api_stage_attempts'] = {stage: sum(len(c.get('api_attempts') or (c.get('api_diagnostic') or {}).get('api_attempts', []))
            or (0 if c.get('cache_hit') else 1) for r in results for c in r.get('api_stage_calls', []) if c.get('stage') == stage)
            for stage in sorted({c['stage'] for r in results for c in r.get('api_stage_calls', []) if c.get('stage')})}
    return stats

def experiment_settings(scene_dir, mode, review_limit=6, object_table='pred'):
    from qwen_api import api_provider, api_config_summary
    code_dir=Path(__file__).resolve().parent
    selected_path=object_table_path(scene_dir,object_table)
    settings={'mode':mode,'object_table':object_table,'objects_file':selected_path.name,
            'api_config':api_config_summary(),
            'review_limit':review_limit if mode in ('category_review','diagnostic','active_diagnostic') else None,
            'model':os.getenv('SEEGROUND_MODEL','').strip(),
            'base_url':os.getenv('SEEGROUND_BASE_URL','').strip().rstrip('/'),
            'code_sha256':{name:hashlib.sha256((code_dir/name).read_bytes()).hexdigest() for name in
                           ['run.py','real_scene.py','qwen_api.py','evidence.py','object_geometry.py','visual_review.py','category_review.py','diagnostic.py','wall_geometry.py','active_diagnostic.py','rgb_review.py','metrics.py','batch_nr3d.py']},
            'scene_sha256':{name:hashlib.sha256((scene_dir/name).read_bytes()).hexdigest() for name in
                            ['scene.npz','objects_gt.json']}}
    if mode in ('diagnostic','active_diagnostic'):
        from query_syntax import syntax_settings
        settings['query_syntax'] = syntax_settings()
        from v4_policy import settings as v4_settings
        settings['v4'] = v4_settings()
        for name in ('query_syntax.py', 'query_rewrite.py', 'size_semantics.py', 'v4_policy.py', 'v4_visual.py', 'v4_relations.py'):
            settings['code_sha256'][name] = hashlib.sha256((code_dir/name).read_bytes()).hexdigest()
        from rgb_review import input_hashes
        settings['rgb_sha256']=input_hashes(scene_dir,load_objects(selected_path))
        structural_path=scene_dir/'scene_instances.json'
        if structural_path.exists():
            settings['structural_instances_sha256']=hashlib.sha256(structural_path.read_bytes()).hexdigest()
    if mode=='active_diagnostic':
        from decision_api import JEV_URL, jev_model
        from react_decision import react_backend
        settings.update(decision_backend=react_backend(),planner_input='text_only',visual_backend=api_provider())
        if settings['decision_backend']=='jev':
            settings.update(jev_model=jev_model(),jev_url=JEV_URL)
        from v4_policy import v7_enabled
        if v7_enabled():
            settings['code_sha256']['v7_evidence.py'] = hashlib.sha256((code_dir/'v7_evidence.py').read_bytes()).hexdigest()
        for name in ('react_decision.py','decision_api.py'):
            settings['code_sha256'][name]=hashlib.sha256((code_dir/name).read_bytes()).hexdigest()
    return settings

def report(out,results,manifest):
    stats=summarize(results)
    (out/'results.json').write_text(json.dumps({'summary':stats,'manifest':manifest,'results':results},ensure_ascii=False,indent=2))
    esc=lambda x:html.escape(str(x))
    table=manifest['settings'].get('object_table','pred')
    cards=[]
    for i,r in enumerate(results):
        images=[Path(p).name for p in r.get('images',[])] or ['view.png']
        image_html=''.join(f'<img loading="lazy" src="q{i+1:03d}/{esc(name)}">' for name in images)
        diagnosis_html=(f'<p>诊断状态：{esc(r["decision_status"])}；证据支持：{esc(r.get("accepted"))}；'
                        f'稳定候选：{esc(r.get("stable_candidate_ids",[]))}</p>' if 'decision_status' in r else '')
        cards.append(f'<article><h2>{i+1}. {"正确" if r.get("correct") else "错误或未完成"}</h2><p>{esc(r["query"])}</p><p>正确编号：{r["target_id"]}；预测编号：{esc(r.get("predicted_id","未完成"))}</p>{diagnosis_html}<p>{esc(r.get("response",r.get("error","")))}</p>{image_html}</article>')
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>SeeGround · Nr3D 小样本试跑</title><style>body{max-width:960px;margin:40px auto;font:18px/1.7 system-ui;padding:20px;background:#f3f5f8}article{background:white;padding:24px;margin:24px 0;border-radius:12px}img{max-width:100%}</style><h1>Nr3D 真实描述试跑</h1><p>'+f'完成 {stats["completed"]}/{stats["total"]}，正确 {stats["correct"]}/{stats["total"]}。'+'</p><p>物体表来源：'+esc(table)+('（人工类别 / oracle OLT）' if table=='gt' else '（预测类别）')+'</p><p>抽样规则：'+esc(manifest['selection'])+'。这不是完整基准成绩。使用 Qwen API、CPU 渲染和简化类别匹配，与论文原始实现有差异。失败请求计入总数。</p>'+''.join(cards))
    return stats

def run_batch(csv_path, scene, rows, available, output, mode='baseline', review_limit=6,
              object_table='pred', selection='first N rows in CSV with matching scan_id and correct_guess=True'):
    if review_limit<1:raise ValueError('review-limit must be positive')
    scene_dir=ROOT/'local_outputs'/scene
    gt_objects={o['bbox_id']:o for o in load_objects(scene_dir/'objects_gt.json')}
    if any(r['target_id'] not in gt_objects for r in rows):raise ValueError('GT 物体表缺少所选目标，请检查数据对齐')
    manifest={'source':SOURCE,'csv_sha256':hashlib.sha256(csv_path.read_bytes()).hexdigest(),'scene':scene,
              'selection':selection,'available':available,'rows':rows,
              'objects_sha256':hashlib.sha256(object_table_path(scene_dir,object_table).read_bytes()).hexdigest(),
              'settings':experiment_settings(scene_dir,mode,review_limit,object_table)}
    output.mkdir(parents=True,exist_ok=True);mp=output/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('输出目录已有不同实验，请更换目录')
    mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    results=[]
    from batch_progress import BatchProgress
    progress = BatchProgress(len(rows))
    for i,row in enumerate(rows):
        out=output/f'q{i+1:03d}';pred=out/'prediction.json';result={**row,'object_table':object_table}
        try:
            if not pred.exists():
                out.mkdir(parents=True, exist_ok=True)
                with (out/'console.log').open('a') as log, redirect_stdout(log), redirect_stderr(log):
                    run_query(scene_dir,row['query'],out,'official Nr3D utterance; CSV row '+str(row['csv_row']),mode=mode,review_limit=review_limit,object_table=object_table)
            prediction=json.loads(pred.read_text())
            if prediction['query']!=row['query']:raise ValueError('缓存查询不一致')
            if prediction.get('object_table')!=object_table:raise ValueError('缓存物体表来源不一致')
            result.update(predicted_id=prediction['predicted_id'],response=prediction['response'],model=prediction['model'],
                          prediction_source=prediction.get('prediction_source'),
                          review_failed=prediction.get('review_failed',False),
                          api_attempts=prediction.get('api_attempts'),
                          recovery_attempts=prediction.get('recovery_attempts',[]),
                          final_review=prediction.get('final_review'), backtrack_count=prediction.get('backtrack_count', 0),
                          internal_budget_usage=prediction.get('internal_budget_usage'),
                          latency_s=prediction.get('latency_s'),pipeline_latency_s=prediction.get('pipeline_latency_s'),
                          api_calls=prediction.get('api_calls'),call_usage=prediction.get('call_usage'),
                          finish_reason=prediction.get('finish_reason'),images=prediction.get('images',[]))
            if 'decision_status' in prediction:
                result.update(decision_status=prediction['decision_status'],accepted=prediction['accepted'],
                              stable_candidate_ids=prediction['diagnosis']['stable_candidate_ids'],
                              diagnosis=prediction['diagnosis'],decision_reasons=prediction.get('decision_reasons',[]))
            gt_obj=gt_objects[row['target_id']]
            result['iou_3d']=(bbox_iou_3d(prediction['bbox_3d'],gt_obj['bbox_3d'])
                              if prediction['predicted_id'] is not None else 0.)
            result['correct']=result['predicted_id']==row['target_id']
        except (ValueError,OSError,KeyError,IndexError) as exc:
            result.update(error=str(exc),error_type=type(exc).__name__,correct=False)
            run_path=out/'run.json'
            if run_path.exists():
                try:
                    failure=json.loads(run_path.read_text())
                    if failure.get('query')==row['query']:
                        result.update(error_stage=failure.get('error_stage'),
                                      pipeline_latency_s=failure.get('pipeline_latency_s'),
                                      api_calls=len(failure.get('calls',[])))
                except (ValueError,OSError):
                    pass
        results.append(result);report(output,results,manifest)
        progress.update(result)
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,default=ROOT/'local_outputs/downloads/nr3d_official.csv');p.add_argument('--scene',default='scene0011_00');p.add_argument('--count',type=int,default=10);p.add_argument('--output',type=Path,default=ROOT/'local_outputs/nr3d_trial_10')
    p.add_argument('--mode',choices=MODES,default='baseline')
    p.add_argument('--object-table',choices=OBJECT_TABLES,default='pred')
    p.add_argument('--review-limit',type=int,default=6);a=p.parse_args()
    rows,available=select_rows(a.csv,a.scene,a.count)
    run_batch(a.csv,a.scene,rows,available,a.output,a.mode,a.review_limit,a.object_table)

if __name__=='__main__':main()
