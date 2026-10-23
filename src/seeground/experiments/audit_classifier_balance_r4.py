"""Independently reconstruct r4 networks, sampling, ensembles, selection and metrics."""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch

import seeground.experiments.audit_pair_train_test as counter
from seeground.experiments.audit_classifier_sweep_r3 import independent_network
from seeground.experiments.audit_pair_views import z
from seeground.categories.category_common_v82 import file_hash
from seeground.experiments.run_rgb_experiment import check_frozen
from seeground.evaluation.run_storage import write_json

# Pure label normalization is cached only for speed; calculations remain independent.
counter.normalized_category=lru_cache(None)(counter.normalized_category)
SEEDS=(42,43,44)


def read(path):
    return json.loads(Path(path).read_text())


def numpy_candidates(bank,metadata):
    result={}
    for name,record in metadata.items():
        if record['kind']=='single':
            result[name]=[bank[f'{record["family"]}_seed{s}'] for s in SEEDS]
        elif record['kind']=='seed_mean':
            result[name]=[np.mean([z(bank[f'{record["family"]}_seed{s}']) for s in SEEDS],axis=0)]
    for name,record in metadata.items():
        if record['kind']=='pair':
            result[name]=[record['alpha']*result['ensemble_'+record['first']][0]+(1-record['alpha'])*result['ensemble_'+record['second']][0]]
    return result


def pick(rows,reference,mode):
    if mode=='P':
        eligible=[r for r in rows if r['mean_macro']>=reference['mean_macro']-1e-7]
        key=lambda r:(-r['mean_correct'],-r['mean_macro'],r['parameters'],r['name'],r['tau'],r['weight'])
    else:
        eligible=[r for r in rows if r['mean_correct']>=reference['mean_correct']-1e-7]
        key=lambda r:(-r['mean_macro'],-r['mean_correct'],r['parameters'],r['name'],r['tau'],r['weight'])
    return min(eligible,key=key)


def paired_bootstrap(left,right,manifest):
    _,index=np.unique([s['scene'].split('_')[0] for s in manifest['samples']],return_inverse=True)
    counts=np.bincount(index)
    draws=np.random.default_rng(20261019).integers(len(counts),size=(10000,len(counts)))
    change=left.mean(0)-right.mean(0)
    sums=np.bincount(index,weights=change)
    samples=100*sums[draws].sum(1)/counts[draws].sum(1)
    return {'delta_pp':float(100*change.mean()),'room_95CI_pp':np.quantile(samples,[.025,.975]).tolist(),
            'candidate_variants':len(left),'reference_variants':len(right),
            'corrected_pairs':[[int((a & ~b).sum()) for b in right] for a in left],
            'broken_pairs':[[int((~a & b).sum()) for b in right] for a in left]}


def check_sampling(out,config,labels):
    gpu_y=torch.from_numpy(labels).cuda()
    gpu_counts=torch.bincount(gpu_y,minlength=197).float()
    replay_records=[]
    for spec in config['specs']:
        for seed in SEEDS:
            folder=out/f'models/{spec["name"]}/seed{seed}'
            record=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
            assert record['spec']==spec and record['seed']==seed
            assert record['config_sha256']==file_hash(out/'config.json')
            assert record['epochs']==40 and record['train_objects']==28730 and record['draws']==40*28730
            assert record['parameter_change_l2']>0
            history=read(folder/'history.json')
            assert len(history)==40 and np.isfinite(history).all()
            saved=np.array(read(folder/'sampling.json'))
            assert saved.shape==(40,197) and (saved.sum(1)==28730).all()
            if 'sampler' in spec:
                generator=torch.Generator(device='cuda').manual_seed(seed)
                weights=(gpu_counts+1).pow(-spec['sampler'])[gpu_y]
                seen=torch.zeros(len(labels),device='cuda',dtype=torch.bool)
                differences=[]
                for epoch in range(40):
                    indices=torch.multinomial(weights,len(labels),replacement=True,generator=generator)
                    actual=torch.bincount(gpu_y[indices],minlength=197).cpu().numpy()
                    if not np.array_equal(actual,saved[epoch]):
                        delta=np.abs(actual-saved[epoch])
                        differences.append({'epoch':epoch+1,'absolute_count_difference':int(delta.sum()),
                                            'maximum_class_count_difference':int(delta.max())})
                    seen[indices]=True
                # Preserve exact-replay failures as a stated limitation, never hide them in a tolerance.
                class_probability=np.bincount(labels,minlength=197)*(np.bincount(labels,minlength=197)+1.)**(-spec['sampler'])
                class_probability/=class_probability.sum()
                expected=28730*class_probability
                deviation=np.abs(saved-expected)/np.sqrt(np.maximum(expected*(1-class_probability),1.))
                assert deviation.max()<8.,(spec['name'],seed,float(deviation.max()))
                replay_records.append({'name':spec['name'],'seed':seed,'exact_epochs':40-len(differences),
                    'differences':differences,'unique_objects_seen_recorded':record['unique_objects_seen'],
                    'unique_objects_seen_replayed':int(seen.sum()),'maximum_sampling_standardized_deviation':float(deviation.max())})
            else:
                expected=np.broadcast_to(np.bincount(labels,minlength=197),saved.shape)
                np.testing.assert_array_equal(saved,expected)
                assert record['unique_objects_seen']==28730
    exact=all(not r['differences'] and r['unique_objects_seen_recorded']==r['unique_objects_seen_replayed'] for r in replay_records)
    return {'status':'exact' if exact else 'not_fully_reproduced','records':replay_records,
            'limitation':None if exact else 'Some categorical sampling counts did not exactly replay; recorded trajectories retained. Underlying numerical cause not established. No models or predictions changed.'}


def audit(args):
    out=args.output
    for marker in ('SELECTION_FROZEN.json','TEST_FROZEN.json'):
        check_frozen(out,marker)
    config=read(out/'config.json')
    for name,digest in config['inputs'].items():
        assert file_hash(name)==digest,name
    for name,digest in config['sources'].items():
        assert file_hash(out/'source_snapshot'/name)==digest,name
    locations={k:Path(v) for k,v in config['input_location'].items()}
    ref,r3=locations['reference'],locations['r3']
    for marker in ('SELECTION_FROZEN.json','TEST_FROZEN.json'):
        check_frozen(r3,marker)
    manifests={s:read(ref/f'manifests/{s}.json') for s in ('train','eval','dev')}
    rooms={s:{r['scene'].split('_')[0] for r in m['samples']} for s,m in manifests.items()}
    assert not rooms['train']&rooms['eval'] and not rooms['train']&rooms['dev'] and not rooms['eval']&rooms['dev']
    assert [len(manifests[s]['samples']) for s in ('train','eval','dev')]==[28730,404,3413]
    labels=np.array([s['label'] for s in manifests['train']['samples']])
    assert (labels>=0).all()
    sampling=check_sampling(out,config,labels)
    print(json.dumps({'stage':'sampling_checked','models':36,'replay':sampling}),flush=True)
    counts=np.bincount(labels,minlength=197)+1
    logp=np.log(counts/counts.sum())
    text=np.load(ref/'features/text.npy').astype(float)
    selection=read(out/'selection.json')
    metadata=read(out/'candidates.json')
    families=read(out/'families.json')
    assert len(families)==16 and len(metadata)==38
    results=read(out/'results.json')['records']
    audited,comparisons=[],{}
    max_difference=0.
    for role,split in (('validation','eval'),('historical_test','dev')):
        manifest=manifests[split]
        shape=np.load(ref/f'features/shape_{split}.npz')['features'].astype(float)
        rgb=np.load(ref/('features/rgb_dev.npz' if split=='dev' else 'features/rgb_eval/features.npz'))['features'].astype(float)
        views_path=locations['r2']/('validation_views' if split=='eval' else 'historical_test_views')
        views=np.load(views_path/'features.npz')['features'].astype(float)
        data=(shape,rgb,views)
        merged=views.mean(1)
        merged/=np.linalg.norm(merged,axis=-1,keepdims=True)
        np.testing.assert_allclose(merged,rgb,atol=1e-5,rtol=0)
        base=z(10*shape@text.T)
        with np.load(out/role/'bank.npz') as archive:
            bank={k:archive[k].astype(float) for k in archive.files}
        for family in families:
            for seed in SEEDS:
                root=r3/f'models/{family[4:]}' if family.startswith('old_') else out/f'models/{family}'
                record=torch.load(root/f'seed{seed}/model.pt',map_location='cpu',weights_only=True)
                expected=independent_network(record,data)
                actual=bank[f'{family}_seed{seed}']
                difference=float(np.abs(actual-expected).max())
                max_difference=max(max_difference,difference)
                np.testing.assert_allclose(actual,expected,atol=2e-3,rtol=3e-4)
        if role=='validation':
            with np.load(out/'validation_bank.npz') as earlier:
                for k,v in bank.items():
                    np.testing.assert_array_equal(v,earlier[k])
        candidates=numpy_candidates(bank,metadata)
        candidates['S0']=[base]
        for method in ('P','M'):
            candidates['r3_'+method]=[np.load(r3/role/f'locked_{method}_seed{s}.npy').astype(float) for s in SEEDS]
        if role=='validation':
            rebuilt=[]
            for row in read(out/'validation_grid.json'):
                raw=candidates[row['name']]
                values=raw if row['name'] in ('S0','r3_P','r3_M') else [
                    (1-row['weight'])*base+row['weight']*z(a-row['tau']*logp) for a in raw]
                measures=[counter.recount(v,manifest)[0] for v in values]
                correct=[m['correct'] for m in measures]
                new={**row,'correct_by_variant':correct,'mean_correct':float(np.mean(correct)),
                     'mean_macro':float(np.mean([m['class_mean_accuracy'] for m in measures]))}
                assert correct==row['correct_by_variant'],row
                assert abs(new['mean_macro']-row['mean_macro'])<2e-7,row
                assert new['mean_correct']==row['mean_correct']
                rebuilt.append(new)
            assert len(rebuilt)==915
            reference=next(r for r in rebuilt if r['name']=='r3_M')
            baseline=next(r for r in rebuilt if r['name']=='S0')
            for mode in ('P','M'):
                expected=pick(rebuilt,reference,mode)
                assert all(expected[k]==selection['locked'][mode][k] for k in ('name','tau','weight','mean_correct'))
            for name in metadata:
                expected=pick([r for r in rebuilt if r['name']==name],baseline,'P')
                assert all(expected[k]==selection['per_candidate'][name][k] for k in ('name','tau','weight','mean_correct'))
            print(json.dumps({'stage':'grid_verified','rows':len(rebuilt)}),flush=True)
        hits={k:[] for k in ('S0','r3_P','r3_M','locked_P','locked_M')}
        for row in (r for r in results if r['role']==role):
            name,variant,sel=row['method'],row['variant'],row['selection']
            raw=candidates[sel['name']][variant]
            expected=raw if sel['name'] in ('S0','r3_P','r3_M') else (1-sel['weight'])*base+sel['weight']*z(raw-sel['tau']*logp)
            actual=np.load(out/role/f'{name}_v{variant}.npy')
            np.testing.assert_allclose(actual,expected,atol=1e-3,rtol=2e-4)
            metrics,hit,_,_=counter.recount(actual,manifest)
            for key,value in metrics.items():
                assert abs(value-row['summary'][key])<2e-7,(role,name,key)
            audited.append({'role':role,'method':name,'variant':variant,'summary':metrics})
            if name in hits:
                hits[name].append(hit)
        hits={k:np.array(a) for k,a in hits.items()}
        if role=='historical_test':
            assert int(hits['S0'].sum())==2262
        for name in ('locked_P','locked_M'):
            for reference in ('S0','r3_M'):
                comparisons[f'{role}:{name}-{reference}']=paired_bootstrap(hits[name],hits[reference],manifest)
        print(json.dumps({'stage':'audited','role':role}),flush=True)
    status='passed' if sampling['status']=='exact' else 'passed_with_sampling_replay_limit'
    summary={'status':status,'sampling_replay':sampling,'locked':selection['locked'],'records':audited,'comparisons':comparisons,
             'maximum_independent_raw_difference':max_difference,'auditor_sha256':file_hash(Path(__file__)),
             'limitation':config['limitation']}
    write_json(out/'audited_summary.json',summary)
    report=['# 均衡采样与集成 r4结果','','28730训练／404验证／3413历史评估，主干冻结；并非独立盲测。',
            '12新训练配置×3种子，4旧家族复用；16单模型家族、16种子集成、6配对集成及3固定参照。','',
            'P验证Macro不得低于r3M，再最大化平均True；M验证平均True不得低于r3M，再最大化Macro。',
            '集成只有一个预测；单模型家族三个种子平均，两者均按平均正确数比较。',
            '锁定选择：'+json.dumps(selection['locked'],ensure_ascii=False),'',
            '| 方法 | 正确数（各变体） | True准确率 | Exact Top-1 | Top-5 | Class Mean |','|---|---|---:|---:|---:|---:|']
    table=['模型\t评估对象数\tTrue正确数\tTrue准确率\tExact Top-1\tTop-5\tClass Mean\tErrors']
    for name in ['S0','r3_P','r3_M','locked_P','locked_M']+list(metadata):
        rows=[r for r in audited if r['role']=='historical_test' and r['method']==name]
        metrics=[r['summary'] for r in rows]
        report.append('| '+name+' | '+'/'.join(str(m['correct']) for m in metrics)+' | '+
            ' | '.join(f'{100*np.mean([m[k] for m in metrics]):.2f}%' for k in ('accuracy','exact_accuracy','top5_accuracy','class_mean_accuracy'))+' |')
        for row in rows:
            m=row['summary']
            label=name+(f'（variant {row["variant"]}）' if len(rows)>1 else '（单次集成预测）' if name!='S0' else '')
            table.append('\t'.join([label,str(m['count']),str(m['correct'])]+
                [f'{100*m[k]:.2f}%' for k in ('accuracy','exact_accuracy','top5_accuracy','class_mean_accuracy')]+['0']))
    report+=['','True按原同义词/复数归一化，Exact为字符串精确匹配；OOV保留计错。',
             '负结果全部保留，未根据3413重新选型。全部915行网格、所有网络/集成/预测和指标、房间bootstrap通过独立核查。',
             '采样精确重放状态：'+sampling['status']+'。原始逐轮抽样计数保留；底层数值原因未确证，不能声称完全逐对象复现训练。原始失败记录未删除，训练模型和分数未修改。',
             '三种子单模型行取三次均值；ensemble/pair行只有一次确定性分数融合结果，没有三种子标准差。']
    (out/'report.md').write_text('\n'.join(report)+'\n')
    (out/'table.tsv').write_text('\n'.join(table)+'\n')
    print(json.dumps({'status':status,'locked':selection['locked'],'maximum_raw_difference':max_difference}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    torch.set_num_threads(2)
    audit(parser.parse_args())
