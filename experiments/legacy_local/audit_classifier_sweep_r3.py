"""Independent NumPy reconstruction and validation-only selection audit for r3."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from audit_pair_views import z
from audit_pair_train_test import bootstrap, recount
from category_common_v82 import file_hash
from run_rgb_experiment import check_frozen
from run_storage import write_json

SEEDS=(42,43,44)


def read(path):
    return json.loads(Path(path).read_text())


def linear(x,s,key):
    return x@s[key+'.weight'].T+s[key+'.bias']


def layernorm(x,s,key):
    return (x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-5)*s[key+'.weight']+s[key+'.bias']


def gelu(x):
    return F.gelu(torch.from_numpy(x)).numpy()


def normalize(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def independent_network(record,data):
    shape,rgb,views=data
    s={k:v.numpy().astype(float) for k,v in record['state_dict'].items()}
    architecture=record['architecture']
    if architecture=='view_attention':
        v=linear(views,s,'project')
        qkv=v@s['attention.in_proj_weight'].T+s['attention.in_proj_bias']
        q,k,value=[a.reshape(len(v),4,4,32).transpose(0,2,1,3) for a in np.split(qkv,3,axis=-1)]
        attention=q@k.transpose(0,1,3,2)/np.sqrt(32)
        attention=np.exp(attention-attention.max(-1,keepdims=True))
        attention/=attention.sum(-1,keepdims=True)
        h=(attention@value).transpose(0,2,1,3).reshape(len(v),4,128)
        h=linear(h,s,'attention.out_proj')
        x=np.concatenate([rgb,layernorm(v+h,s,'norm').mean(1)],-1)
        return linear(layernorm(x,s,'head.0'),s,'head.2')
    x=rgb if architecture.startswith('rgb') else shape if architecture.startswith('shape') else np.concatenate([shape,rgb],-1)
    if architecture.endswith('adapter'):
        adapted=linear(gelu(linear(x,s,'adapter.0')),s,'adapter.3')
        return linear(normalize(x+adapted),s,'head')
    return linear(gelu(linear(layernorm(x,s,'head.0'),s,'head.1')),s,'head.4')


def independent_analytic(record,data):
    shape,rgb,_=data
    x={'rgb':rgb,'shape':shape,'joint':normalize(np.concatenate([shape,rgb],-1))}[record['modality']]
    w=record['weight'].numpy().astype(float)
    if record['kind']=='prototype':
        return 10*x@w.T
    x=(x-record['mean'].numpy())/record['std'].numpy()
    return np.concatenate([x,np.ones((len(x),1))],1)@w


def select(rows,baseline,mode):
    if mode=='P':
        eligible=[r for r in rows if r['mean_macro']>=baseline['class_mean_accuracy']-1e-7]
        key=lambda r:(-r['total_correct'],-r['mean_macro'],r['parameters'],r['name'],r['tau'],r['weight'])
    else:
        eligible=[r for r in rows if r['total_correct']>=3*baseline['correct']]
        key=lambda r:(-r['mean_macro'],-r['total_correct'],r['parameters'],r['name'],r['tau'],r['weight'])
    return min(eligible,key=key)


def audit(args):
    out=args.output
    for marker in ('SELECTION_FROZEN.json','TEST_FROZEN.json'):
        check_frozen(out,marker)
    config=read(out/'config.json')
    for name,digest in config['inputs'].items():
        assert file_hash(name)==digest,name
    for name,digest in config['sources'].items():
        assert file_hash(out/'source_snapshot'/name)==digest,name
    ref=Path(config['input_location']['reference'])
    r1,r2=(Path(config['input_location'][s]) for s in ('r1','r2'))
    check_frozen(r2,'SELECTION_FROZEN.json')
    check_frozen(r2,'TEST_FROZEN.json')
    manifests={split:read(ref/f'manifests/{split}.json') for split in ('train','eval','dev')}
    rooms={s:{r['scene'].split('_')[0] for r in m['samples']} for s,m in manifests.items()}
    assert not rooms['train']&rooms['eval'] and not rooms['train']&rooms['dev'] and not rooms['eval']&rooms['dev']
    assert [len(manifests[s]['samples']) for s in ('train','eval','dev')]==[28730,404,3413]
    labels=np.array([s['label'] for s in manifests['train']['samples']])
    counts=np.bincount(labels,minlength=197)+1
    logp=np.log(counts/counts.sum())
    text=np.load(ref/'features/text.npy').astype(float)
    candidates=read(out/'candidate_manifest.json')
    assert len(candidates)==24 and sum(not r['deterministic'] for r in candidates.values())==12
    training_shape=np.load(ref/'features/shape_train.npz')['features'].astype(float)
    training_rgb=np.load(ref/'features/rgb_train/features.npz')['features'].astype(float)
    for modality in ('rgb','shape','joint'):
        x={'rgb':training_rgb,'shape':training_shape,
           'joint':normalize(np.concatenate([training_shape,training_rgb],-1))}[modality]
        sums=np.zeros((197,x.shape[1]))
        np.add.at(sums,labels,x)
        prototype=torch.load(out/f'models/{modality}_prototype_0/model.pt',map_location='cpu',weights_only=True)
        np.testing.assert_allclose(prototype['weight'],normalize(sums),atol=2e-5,rtol=1e-4)
        mean,std=x.mean(0),np.maximum(x.std(0),1e-6)
        expanded=np.concatenate([(x-mean)/std,np.ones((len(x),1))],1)
        covariance=expanded.T@expanded/len(x)
        target_sums=np.zeros((197,expanded.shape[1]))
        np.add.at(target_sums,labels,expanded)
        target=target_sums.T/len(x)
        for strength in (.1,1.,10.):
            record=torch.load(out/f'models/{modality}_ridge_{strength:g}/model.pt',map_location='cpu',weights_only=True)
            np.testing.assert_allclose(record['mean'],mean,atol=2e-6,rtol=1e-4)
            np.testing.assert_allclose(record['std'],std,atol=2e-6,rtol=1e-4)
            regularized=covariance+np.diag([strength]*(covariance.shape[0]-1)+[0.])
            residual=regularized@record['weight'].numpy()-target
            assert np.abs(residual).max()<1e-4,(modality,strength,float(np.abs(residual).max()))
    del training_shape,training_rgb,x,expanded
    for name,item in candidates.items():
        for seed in SEEDS if not item['deterministic'] else (None,):
            folder=out/f'models/{name}'
            if seed is not None:
                folder=folder/f'seed{seed}'
            record=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
            assert record['config_sha256']==file_hash(out/'config.json')
            assert record['train_objects']==28730
            if seed is not None:
                history=read(folder/'history.json')
                assert len(history)==40 and np.isfinite(history).all()
                assert record['epochs']==40 and record['seed']==seed and record['parameter_change_l2']>0
    selection=read(out/'selection.json')
    results=read(out/'results.json')['records']
    audited=[]
    comparisons={}
    maximum_raw_difference=0.
    for role,split,views_path in (('validation','eval',r2/'validation_views'),('historical_test','dev',r2/'historical_test_views')):
        manifest=manifests[split]
        shape=np.load(ref/f'features/shape_{split}.npz')['features'].astype(float)
        rgb=np.load(ref/('features/rgb_dev.npz' if split=='dev' else 'features/rgb_eval/features.npz'))['features'].astype(float)
        views=np.load(views_path/'features.npz')['features'].astype(float)
        np.testing.assert_allclose(normalize(views.mean(1)),rgb,atol=1e-5,rtol=0)
        data=(shape,rgb,views)
        base=z(10*shape@text.T)
        baseline=recount(base,manifest)[0]
        raw={}
        for name,item in candidates.items():
            values=[]
            for seed in SEEDS:
                folder=out/f'models/{name}' if item['deterministic'] else out/f'models/{name}/seed{seed}'
                record=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
                expected=independent_analytic(record,data) if item['deterministic'] else independent_network(record,data)
                saved=np.load(out/role/f'{name}_seed{seed}_raw.npy')
                maximum_raw_difference=max(maximum_raw_difference,float(np.abs(saved-expected).max()))
                np.testing.assert_allclose(saved,expected,atol=2e-3,rtol=3e-4)
                values.append(saved.astype(float))
                if role=='validation':
                    earlier=np.load(out/'validation_raw'/(f'{name}.npy' if item['deterministic'] else f'{name}_seed{seed}.npy'))
                    np.testing.assert_array_equal(saved,earlier)
            raw[name]=values
        raw['S0']=[base]*3
        raw['r2_B']=[np.load(r2/role/f'B_seed{seed}.npy') for seed in SEEDS]
        if role=='validation':
            rebuilt=[]
            for row in read(out/'validation_grid.json'):
                if row['name'] in ('S0','r2_B'):
                    values=raw[row['name']]
                else:
                    values=[(1-row['weight'])*base+row['weight']*z(a-row['tau']*logp) for a in raw[row['name']]]
                measured=[recount(a,manifest)[0] for a in values]
                actual={**row,'total_correct':sum(r['correct'] for r in measured),
                        'mean_macro':float(np.mean([r['class_mean_accuracy'] for r in measured]))}
                assert row['total_correct']==actual['total_correct'],row
                assert abs(row['mean_macro']-actual['mean_macro'])<2e-7,row
                rebuilt.append(actual)
            for mode in ('P','M'):
                expected=select(rebuilt,baseline,mode)
                assert all(selection['locked'][mode][k]==expected[k] for k in ('name','tau','weight','total_correct'))
            for name in candidates:
                expected=select([r for r in rebuilt if r['name']==name],baseline,'P')
                assert all(selection['per_candidate'][name][k]==expected[k] for k in ('name','tau','weight','total_correct'))
        hit_by={name:[] for name in ('S0','r2_B','locked_P','locked_M')}
        for row in (r for r in results if r['role']==role):
            name,seed,sel=row['method'],row['seed'],row['selection']
            actual=np.load(out/role/f'{name}_seed{seed}.npy')
            a=raw[sel['name']][SEEDS.index(seed)]
            expected=a if sel['name'] in ('S0','r2_B') else (1-sel['weight'])*base+sel['weight']*z(a-sel['tau']*logp)
            np.testing.assert_allclose(actual,expected,atol=1e-3,rtol=2e-4)
            measured,hit,_,_=recount(actual,manifest)
            for key,value in measured.items():
                assert abs(value-row['summary'][key])<2e-7,(role,name,key)
            raw_measured=recount(a,manifest)[0]
            audited.append({'role':role,'method':name,'seed':seed,'summary':measured,'raw_summary':raw_measured})
            if name in hit_by:
                hit_by[name].append(hit)
        hit_by={k:np.array(v) for k,v in hit_by.items()}
        if role=='historical_test':
            assert all(int(h.sum())==2262 for h in hit_by['S0'])
        # Reuse the established room bootstrap through explicit aliases.
        for candidate in ('locked_P','locked_M'):
            aliases={'E':hit_by[candidate],'B':hit_by['r2_B'],'C':hit_by['S0'],'D':hit_by['S0'],'S0':hit_by['S0']}
            comparisons[role+'_'+candidate]=bootstrap(aliases,manifest,'E')
        print(json.dumps({'stage':'audited','role':role,'records':sum(r['role']==role for r in audited)}),flush=True)
    summary={'status':'passed','locked':selection['locked'],'records':audited,'comparisons':comparisons,
             'maximum_independent_raw_difference':maximum_raw_difference,
             'auditor_sha256':file_hash(Path(__file__)),'limitation':config['limitation']}
    write_json(out/'audited_summary.json',summary)
    table=['模型\t评估对象数\tTrue正确数\tTrue准确率\tExact Top-1\tTop-5\tClass Mean\tErrors']
    report=['# 多结构与长尾训练 r3结果','','28730训练／404验证／3413历史评估。全部评估此前均有暴露，非独立盲测。',
            '12种网络/损失×3种子，40轮最后epoch；另12种确定性原型/岭回归。仅验证选择融合和P/M方案。','',
            'P为验证Macro不降时最大化True；M为验证True不降时最大化Macro。所有候选同时报告，不能按3413另选。','',
            '锁定选择：'+json.dumps(selection['locked'],ensure_ascii=False),'',
            '| 方法 | True正确数42/43/44 | True准确率 | Exact Top-1 | Top-5 | Class Mean |',
            '|---|---|---:|---:|---:|---:|']
    order=['S0','r2_B','locked_P','locked_M']+list(candidates)
    for name in order:
        rows=[r for r in audited if r['role']=='historical_test' and r['method']==name]
        measures=[r['summary'] for r in rows]
        report.append('| '+name+' | '+'/'.join(str(m['correct']) for m in measures)+' | '+
                      ' | '.join(f'{100*np.mean([m[k] for m in measures]):.2f}%' for k in ('accuracy','exact_accuracy','top5_accuracy','class_mean_accuracy'))+' |')
        for row in rows:
            m=row['summary']
            table.append('\t'.join([name+f'（seed{row["seed"]}）',str(m['count']),str(m['correct'])]+
                [f'{100*m[k]:.2f}%' for k in ('accuracy','exact_accuracy','top5_accuracy','class_mean_accuracy')]+['0']))
    report+=['','所有百分比为三种子均值，非种子集成；确定性模型三个种子重复同一预测。True为既有同义词/复数归一化，Exact为标签字符串匹配。',
             '负结果完整保留。源码与模型封存、36个神经网络训练参数变化/40轮记录、全部验证网格、独立NumPy架构前向、全部分类指标及房间bootstrap通过。',
             '独立从28730训练对象重算原型均值、岭回归标准化统计与正规方程残差，拟合核查通过。',
             '房间bootstrap仅针对固定模型的房间抽样，不消除预训练或历史开发暴露。']
    (out/'report.md').write_text('\n'.join(report)+'\n')
    (out/'table.tsv').write_text('\n'.join(table)+'\n')
    print(json.dumps({'status':'passed','locked':selection['locked'],'maximum_raw_difference':maximum_raw_difference}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    torch.set_num_threads(2)
    audit(parser.parse_args())
