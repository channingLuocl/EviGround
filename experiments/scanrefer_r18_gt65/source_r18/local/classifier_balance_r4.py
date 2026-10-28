"""Fixed training-balancing sweep and validation-selected seed/model ensembles."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from category_common_v82 import file_hash,run_lock
from classifier_sweep_r3 import Classifier,SEEDS,fast_metric,load_data,network_predict,read
from run_rgb_experiment import check_frozen,freeze
from run_storage import write_json
from train_pair_views import standardize

SPECS=[{'name':f'w{int(p*100):03d}','power':p} for p in (.25,.75,1.)]+[
    {'name':f'sample{int(p*100):03d}','sampler':p} for p in (.5,1.)]+[
    {'name':f'deferred{int(p*100):03d}','power':p,'deferred':True} for p in (.5,1.)]+[
    {'name':'focal','power':.5,'focal':2.}, {'name':'smooth','power':.5,'smooth':.05},
    {'name':'width128','power':.5,'width':128},{'name':'width512','power':.5,'width':512},
    {'name':'noise','power':.5,'noise':.01}]
REUSED=('rgb_mlp_ce','rgb_mlp_balanced','rgb_mlp_weighted','view_attention_ce')
TAUS=(0.,.25,.5,1.)
WEIGHTS=(0.,.2,.4,.6,.8,1.)


def make_model(text,architecture='rgb_mlp',width=256):
    model=Classifier(architecture,text)
    if architecture=='rgb_mlp' and width!=256:
        model.head[1]=nn.Linear(text.shape[1],width)
        model.head[4]=nn.Linear(width,len(text))
    return model


def class_weights(counts,power):
    weights=(counts+1).pow(-power)
    return weights/weights.mean()


def training_loss(logits,y,counts,spec,epoch):
    power=spec.get('power',0.)
    if spec.get('deferred') and epoch<20:
        power=0.
    weights=class_weights(counts,power)
    if 'focal' in spec:
        logp=F.log_softmax(logits,dim=-1).gather(1,y[:,None]).squeeze(1)
        values=-(1-logp.exp()).pow(spec['focal'])*logp*weights[y]
        return values.sum()/weights[y].sum()
    return F.cross_entropy(logits,y,weight=weights,label_smoothing=spec.get('smooth',0.))


def sample_indices(labels,counts,spec,generator):
    if 'sampler' in spec:
        weights=(counts+1).pow(-spec['sampler'])[labels]
        return torch.multinomial(weights,len(labels),replacement=True,generator=generator)
    return torch.randperm(len(labels),device=labels.device,generator=generator)


def fit(args,spec,seed,data,labels,text,counts):
    folder=args.output/f'models/{spec["name"]}/seed{seed}'
    folder.mkdir(parents=True)
    torch.manual_seed(seed)
    model=make_model(text,width=spec.get('width',256)).to(args.device)
    initial=[p.detach().clone() for p in model.parameters()]
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    generator=torch.Generator(device=args.device).manual_seed(seed)
    seen=torch.zeros(len(labels),device=args.device,dtype=torch.bool)
    history,sampling=[],[]
    started=time.time()
    for epoch in range(40):
        model.train()
        total=0.
        indices=sample_indices(labels,counts,spec,generator)
        seen[indices]=True
        sampling.append(torch.bincount(labels[indices],minlength=len(text)).cpu().tolist())
        for batch in indices.split(256):
            shape,rgb,views=(x[batch] for x in data)
            if 'noise' in spec:
                rgb=F.normalize(rgb+spec['noise']*torch.randn(rgb.shape,device=rgb.device,generator=generator),dim=-1)
            optimizer.zero_grad(set_to_none=True)
            loss=training_loss(model(shape,rgb,views),labels[batch],counts,spec,epoch)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            optimizer.step()
            total+=float(loss.detach())*len(batch)
        history.append(total/len(labels))
    change=float(sum((p.detach()-p0).square().sum() for p,p0 in zip(model.parameters(),initial)).sqrt())
    assert change>0
    record={'architecture':'rgb_mlp','width':spec.get('width',256),'spec':spec,'seed':seed,'epochs':40,
        'train_objects':len(labels),'unique_objects_seen':int(seen.sum()),'draws':40*len(labels),
        'parameters':sum(p.numel() for p in model.parameters()),'parameter_change_l2':change,
        'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
        'config_sha256':file_hash(args.output/'config.json')}
    torch.save(record,folder/'model.pt')
    write_json(folder/'history.json',history)
    write_json(folder/'sampling.json',sampling)
    print(json.dumps({'stage':'trained','name':spec['name'],'seed':seed,'loss':history[-1],
                      'unique_seen':record['unique_objects_seen'],'seconds':time.time()-started}),flush=True)
    return model.eval(),record['parameters']


def build_candidates(bank,families):
    scores,metadata={},{}
    for family,parameters in families.items():
        raw=[bank[f'{family}_seed{s}'] for s in SEEDS]
        scores['single_'+family]=raw
        metadata['single_'+family]={'family':family,'parameters':parameters,'variants':3,'kind':'single'}
        scores['ensemble_'+family]=[torch.stack([standardize(x) for x in raw]).mean(0)]
        metadata['ensemble_'+family]={'family':family,'parameters':3*parameters,'variants':1,'kind':'seed_mean'}
    for first in ('old_rgb_mlp_ce','old_view_attention_ce'):
        second='old_rgb_mlp_weighted'
        for alpha in (.25,.5,.75):
            name=f'pair_{first}_{alpha:g}'
            scores[name]=[alpha*scores['ensemble_'+first][0]+(1-alpha)*scores['ensemble_'+second][0]]
            metadata[name]={'first':first,'second':second,'alpha':alpha,'variants':1,'kind':'pair',
                            'parameters':3*(families[first]+families[second])}
    return scores,metadata


def measured_row(name,scores,manifest,parameters,tau,weight):
    metrics=[fast_metric(x,manifest) for x in scores]
    correct=[r[0] for r in metrics]
    return {'name':name,'tau':tau,'weight':weight,'parameters':parameters,'variants':len(scores),
            'correct_by_variant':correct,'mean_correct':float(np.mean(correct)),
            'mean_macro':float(np.mean([r[1] for r in metrics]))}


def choose(rows,reference,mode):
    if mode=='P':
        eligible=[r for r in rows if r['mean_macro']>=reference['mean_macro']-1e-7]
        key=lambda r:(-r['mean_correct'],-r['mean_macro'],r['parameters'],r['name'],r['tau'],r['weight'])
    else:
        eligible=[r for r in rows if r['mean_correct']>=reference['mean_correct']-1e-7]
        key=lambda r:(-r['mean_macro'],-r['mean_correct'],r['parameters'],r['name'],r['tau'],r['weight'])
    return min(eligible,key=key)


def setup(args):
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    for marker in ('SELECTION_FROZEN.json','TEST_FROZEN.json'):
        check_frozen(args.r3,marker)
    manifests={s:read(args.reference/f'manifests/{s}.json') for s in ('train','eval','dev')}
    assert [len(manifests[s]['samples']) for s in ('train','eval','dev')]==[28730,404,3413]
    assert all(m['vocabulary']==manifests['train']['vocabulary'] for m in manifests.values())
    rooms={s:{r['scene'].split('_')[0] for r in m['samples']} for s,m in manifests.items()}
    assert not rooms['train']&rooms['eval'] and not rooms['train']&rooms['dev'] and not rooms['eval']&rooms['dev']
    sources=('classifier_balance_r4.py','classifier_sweep_r3.py','train_pair_views.py','category_common_v82.py','run_rgb_experiment.py','run_storage.py')
    paths=[args.design,args.r3/'SELECTION_FROZEN.json',args.r3/'TEST_FROZEN.json']+[
        args.reference/f'manifests/{s}.json' for s in manifests]+[
        args.reference/f'features/{n}' for n in ('text.npy','shape_train.npz','shape_eval.npz','rgb_train/features.npz','rgb_eval/features.npz')]
    cfg={'version':'balance-ensemble-r4','seeds':SEEDS,'epochs':40,'specs':SPECS,'weights':WEIGHTS,'taus':TAUS,
         'input_location':{k:str(getattr(args,k)) for k in ('reference','r1','r2','r3')},
         'sources':{s:file_hash(Path(__file__).parent/s) for s in sources},
         'inputs':{str(p):file_hash(p) for p in paths},'torch':str(torch.__version__),
         'roles':{'train':28730,'validation':404,'historical_test':3413},
         'limitation':'historically exposed evaluation; original frozen 3D pretraining exposure unknown'}
    cfg=json.loads(json.dumps(cfg))
    if (out/'config.json').exists():
        assert read(out/'config.json')==cfg
    else:
        write_json(out/'config.json',cfg)
        (out/'source_snapshot').mkdir()
        for s in sources:
            shutil.copyfile(Path(__file__).parent/s,out/'source_snapshot'/s)
        shutil.copyfile(args.design,out/'source_snapshot/design.md')
    return manifests


def load_family(args,family,seed,text):
    if family.startswith('old_'):
        path=args.r3/f'models/{family[4:]}/seed{seed}/model.pt'
    else:
        path=args.output/f'models/{family}/seed{seed}/model.pt'
    record=torch.load(path,map_location=args.device,weights_only=True)
    if not family.startswith('old_'):
        assert record['config_sha256']==file_hash(args.output/'config.json')
    model=make_model(text,record['architecture'],record.get('width',256)).to(args.device)
    model.load_state_dict(record['state_dict'],strict=True)
    return model.eval(),record['parameters']


def controls(args,role,base):
    result={'S0':[base]}
    for method in ('P','M'):
        result['r3_'+method]=[torch.from_numpy(np.load(args.r3/role/f'locked_{method}_seed{s}.npy')).to(args.device) for s in SEEDS]
    return result


def train_select(args):
    manifests=setup(args)
    assert not (args.output/'SELECTION_FROZEN.json').exists()
    _,train=load_data(args,'train')
    validation,data=load_data(args,'eval')
    text=torch.from_numpy(np.load(args.reference/'features/text.npy')).to(args.device)
    y=torch.tensor([s['label'] for s in manifests['train']['samples']],device=args.device)
    assert (y>=0).all()
    counts=torch.bincount(y,minlength=len(text)).float()
    logp=((counts+1)/(counts.sum()+len(text))).log()
    base=standardize(10*data[0]@text.T)
    bank,families={},{}
    for spec in SPECS:
        for seed in SEEDS:
            model,parameters=fit(args,spec,seed,train,y,text,counts)
            bank[f'{spec["name"]}_seed{seed}']=network_predict(model,data)
            families[spec['name']]=parameters
            del model
    for old in REUSED:
        family='old_'+old
        for seed in SEEDS:
            model,parameters=load_family(args,family,seed,text)
            bank[f'{family}_seed{seed}']=network_predict(model,data)
            families[family]=parameters
    np.savez_compressed(args.output/'validation_bank.npz',**{k:v.cpu().numpy() for k,v in bank.items()})
    candidates,metadata=build_candidates(bank,families)
    grids=[]
    for name,scores in candidates.items():
        for tau in TAUS:
            adjusted=[standardize(s-tau*logp) for s in scores]
            for weight in WEIGHTS:
                values=[(1-weight)*base+weight*s for s in adjusted]
                grids.append(measured_row(name,values,validation,metadata[name]['parameters'],tau,weight))
    references=controls(args,'validation',base)
    control_rows=[measured_row(name,values,validation,{'S0':0,'r3_P':232773,'r3_M':182981}[name],0.,1.) for name,values in references.items()]
    grids+=control_rows
    reference=next(r for r in control_rows if r['name']=='r3_M')
    baseline=next(r for r in control_rows if r['name']=='S0')
    locked={mode:choose(grids,reference,mode) for mode in ('P','M')}
    per_candidate={name:choose([r for r in grids if r['name']==name],baseline,'P') for name in candidates}
    write_json(args.output/'families.json',families)
    write_json(args.output/'candidates.json',metadata)
    write_json(args.output/'validation_grid.json',grids)
    write_json(args.output/'selection.json',{'reference':reference,'baseline':baseline,'locked':locked,'per_candidate':per_candidate})
    files=[str(p.relative_to(args.output)) for pattern in ('models/*/*/*','source_snapshot/*') for p in args.output.glob(pattern) if p.is_file()]
    files+=['config.json','families.json','candidates.json','validation_grid.json','selection.json','validation_bank.npz']
    freeze(args.output,'SELECTION_FROZEN.json',files)
    print(json.dumps({'stage':'selection_frozen','locked':locked}),flush=True)


def evaluate(args):
    setup(args)
    check_frozen(args.output,'SELECTION_FROZEN.json')
    assert not (args.output/'TEST_FROZEN.json').exists()
    selection=read(args.output/'selection.json')
    families=read(args.output/'families.json')
    y=torch.tensor([s['label'] for s in read(args.reference/'manifests/train.json')['samples']],device=args.device)
    text=torch.from_numpy(np.load(args.reference/'features/text.npy')).to(args.device)
    counts=torch.bincount(y,minlength=len(text)).float()+1
    logp=(counts/counts.sum()).log()
    from shape_scale import score
    records=[]
    for role,split in (('validation','eval'),('historical_test','dev')):
        manifest,data=load_data(args,split)
        base=standardize(10*data[0]@text.T)
        bank={}
        for family in families:
            for seed in SEEDS:
                model,_=load_family(args,family,seed,text)
                bank[f'{family}_seed{seed}']=network_predict(model,data)
        candidates,_=build_candidates(bank,families)
        fixed=controls(args,role,base)
        candidates.update(fixed)
        folder=args.output/role
        folder.mkdir()
        np.savez_compressed(folder/'bank.npz',**{k:v.cpu().numpy() for k,v in bank.items()})
        to_score={**selection['per_candidate'],**{name:{'name':name,'weight':1.,'tau':0.} for name in fixed},
                  **{'locked_'+mode:r for mode,r in selection['locked'].items()}}
        for name,sel in to_score.items():
            for index,raw in enumerate(candidates[sel['name']]):
                values=raw if sel['name'] in fixed else (1-sel['weight'])*base+sel['weight']*standardize(raw-sel['tau']*logp)
                saved=values.cpu().numpy()
                np.save(folder/f'{name}_v{index}.npy',saved)
                records.append({'role':role,'method':name,'variant':index,'selection':sel,'summary':score(saved,manifest)})
        print(json.dumps({'stage':role,'locked':[{k:r[k] for k in ('method','variant','summary')} for r in records if r['role']==role and r['method'].startswith('locked_')]}),flush=True)
    write_json(args.output/'results.json',{'records':records})
    freeze(args.output,'TEST_FROZEN.json',['results.json','SELECTION_FROZEN.json']+[
        str(p.relative_to(args.output)) for role in ('validation','historical_test') for p in (args.output/role).glob('*') if p.is_file()])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('train-select','evaluate'))
    for name in ('reference','r1','r2','r3','output','design'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    with run_lock(args.output):
        if args.stage=='train-select':
            train_select(args)
        else:
            with torch.inference_mode():
                evaluate(args)
