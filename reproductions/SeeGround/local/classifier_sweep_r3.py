"""Train a fixed architecture/loss sweep; select on revealed validation before historical scoring."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from category_common_v82 import file_hash, normalized_category, run_lock
from run_rgb_experiment import check_frozen, freeze
from run_storage import write_json
from train_pair_views import standardize

SEEDS = (42, 43, 44)
SPECS = [('rgb_mlp', k) for k in ('ce','balanced','weighted')]+[
    (a,k) for a in ('rgb_adapter','shape_adapter') for k in ('ce','balanced')]+[
    ('joint_mlp',k) for k in ('ce','balanced','mixup')]+[
    ('view_attention',k) for k in ('ce','balanced')]
TYPES = ('rgb','shape','joint')
WEIGHTS = (0., .2, .4, .6, .8, 1.)
TAUS = (0., .25, .5, 1.)


def read(path):
    return json.loads(Path(path).read_text())


class Classifier(nn.Module):
    def __init__(self, architecture, text):
        super().__init__()
        self.architecture=architecture
        dimension, classes=text.shape[1],len(text)
        if architecture.endswith('adapter'):
            self.adapter=nn.Sequential(nn.Linear(dimension,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,dimension))
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)
            self.head=nn.Linear(dimension,classes)
            with torch.no_grad():
                self.head.weight.copy_(10*text)
                self.head.bias.zero_()
        elif architecture=='view_attention':
            self.project=nn.Linear(dimension,128)
            self.attention=nn.MultiheadAttention(128,4,dropout=.1,batch_first=True)
            self.norm=nn.LayerNorm(128)
            self.head=nn.Sequential(nn.LayerNorm(dimension+128),nn.Dropout(.3),nn.Linear(dimension+128,classes))
        else:
            size=2*dimension if architecture=='joint_mlp' else dimension
            self.head=nn.Sequential(nn.LayerNorm(size),nn.Linear(size,256),nn.GELU(),nn.Dropout(.3),nn.Linear(256,classes))

    def forward(self, shape, rgb, views):
        if self.architecture=='view_attention':
            v=F.dropout(views,.1,self.training)
            v=self.project(v)
            h,_=self.attention(v,v,v,need_weights=False)
            x=torch.cat([rgb,self.norm(v+h).mean(1)],-1)
            return self.head(x)
        if self.architecture.startswith('rgb'):
            x=rgb
        elif self.architecture.startswith('shape'):
            x=shape
        else:
            x=torch.cat([shape,rgb],-1)
        x=F.dropout(x,.1,self.training)
        if self.architecture.endswith('adapter'):
            x=F.normalize(x+self.adapter(x),dim=-1)
        return self.head(x)


def loss_function(logits, labels, counts, kind):
    if kind=='balanced':
        return F.cross_entropy(logits+(counts+1).log(),labels)
    if kind=='weighted':
        weight=(counts+1).rsqrt()
        return F.cross_entropy(logits,labels,weight=weight/weight.mean())
    return F.cross_entropy(logits,labels)


def mix_features(values, permutation, amount):
    return tuple(amount*x+(1-amount)*x[permutation] for x in values)


def fit_network(args, architecture, kind, seed, train, labels, text, counts):
    folder=args.output/f'models/{architecture}_{kind}/seed{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    checkpoint=folder/'model.pt'
    if checkpoint.exists():
        raise ValueError(f'Existing model preserved: {checkpoint}')
    torch.manual_seed(seed)
    model=Classifier(architecture,text).to(args.device)
    initial=[p.detach().clone() for p in model.parameters()]
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    generator=torch.Generator(device=args.device).manual_seed(seed)
    rng=np.random.default_rng(seed+100)
    history=[]
    started=time.time()
    for epoch in range(40):
        model.train()
        total=0.
        for indices in torch.randperm(len(labels),device=args.device,generator=generator).split(256):
            values=tuple(t[indices] for t in train)
            y=labels[indices]
            optimizer.zero_grad(set_to_none=True)
            if kind=='mixup':
                amount=float(rng.beta(.2,.2))
                permutation=torch.randperm(len(y),device=args.device,generator=generator)
                logits=model(*mix_features(values,permutation,amount))
                loss=amount*loss_function(logits,y,counts,'ce')+(1-amount)*loss_function(logits,y[permutation],counts,'ce')
            else:
                loss=loss_function(model(*values),y,counts,kind)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            total+=float(loss.detach())*len(y)
        history.append(total/len(labels))
    change=float(sum((p.detach()-p0).square().sum() for p,p0 in zip(model.parameters(),initial)).sqrt())
    assert change>0
    record={'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
            'architecture':architecture,'loss':kind,'seed':seed,'epochs':40,'train_objects':len(labels),
            'parameters':sum(p.numel() for p in model.parameters()),'parameter_change_l2':change,
            'config_sha256':file_hash(args.output/'config.json')}
    torch.save(record,checkpoint)
    write_json(folder/'history.json',history)
    print(json.dumps({'stage':'network','name':architecture+'_'+kind,'seed':seed,'loss':history[-1],
                      'seconds':time.time()-started,'parameter_change':change}),flush=True)
    return model.eval(),record['parameters']


def get_values(data, modality):
    shape,rgb,_=data
    return {'shape':shape,'rgb':rgb,'joint':F.normalize(torch.cat([shape,rgb],-1),dim=-1)}[modality]


def fit_analytic(x, labels, classes, kind, strength):
    if kind=='prototype':
        sums=torch.zeros(classes,x.shape[1],device=x.device)
        sums.index_add_(0,labels,x)
        return {'kind':kind,'weight':F.normalize(sums,dim=-1).cpu()}
    # Train-only feature standardization; mean loss with an unpenalized intercept.
    mean=x.mean(0)
    std=x.std(0,correction=0).clamp_min(1e-6)
    augmented=torch.cat([(x-mean)/std,torch.ones(len(x),1,device=x.device)],1).double()
    target=F.one_hot(labels,classes).double()
    reg=torch.eye(augmented.shape[1],device=x.device,dtype=torch.float64)*strength
    reg[-1,-1]=0
    weight=torch.linalg.solve(augmented.T@augmented/len(x)+reg,augmented.T@target/len(x))
    return {'kind':kind,'mean':mean.cpu(),'std':std.cpu(),'weight':weight.cpu(),'strength':strength}


def analytic_predict(record, x):
    if record['kind']=='prototype':
        return 10*x@record['weight'].to(x.device).T
    x=(x-record['mean'].to(x.device))/record['std'].to(x.device)
    x=torch.cat([x,torch.ones(len(x),1,device=x.device)],1).double()
    return (x@record['weight'].to(x.device)).float()


@torch.inference_mode()
def network_predict(model, data):
    model.eval()
    return torch.cat([model(*(t[start:start+512] for t in data)) for start in range(0,len(data[0]),512)])


def load_data(args, split):
    ref=args.reference
    manifest=read(ref/f'manifests/{split}.json')
    shape=torch.from_numpy(np.load(ref/f'features/shape_{split}.npz')['features']).to(args.device)
    path=ref/('features/rgb_dev.npz' if split=='dev' else f'features/rgb_{split}/features.npz')
    record=np.load(path)
    assert record['valid'].all()
    rgb=torch.from_numpy(record['features']).to(args.device)
    views_path={'train':args.r1/'views','eval':args.r2/'validation_views','dev':args.r2/'historical_test_views'}[split]
    seal=read(views_path/'COMPLETE.json')
    assert file_hash(views_path/'features.npz')==seal['features_sha256']
    views=torch.from_numpy(np.load(views_path/'features.npz')['features']).to(args.device)
    assert len(shape)==len(rgb)==len(views)==len(manifest['samples'])
    torch.testing.assert_close(F.normalize(views.mean(1),dim=-1),rgb,atol=1e-5,rtol=0)
    return manifest,(shape,rgb,views)


def fast_metric(values, manifest):
    vocab=tuple(manifest['vocabulary'])
    targets=tuple(s['category'] for s in manifest['samples'])
    names,target,group_count=metric_targets(vocab,targets)
    prediction=values.argmax(-1).detach().cpu().numpy()
    hit=names[prediction]==target
    sums=np.bincount(target,weights=hit,minlength=len(group_count))
    return int(hit.sum()),float((sums/group_count).mean())


@lru_cache(None)
def metric_targets(vocabulary, categories):
    normalized=[normalized_category(c) for c in categories]
    unique=sorted(set(normalized))
    mapping={s:i for i,s in enumerate(unique)}
    target=np.array([mapping[s] for s in normalized])
    names=np.array([mapping.get(normalized_category(s),-1) for s in vocabulary])
    return names,target,np.bincount(target,minlength=len(unique))


def selection_key(row, mode):
    order=(-row['total_correct'],-row['mean_macro']) if mode=='P' else (-row['mean_macro'],-row['total_correct'])
    return (*order,row['parameters'],row['name'],row['tau'],row['weight'])


def choose(rows, base_correct, base_macro, mode):
    if mode=='P':
        eligible=[r for r in rows if r['mean_macro']>=base_macro-1e-7]
    else:
        eligible=[r for r in rows if r['total_correct']>=3*base_correct]
    return min(eligible,key=lambda r:selection_key(r,mode))


def grid_for(name, logits, base, logp, manifest, parameters):
    rows=[]
    for tau in TAUS:
        adjusted=[standardize(a-tau*logp) for a in logits]
        for weight in WEIGHTS:
            metrics=[fast_metric((1-weight)*base+weight*a,manifest) for a in adjusted]
            rows.append({'name':name,'tau':tau,'weight':weight,'parameters':parameters,
                'total_correct':sum(v[0] for v in metrics),'mean_macro':float(np.mean([v[1] for v in metrics]))})
    return rows


def setup(args):
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    manifests={s:read(args.reference/f'manifests/{s}.json') for s in ('train','eval','dev')}
    assert [len(manifests[s]['samples']) for s in ('train','eval','dev')]==[28730,404,3413]
    assert all(m['vocabulary']==manifests['train']['vocabulary'] for m in manifests.values())
    rooms={s:{r['scene'].split('_')[0] for r in m['samples']} for s,m in manifests.items()}
    assert not rooms['train']&rooms['eval'] and not rooms['train']&rooms['dev'] and not rooms['eval']&rooms['dev']
    sources=('classifier_sweep_r3.py','train_pair_views.py','category_common_v82.py','run_storage.py','run_rgb_experiment.py')
    paths=[args.design]+[args.reference/f'manifests/{s}.json' for s in manifests]+[
        args.reference/f'features/{n}' for n in ('text.npy','shape_train.npz','shape_eval.npz','rgb_train/features.npz','rgb_eval/features.npz')]+[
        args.r1/'views/COMPLETE.json',args.r1/'views/features.npz',args.r2/'SELECTION_FROZEN.json']
    config={'version':'architecture-longtail-sweep-r3','epochs':40,'seeds':list(SEEDS),'specs':SPECS,
        'sources':{n:file_hash(Path(__file__).parent/n) for n in sources},
        'inputs':{str(p):file_hash(p) for p in paths},'roles':{'train':28730,'validation':404,'historical_test':3413},
        'input_location':{'reference':str(args.reference),'r1':str(args.r1),'r2':str(args.r2)},
        'weights':WEIGHTS,'taus':TAUS,'torch':str(torch.__version__),
        'limitation':'historically exposed evaluation; frozen 3D pretraining scene exposure unknown'}
    config=json.loads(json.dumps(config))
    if (out/'config.json').exists():
        assert read(out/'config.json')==config
    else:
        write_json(out/'config.json',config)
        (out/'source_snapshot').mkdir()
        for name in sources:
            shutil.copyfile(Path(__file__).parent/name,out/'source_snapshot'/name)
        shutil.copyfile(args.design,out/'source_snapshot/design.md')
    check_frozen(args.r2,'SELECTION_FROZEN.json')
    return manifests


def train_select(args):
    manifests=setup(args)
    assert not (args.output/'SELECTION_FROZEN.json').exists()
    train_m,train=load_data(args,'train')
    validation,data=load_data(args,'eval')
    text=torch.from_numpy(np.load(args.reference/'features/text.npy')).to(args.device)
    labels=torch.tensor([s['label'] for s in train_m['samples']],device=args.device)
    assert (labels>=0).all()
    counts=torch.bincount(labels,minlength=len(text)).float()
    logp=((counts+1)/(counts.sum()+len(text))).log()
    base=standardize(10*data[0]@text.T)
    base_correct,base_macro=fast_metric(base,validation)
    folder=args.output/'validation_raw'
    folder.mkdir()
    candidates={}
    for architecture,kind in SPECS:
        name=architecture+'_'+kind
        candidates[name]={'parameters':None,'scores':[]}
        for seed in SEEDS:
            model,parameters=fit_network(args,architecture,kind,seed,train,labels,text,counts)
            values=network_predict(model,data)
            np.save(folder/f'{name}_seed{seed}.npy',values.cpu().numpy())
            candidates[name]['parameters']=parameters
            candidates[name]['scores'].append(values)
            del model
    for modality in TYPES:
        x=get_values(train,modality)
        for kind,strength in [('prototype',0.)]+[('ridge',t) for t in (.1,1.,10.)]:
            name=f'{modality}_{kind}_{strength:g}'
            record=fit_analytic(x,labels,len(text),kind,strength)
            record.update(modality=modality,train_objects=len(labels),config_sha256=file_hash(args.output/'config.json'))
            path=args.output/f'models/{name}/model.pt'
            path.parent.mkdir()
            torch.save(record,path)
            values=analytic_predict(record,get_values(data,modality))
            np.save(folder/f'{name}.npy',values.cpu().numpy())
            parameters=int(record['weight'].numel())
            candidates[name]={'parameters':parameters,'scores':[values]*3}
            print(json.dumps({'stage':'analytic','name':name}),flush=True)
    grids=[]
    for name,item in candidates.items():
        grids+=grid_for(name,item['scores'],base,logp,validation,item['parameters'])
    grids.append({'name':'S0','tau':0.,'weight':0.,'parameters':0,'total_correct':3*base_correct,'mean_macro':base_macro})
    r2=[torch.from_numpy(np.load(args.r2/f'validation/B_seed{seed}.npy')).to(args.device) for seed in SEEDS]
    measured=[fast_metric(a,validation) for a in r2]
    grids.append({'name':'r2_B','tau':0.,'weight':1.,'parameters':101061,'total_correct':sum(a[0] for a in measured),
                  'mean_macro':float(np.mean([a[1] for a in measured]))})
    selection={mode:choose(grids,base_correct,base_macro,mode) for mode in ('P','M')}
    # Each candidate gets its own validation-selected P setting for transparent controls.
    per_candidate={name:choose([r for r in grids if r['name']==name],base_correct,base_macro,'P') for name in candidates}
    write_json(args.output/'selection.json',{'locked':selection,'per_candidate':per_candidate})
    write_json(args.output/'validation_grid.json',grids)
    write_json(args.output/'candidate_manifest.json',{name:{'parameters':r['parameters'],'deterministic':name not in {a+'_'+k for a,k in SPECS}} for name,r in candidates.items()})
    files=[str(p.relative_to(args.output)) for pattern in ('models/*/*/*','models/*/*','validation_raw/*','source_snapshot/*') for p in args.output.glob(pattern) if p.is_file()]
    files+=['config.json','selection.json','validation_grid.json','candidate_manifest.json']
    freeze(args.output,'SELECTION_FROZEN.json',sorted(set(files)))
    print(json.dumps({'stage':'selection_frozen','locked':selection}),flush=True)


def predict_candidates(args,data,text):
    candidates={}
    for architecture,kind in SPECS:
        name=architecture+'_'+kind
        values=[]
        for seed in SEEDS:
            record=torch.load(args.output/f'models/{name}/seed{seed}/model.pt',map_location=args.device,weights_only=True)
            assert record['config_sha256']==file_hash(args.output/'config.json')
            model=Classifier(architecture,text).to(args.device)
            model.load_state_dict(record['state_dict'],strict=True)
            values.append(network_predict(model,data))
        candidates[name]=values
    for name,record in read(args.output/'candidate_manifest.json').items():
        if record['deterministic']:
            ckpt=torch.load(args.output/f'models/{name}/model.pt',map_location='cpu',weights_only=True)
            values=analytic_predict(ckpt,get_values(data,ckpt['modality']))
            candidates[name]=[values]*3
    return candidates


def evaluate(args):
    setup(args)
    check_frozen(args.output,'SELECTION_FROZEN.json')
    assert not (args.output/'TEST_FROZEN.json').exists()
    selection=read(args.output/'selection.json')
    text=torch.from_numpy(np.load(args.reference/'features/text.npy')).to(args.device)
    labels=torch.tensor([s['label'] for s in read(args.reference/'manifests/train.json')['samples']],device=args.device)
    counts=torch.bincount(labels,minlength=len(text)).float()+1
    logp=(counts/counts.sum()).log()
    from shape_scale import score
    records=[]
    for role,split in (('validation','eval'),('historical_test','dev')):
        manifest,data=load_data(args,split)
        candidates=predict_candidates(args,data,text)
        base=standardize(10*data[0]@text.T)
        candidates['S0']=[base]*3
        candidates['r2_B']=[torch.from_numpy(np.load(args.r2/f'{role}/B_seed{seed}.npy')).to(args.device) for seed in SEEDS]
        folder=args.output/role
        folder.mkdir()
        to_score={**selection['per_candidate'],'S0':{'name':'S0','tau':0.,'weight':0.},
                  'r2_B':{'name':'r2_B','tau':0.,'weight':1.},
                  **{'locked_'+mode:r for mode,r in selection['locked'].items()}}
        for name,sel in to_score.items():
            for seed,raw in zip(SEEDS,candidates[sel['name']]):
                if sel['name'] in ('S0','r2_B'):
                    values=raw
                else:
                    values=(1-sel['weight'])*base+sel['weight']*standardize(raw-sel['tau']*logp)
                saved=values.cpu().numpy()
                np.save(folder/f'{name}_seed{seed}.npy',saved)
                records.append({'role':role,'method':name,'seed':seed,'selection':sel,'summary':score(saved,manifest)})
                if name in selection['per_candidate']:
                    np.save(folder/f'{name}_seed{seed}_raw.npy',raw.cpu().numpy())
        print(json.dumps({'stage':role,'locked':[{k:r[k] for k in ('method','seed','summary')} for r in records if r['role']==role and r['method'].startswith('locked_')]}),flush=True)
    write_json(args.output/'results.json',{'records':records})
    freeze(args.output,'TEST_FROZEN.json',['results.json','SELECTION_FROZEN.json']+[
        str(p.relative_to(args.output)) for role in ('validation','historical_test') for p in (args.output/role).glob('*.npy')])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('train-select','evaluate'))
    for name in ('reference','r1','r2','output','design'):
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
