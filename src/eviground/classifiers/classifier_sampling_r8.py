"""Frozen PointNeXt sampling/yaw voting, with three validation objectives."""
from eviground.paths import source_file
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from eviground.categories.category_common_v82 import file_hash, instance_seed, load_cloud, run_lock
from eviground.classifiers.classifier_backbone_r5 import make_encoder, read
from eviground.experiments.audit_pair_views import z
from eviground.experiments.audit_pair_train_test import recount
import eviground.experiments.audit_classifier_balance_r4 as audit_classifier_balance_r4
from eviground.classifiers.pointcloud_classifier_v82 import prepare_points
from eviground.experiments.run_rgb_experiment import check_frozen, freeze
from eviground.evaluation.run_storage import write_json

PASSES=tuple((seed,0.) for seed in range(42,49))+((42,-10.),(42,10.))
GROUPS={'single':(0,), 'sample3':(0,1,2), 'sample7':tuple(range(7)),
        'yaw3':(0,7,8), 'sample_yaw9':tuple(range(9))}
WEIGHTS=(0.,.2,.4,.6,.8,1.)


def yaw(points,degrees):
    result=points.copy()
    angle=np.deg2rad(degrees)
    x,y=points[:,0],points[:,1]
    result[:,0]=x*np.cos(angle)-y*np.sin(angle)
    result[:,1]=x*np.sin(angle)+y*np.cos(angle)
    return result


def rgb_branch(reference,base):
    return (reference-.4*base)/.6


def vote(features,text,indices):
    return 10*np.asarray(features,dtype=np.float64)[:,indices].mean(1)@np.asarray(text,dtype=np.float64).T


def choose(rows,reference,mode):
    if mode=='B':
        rows=[r for r in rows if r['macro']>=reference['macro']-1e-10]
    elif mode=='M':
        rows=[r for r in rows if r['correct']>=reference['correct']]
    else:
        assert mode=='P'
    def key(r):
        metrics=(-r['macro'],-r['correct']) if mode=='M' else (-r['correct'],-r['macro'])
        return (*metrics,r['passes'],r['name'],r['rgb_weight'])
    return min(rows,key=key)


def setup(args):
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    manifests={s:read(args.reference/f'manifests/{s}.json') for s in ('train','eval','dev')}
    rooms={s:{r['scene'].split('_')[0] for r in m['samples']} for s,m in manifests.items()}
    assert not rooms['train']&rooms['eval'] and not rooms['train']&rooms['dev'] and not rooms['eval']&rooms['dev']
    assert [len(manifests[s]['samples']) for s in manifests]==[28730,404,3413]
    r4=read(args.r4/'selection.json')['locked']['M']
    assert r4['name']=='pair_old_rgb_mlp_ce_0.5' and r4['tau']==0 and r4['weight']==.6
    for marker in ('FEATURES_FROZEN.json','EVAL_FEATURES_FROZEN.json'):
        check_frozen(args.reference,marker)
    check_frozen(args.r4,'TEST_FROZEN.json')
    sources=('classifier_sampling_r8.py','classifier_backbone_r5.py','pointcloud_classifier_v82.py',
             'category_common_v82.py','audit_pair_views.py','audit_pair_train_test.py',
             'run_rgb_experiment.py','run_storage.py')
    inputs=[args.design,args.checkpoint,args.reference/'features/text.npy',args.r4/'selection.json',
            args.r4/'validation/S0_v0.npy',args.r4/'validation/locked_M_v0.npy',
            args.reference/'features/shape_train.npz',args.reference/'features/shape_eval.npz']
    inputs += [args.reference/f'manifests/{s}.json' for s in manifests]
    config={'version':'pointnext-sampling-vote-r8','passes':PASSES,'groups':GROUPS,'rgb_weights':WEIGHTS,
            'objectives':{'P':'max validation True','B':'max True with Macro floor r4 M','M':'max Macro with True floor r4 M'},
            'source':{n:file_hash(source_file(n)) for n in sources},
            'inputs':{str(p):file_hash(p) for p in inputs},'torch':str(torch.__version__),
            'limitation':'historically revealed evaluation; original pretraining scene exposure unknown'}
    config=json.loads(json.dumps(config))
    if (out/'config.json').exists():
        assert read(out/'config.json')==config
    else:
        write_json(out/'config.json',config)
        for d in ('source_snapshot','manifests'):
            (out/d).mkdir()
        for n in sources:
            shutil.copyfile(source_file(n),out/'source_snapshot'/n)
        shutil.copyfile(args.design,out/'source_snapshot/design.md')
        for split in manifests:
            shutil.copyfile(args.reference/f'manifests/{split}.json',out/f'manifests/{split}.json')
        freeze(out,'PROTOCOL_FROZEN.json',['config.json']+[
            str(p.relative_to(out)) for d in ('source_snapshot','manifests') for p in (out/d).iterdir()])
    check_frozen(out,'PROTOCOL_FROZEN.json')
    return manifests


def preflight(args):
    manifests=setup(args)
    assert not (args.output/'PREFLIGHT.json').exists()
    model=make_encoder(args)
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    expected=np.load(args.reference/'features/shape_train.npz')['features']
    records=[]
    for i in (0,14365,28729):
        row=manifests['train']['samples'][i]
        xyz,_,ids=load_cloud(row['cloud'])
        x=prepare_points(xyz[ids==row['bbox_id']],None,'pointnext',instance_seed(42,row['scene'],row['bbox_id']))
        with torch.inference_mode():
            a=F.normalize(model(torch.from_numpy(x[None]).cuda()),dim=-1).cpu().numpy()[0]
            b=F.normalize(model(torch.from_numpy(x[None]).cuda()),dim=-1).cpu().numpy()[0]
        np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(a,expected[i],atol=2e-5,rtol=2e-4)
        records.append({'index':i,'max_abs':float(np.max(np.abs(a-expected[i])))})
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    write_json(args.output/'PREFLIGHT.json',{'status':'passed','records':records,'all_model_state':'unchanged',
               'repeat':'exact','checkpoint_sha256':file_hash(args.checkpoint)})
    print(json.dumps({'stage':'preflight_passed','records':records}),flush=True)


@torch.inference_mode()
def extract(args,split):
    path=args.output/f'features_{split}.npz'
    marker=path.with_suffix('.json')
    if marker.exists():
        assert file_hash(path)==read(marker)['sha256']
        return np.load(path)['features']
    assert not path.exists(), 'Incomplete extraction preserved'
    manifest=read(args.output/f'manifests/{split}.json')
    model=make_encoder(args)
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    values=[]
    current=None
    started=time.time()
    sources={r['scene']:r['cloud_sha256'] for r in manifest['sources']}
    for i,row in enumerate(manifest['samples']):
        if row['scene']!=current:
            current=row['scene']
            assert file_hash(row['cloud'])==sources[current]
            xyz,_,ids=load_cloud(row['cloud'])
        obj=xyz[ids==row['bbox_id']]
        points=[]
        for seed,degrees in PASSES:
            x=prepare_points(obj,None,'pointnext',instance_seed(seed,current,row['bbox_id']))
            points.append(yaw(x,degrees))
        embedding=F.normalize(model(torch.from_numpy(np.stack(points)).cuda()).float(),dim=-1).cpu().numpy()
        assert embedding.shape==(9,512) and np.isfinite(embedding).all()
        values.append(embedding)
        if (i+1)%500==0:
            print(json.dumps({'stage':'extract','split':split,'objects':i+1,'seconds':time.time()-started}),flush=True)
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    values=np.stack(values)
    old=np.load(args.reference/f'features/shape_{split}.npz')['features']
    np.testing.assert_allclose(values[:,0],old,atol=2e-5,rtol=2e-4)
    np.savez_compressed(path,features=values)
    write_json(marker,{'sha256':file_hash(path),'objects':len(values),'passes':9,'seconds':time.time()-started,
               'single_bridge_max_abs':float(np.max(np.abs(values[:,0]-old))),'model_state':'unchanged'})
    return values


def bases(args,role):
    base=np.load(args.r4/f'{role}/S0_v0.npy').astype(float)
    reference=np.load(args.r4/f'{role}/locked_M_v0.npy').astype(float)
    rgb=rgb_branch(reference,base)
    np.testing.assert_allclose(.4*base+.6*rgb,reference,atol=2e-15,rtol=2e-15)
    return base,reference,rgb


def select(args):
    manifests=setup(args)
    assert read(args.output/'PREFLIGHT.json')['status']=='passed'
    assert not (args.output/'SELECTION_FROZEN.json').exists()
    text=np.load(args.reference/'features/text.npy')
    features=extract(args,'eval')
    base,reference,rgb=bases(args,'validation')
    rows=[]
    for name,indices in GROUPS.items():
        logits=z(vote(features,text,indices))
        if name=='single':
            np.testing.assert_allclose(logits,base,atol=2e-5,rtol=2e-4)
        for w in WEIGHTS:
            summary=recount((1-w)*logits+w*rgb,manifests['eval'])[0]
            rows.append({'name':name,'rgb_weight':w,'passes':len(indices),'correct':summary['correct'],
                         'macro':summary['class_mean_accuracy']})
    for name,v in (('S0',base),('r4_M',reference)):
        summary=recount(v,manifests['eval'])[0]
        rows.append({'name':name,'rgb_weight':0. if name=='S0' else .6,'passes':1,
                     'correct':summary['correct'],'macro':summary['class_mean_accuracy']})
    ref=next(r for r in rows if r['name']=='r4_M')
    selection={'locked':{m:choose(rows,ref,m) for m in ('P','B','M')},
               'per_candidate':{name:choose([r for r in rows if r['name']==name],ref,'P') for name in GROUPS}}
    write_json(args.output/'validation_grid.json',rows)
    write_json(args.output/'selection.json',selection)
    freeze(args.output,'SELECTION_FROZEN.json',['config.json','PROTOCOL_FROZEN.json','PREFLIGHT.json',
              'features_eval.npz','features_eval.json','validation_grid.json','selection.json'])
    print(json.dumps({'stage':'selection_frozen','selection':selection}),flush=True)


def evaluate(args):
    manifests=setup(args)
    check_frozen(args.output,'SELECTION_FROZEN.json')
    assert not (args.output/'TEST_FROZEN.json').exists()
    selection=read(args.output/'selection.json')
    text=np.load(args.reference/'features/text.npy')
    records=[]
    for role,split in (('validation','eval'),('historical_test','dev')):
        features=extract(args,split)
        base,reference,rgb=bases(args,role)
        chosen={**selection['per_candidate'],**{'locked_'+m:r for m,r in selection['locked'].items()}}
        chosen.update({'S0':{'name':'S0','rgb_weight':0.,'passes':1},'r4_M':{'name':'r4_M','rgb_weight':.6,'passes':1}})
        folder=args.output/role
        folder.mkdir()
        for name,sel in chosen.items():
            if sel['name'] in ('S0','r4_M'):
                values=base if sel['name']=='S0' else reference
            else:
                shape=z(vote(features,text,GROUPS[sel['name']]))
                w=sel['rgb_weight']
                values=(1-w)*shape+w*rgb
            np.save(folder/f'{name}.npy',values)
            records.append({'role':role,'method':name,'selection':sel,'summary':recount(values,manifests[split])[0]})
        print(json.dumps({'stage':role,'locked':[r for r in records if r['role']==role and r['method'].startswith('locked_')]}),flush=True)
    write_json(args.output/'results.json',{'records':records})
    freeze(args.output,'TEST_FROZEN.json',['results.json','SELECTION_FROZEN.json','features_dev.npz','features_dev.json']+[
        str(p.relative_to(args.output)) for d in ('validation','historical_test') for p in (args.output/d).iterdir()])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('preflight','select','evaluate'))
    for n in ('reference','r4','output','checkpoint','design'):
        parser.add_argument('--'+n,type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    with run_lock(args.output):
        globals()[args.stage](args)
