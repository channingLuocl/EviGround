"""Controlled geometry/RGB/regularized-RGB PointNeXt training."""
from eviground.paths import source_file
import argparse
from contextlib import nullcontext
from pathlib import Path
import json
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from eviground.categories.category_common_v82 import file_hash, instance_seed, load_cloud, run_lock
from eviground.classifiers.classifier_backbone_r5 import make_encoder, head_for, predict, read, selector
from eviground.classifiers.classifier_sweep_r3 import fast_metric
from eviground.classifiers.pointnext_color_r6 import ColorPointNext, STEM, color_input, configure, pack_points
from eviground.experiments.run_rgb_experiment import check_frozen, freeze
from eviground.evaluation.run_storage import write_json
from eviground.experiments.train_pair_views import standardize

SEEDS=(42,43,44)
SPECS=('geometry','color','color_regularized')
EPOCHS=(4,8,12)
WEIGHTS=(0.,.2,.4,.6,.8,1.)


def setup(args):
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    manifests={s:read(args.reference/f'manifests/{s}.json') for s in ('train','eval','dev')}
    assert [len(manifests[s]['samples']) for s in manifests]==[28730,404,3413]
    rooms={s:{r['scene'].split('_')[0] for r in m['samples']} for s,m in manifests.items()}
    assert not rooms['train']&rooms['eval'] and not rooms['train']&rooms['dev'] and not rooms['eval']&rooms['dev']
    assert all(m['vocabulary']==manifests['train']['vocabulary'] for m in manifests.values())
    sources=('classifier_color_r6.py','pointnext_color_r6.py','classifier_backbone_r5.py','pointcloud_classifier_v82.py',
             'category_common_v82.py','classifier_sweep_r3.py','train_pair_views.py','run_rgb_experiment.py','run_storage.py')
    paths=[args.design,args.checkpoint,args.smoke,args.r4/'SELECTION_FROZEN.json',args.r4/'validation/locked_M_v0.npy']
    paths += [args.reference/f'manifests/{s}.json' for s in manifests]
    paths += [args.reference/f'features/{n}' for n in ('text.npy','shape_eval.npz')]
    config={'version':'pointnext-color-r6','seeds':SEEDS,'specs':SPECS,'epochs':12,'checkpoints':EPOCHS,'batch':128,
            'body_lr':1e-5,'head_lr':.001,'weight_decay':.01,'warmup_head_epochs':2,'bn':'frozen',
            'loss':'inverse-sqrt-weighted CE','inputs':{str(p):file_hash(p) for p in paths},
            'source':{n:file_hash(source_file(n)) for n in sources},'torch':str(torch.__version__),
            'r5_geometry_caches':str(args.r5/'points'),'limitation':'historically revealed evaluation; original pretraining exposure unknown'}
    config=json.loads(json.dumps(config))
    assert read(args.smoke)['status']=='passed' and read(args.smoke)['objects']==128
    if (out/'config.json').exists():
        assert read(out/'config.json')==config
    else:
        write_json(out/'config.json',config)
        for folder in ('source_snapshot','manifests'):
            (out/folder).mkdir()
        for n in sources:
            shutil.copyfile(source_file(n),out/'source_snapshot'/n)
        shutil.copyfile(args.design,out/'source_snapshot/design.md')
        shutil.copyfile(args.smoke,out/'SMOKE.json')
        for split in manifests:
            shutil.copyfile(args.reference/f'manifests/{split}.json',out/f'manifests/{split}.json')
        freeze(out,'PROTOCOL_FROZEN.json',['config.json','SMOKE.json']+[
            str(p.relative_to(out)) for d in ('source_snapshot','manifests') for p in (out/d).iterdir()])
    check_frozen(out,'PROTOCOL_FROZEN.json')
    return manifests


def points(args,split):
    target=args.output/f'points/{split}.npy'
    seal=target.with_suffix('.json')
    if seal.exists():
        assert file_hash(target)==read(seal)['sha256']
    else:
        assert not target.exists(), 'Incomplete cache preserved; inspect before retrying'
        target.parent.mkdir(exist_ok=True)
        manifest=read(args.output/f'manifests/{split}.json')
        original=args.r5/f'points/{split}.npy'
        assert file_hash(original)==read(original.with_suffix('.json'))['sha256']
        old=np.load(original,mmap_mode='r')
        cache=np.lib.format.open_memmap(target,mode='w+',dtype=np.float32,shape=(len(old),2048,7))
        sources={r['scene']:r['cloud_sha256'] for r in manifest['sources']}
        current=None
        for i,row in enumerate(manifest['samples']):
            if row['scene']!=current:
                current=row['scene']
                assert file_hash(row['cloud'])==sources[current]
                xyz,rgb,ids=load_cloud(row['cloud'])
                assert rgb is not None
            selected=ids==row['bbox_id']
            cache[i]=pack_points(xyz[selected],rgb[selected],instance_seed(42,current,row['bbox_id']))
            np.testing.assert_array_equal(cache[i,:,:4],old[i])
            if (i+1)%2000==0:
                print(json.dumps({'stage':'cache','split':split,'objects':i+1}),flush=True)
        cache.flush()
        del cache
        write_json(seal,{'sha256':file_hash(target),'geometry_sha256':file_hash(original),
                        'manifest_sha256':file_hash(args.output/f'manifests/{split}.json'),'objects':len(old),
                        'channels':'XYZ_height_RGBdiv255minus0.5','geometry':'all_objects_exact_match_r5'})
    return torch.from_numpy(np.load(target)).cuda()


def grid(args,scores,manifest):
    text=torch.from_numpy(np.load(args.reference/'features/text.npy')).cuda()
    shape=torch.from_numpy(np.load(args.reference/'features/shape_eval.npz')['features']).cuda()
    bases={'S0':standardize(10*shape@text.T),'r4_M':torch.from_numpy(np.load(args.r4/'validation/locked_M_v0.npy')).cuda()}
    rows=[]
    for (spec,epoch),(values,parameters) in scores.items():
        for base,b in bases.items():
            for w in WEIGHTS:
                measures=[fast_metric((1-w)*b+w*standardize(v),manifest) for v in values]
                rows.append({'name':spec,'epoch':epoch,'base':base,'weight':w,'parameters':parameters,
                             'correct_by_seed':[m[0] for m in measures],'mean_correct':float(np.mean([m[0] for m in measures])),
                             'mean_macro':float(np.mean([m[1] for m in measures]))})
    for name,b in bases.items():
        correct,macro=fast_metric(b,manifest)
        rows.append({'name':name,'epoch':0,'base':name,'weight':0.,'parameters':0,'correct_by_seed':[correct]*3,
                     'mean_correct':float(correct),'mean_macro':macro})
    reference=next(r for r in rows if r['name']=='r4_M')
    baseline=next(r for r in rows if r['name']=='S0')
    selection={'locked':{m:selector(rows,reference,m) for m in ('P','M')},
               'per_candidate':{name:selector([r for r in rows if r['name']==name],baseline,'P') for name in SPECS}}
    return rows,selection


def train_select(args):
    manifests=setup(args)
    assert not (args.output/'SELECTION_FROZEN.json').exists()
    text=torch.from_numpy(np.load(args.reference/'features/text.npy')).cuda()
    train,validation=points(args,'train'),points(args,'eval')
    y=torch.tensor([r['label'] for r in manifests['train']['samples']],device='cuda')
    assert (y>=0).all()
    weight=(torch.bincount(y,minlength=len(text)).float()+1).rsqrt()
    weight/=weight.mean()
    output=args.output/'validation_raw'
    output.mkdir(exist_ok=True)
    scores={}
    for spec in SPECS:
        for seed in SEEDS:
            torch.manual_seed(seed)
            model=ColorPointNext(make_encoder(args))
            configure(model)
            initial={k:v.detach().clone() for k,v in model.state_dict().items()}
            head=head_for(text)
            trainable=[p for p in model.parameters() if p.requires_grad]
            parameters=sum(p.numel() for p in trainable)+sum(p.numel() for p in head.parameters())
            optimizer=torch.optim.AdamW([{'params':trainable,'lr':1e-5},{'params':head.parameters(),'lr':.001}],weight_decay=.01)
            schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=12)
            generator=torch.Generator(device='cuda').manual_seed(seed)
            colorgen=torch.Generator(device='cuda').manual_seed(seed+1000)
            folder=args.output/f'models/{spec}/seed{seed}'
            folder.mkdir(parents=True,exist_ok=True)
            assert not list(folder.iterdir()), 'Existing models preserved; do not silently restart'
            history=[]
            for epoch in range(1,13):
                started=time.time()
                total,seen=0.,0
                for indices in torch.randperm(len(y),device='cuda',generator=generator).split(128):
                    optimizer.zero_grad(set_to_none=True)
                    x=color_input(train[indices],spec,colorgen)
                    with torch.no_grad() if epoch<=2 else nullcontext():
                        embedding=F.normalize(model(x),dim=-1)
                    loss=F.cross_entropy(head(F.dropout(embedding,.3,True)),y[indices],weight=weight)
                    assert torch.isfinite(loss)
                    loss.backward()
                    if epoch==3 and seen==0:
                        grads={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.requires_grad}
                        rgbgrad=float(dict(model.named_parameters())[STEM+'.weight'].grad[:,4:].norm())
                        assert np.isfinite(list(grads.values())).all() and sum(grads.values())>0
                        assert (rgbgrad==0) if spec=='geometry' else (rgbgrad>0)
                        write_json(folder/'gradients.json',{'all':grads,'rgb_norm':rgbgrad})
                    optimizer.step()
                    total+=float(loss.detach())*len(indices)
                    seen+=len(indices)
                schedule.step()
                assert seen==28730
                record={'epoch':epoch,'objects':seen,'loss':total/seen,'seconds':time.time()-started}
                history.append(record)
                write_json(folder/'history.json',history)
                print(json.dumps({'stage':'epoch','spec':spec,'seed':seed,**record}),flush=True)
                if epoch in EPOCHS:
                    valinput=color_input(validation,'geometry' if spec=='geometry' else 'color')
                    values=predict(model,head,valinput)
                    np.save(output/f'{spec}_e{epoch}_seed{seed}.npy',values.cpu().numpy())
                    scores.setdefault((spec,epoch),([],parameters))[0].append(values)
                    changed={}
                    for n,v in model.state_dict().items():
                        if not torch.equal(v,initial[n]):
                            assert n in dict(model.named_parameters()) and dict(model.named_parameters())[n].requires_grad
                            changed[n]=float((v-initial[n]).norm())
                    rgbnorm=float(dict(model.named_parameters())[STEM+'.weight'][:,4:].norm())
                    assert (rgbnorm==0) if spec=='geometry' else (rgbnorm>0)
                    torch.save({'encoder':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                                'head':{k:v.detach().cpu() for k,v in head.state_dict().items()},'epoch':epoch,
                                'spec':spec,'seed':seed,'objects':seen,'parameters':parameters,'changed':changed,
                                'rgb_weight_norm':rgbnorm,'config_sha256':file_hash(args.output/'config.json')},folder/f'e{epoch}.pt')
            del model,head,initial,optimizer
    rows,selection=grid(args,scores,manifests['eval'])
    write_json(args.output/'validation_grid.json',rows)
    write_json(args.output/'selection.json',selection)
    freeze(args.output,'SELECTION_FROZEN.json',['config.json','PROTOCOL_FROZEN.json','selection.json','validation_grid.json']+[
        str(p.relative_to(args.output)) for d in ('models','validation_raw') for p in (args.output/d).rglob('*') if p.is_file()])
    print(json.dumps({'stage':'selection_frozen','selection':selection}),flush=True)


@torch.inference_mode()
def evaluate(args):
    manifests=setup(args)
    check_frozen(args.output,'SELECTION_FROZEN.json')
    assert not (args.output/'TEST_FROZEN.json').exists()
    from eviground.geometry.shape_scale import score
    text=torch.from_numpy(np.load(args.reference/'features/text.npy')).cuda()
    selection=read(args.output/'selection.json')
    records=[]
    for role,split in (('validation','eval'),('historical_test','dev')):
        inputs=points(args,split)
        shape=torch.from_numpy(np.load(args.reference/f'features/shape_{split}.npz')['features']).cuda()
        bases={'S0':standardize(10*shape@text.T),'r4_M':torch.from_numpy(np.load(args.r4/f'{role}/locked_M_v0.npy')).cuda()}
        folder=args.output/role
        folder.mkdir()
        raw={}
        for spec in SPECS:
            x=color_input(inputs,'geometry' if spec=='geometry' else 'color')
            for epoch in EPOCHS:
                raw[spec,epoch]=[]
                for seed in SEEDS:
                    state=torch.load(args.output/f'models/{spec}/seed{seed}/e{epoch}.pt',map_location='cpu',weights_only=True)
                    assert state['config_sha256']==file_hash(args.output/'config.json')
                    model=ColorPointNext(make_encoder(args))
                    model.load_state_dict(state['encoder'],strict=True)
                    head=head_for(text)
                    head.load_state_dict(state['head'],strict=True)
                    values=predict(model,head,x)
                    if split=='eval':
                        np.testing.assert_allclose(values.cpu().numpy(),np.load(args.output/f'validation_raw/{spec}_e{epoch}_seed{seed}.npy'),atol=1e-5,rtol=1e-5)
                    np.save(folder/f'{spec}_e{epoch}_seed{seed}_raw.npy',values.cpu().numpy())
                    raw[spec,epoch].append(values)
                    del model,head
        selected={**selection['per_candidate'],**{'locked_'+m:s for m,s in selection['locked'].items()}}
        selected.update({n:{'name':n,'base':n,'epoch':0,'weight':0.} for n in bases})
        for name,sel in selected.items():
            values=[bases[sel['name']]]*3 if sel['name'] in bases else [
                (1-sel['weight'])*bases[sel['base']]+sel['weight']*standardize(v) for v in raw[sel['name'],sel['epoch']]]
            for seed,v in zip(SEEDS,values):
                saved=v.cpu().numpy()
                np.save(folder/f'{name}_seed{seed}.npy',saved)
                records.append({'role':role,'method':name,'seed':seed,'selection':sel,'summary':score(saved,manifests[split])})
        print(json.dumps({'stage':role,'locked':[r for r in records if r['role']==role and r['method'].startswith('locked_')]}),flush=True)
    write_json(args.output/'results.json',{'records':records})
    freeze(args.output,'TEST_FROZEN.json',['results.json','SELECTION_FROZEN.json']+[
        str(p.relative_to(args.output)) for d in ('validation','historical_test') for p in (args.output/d).iterdir()])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('train-select','evaluate'))
    for name in ('reference','r4','r5','output','checkpoint','design','smoke'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    with run_lock(args.output):
        (train_select if args.stage=='train-select' else evaluate)(args)
