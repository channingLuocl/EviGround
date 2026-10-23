"""Check r5 checkpoints, replay forward passes, and independently recount selection/metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from seeground.experiments.audit_classifier_balance_r4 import paired_bootstrap
from seeground.experiments.audit_pair_train_test import recount
from seeground.experiments.audit_pair_views import z
from seeground.categories.category_common_v82 import file_hash
from seeground.classifiers.classifier_backbone_r5 import EPOCHS, PREFIXES, SEEDS, SPECS, make_encoder, read
from seeground.experiments.run_rgb_experiment import check_frozen
from seeground.evaluation.run_storage import write_json


def pick(rows, reference, mode):
    if mode=='P':
        rows=[r for r in rows if r['mean_macro'] >= reference['mean_macro']-1e-7]
        key=lambda r:(-r['mean_correct'],-r['mean_macro'],r['parameters'],r['name'],r['epoch'],r['base'],r['weight'])
    else:
        rows=[r for r in rows if r['mean_correct'] >= reference['mean_correct']]
        key=lambda r:(-r['mean_macro'],-r['mean_correct'],r['parameters'],r['name'],r['epoch'],r['base'],r['weight'])
    return sorted(rows,key=key)[0]


def audit(args):
    root=args.output
    for marker in ('PROTOCOL_FROZEN.json','SELECTION_FROZEN.json','TEST_FROZEN.json'):
        check_frozen(root,marker)
    for marker in ('FEATURES_FROZEN.json','EVAL_FEATURES_FROZEN.json'):
        check_frozen(args.reference,marker)
    check_frozen(args.r4,'TEST_FROZEN.json')
    config=read(root/'config.json')
    for path,sha in config['inputs'].items():
        assert file_hash(path)==sha,path
    for name,sha in config['source'].items():
        assert file_hash(root/'source_snapshot'/name)==sha,name
    manifests={s:read(root/f'manifests/{s}.json') for s in ('train','eval','dev')}
    rooms={s:{r['scene'].split('_')[0] for r in m['samples']} for s,m in manifests.items()}
    assert not rooms['train']&rooms['eval'] and not rooms['train']&rooms['dev'] and not rooms['eval']&rooms['dev']
    assert [len(manifests[s]['samples']) for s in manifests]==[28730,404,3413]
    assert read(root/'SMOKE.json')['status']=='passed'
    original=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    counts={'checkpoints':0,'unchanged_state_arrays':0,'changed_state_arrays':0,'raw_arrays_replayed':0}
    max_forward=0.
    for spec in SPECS:
        for seed in SEEDS:
            directory=root/f'models/{spec}/seed{seed}'
            history=read(directory/'history.json')
            assert [r['epoch'] for r in history]==list(range(1,13))
            assert all(r['objects']==28730 and np.isfinite(r['loss']) for r in history)
            if spec!='frozen':
                gradients=read(directory/'gradients.json')
                assert sum(gradients.values())>0 and np.isfinite(list(gradients.values())).all()
            for epoch in EPOCHS:
                ckpt=torch.load(directory/f'e{epoch}.pt',map_location='cpu',weights_only=True)
                assert (ckpt['spec'],ckpt['seed'],ckpt['epoch'],ckpt['objects'])==(spec,seed,epoch,28730)
                assert ckpt['config_sha256']==file_hash(root/'config.json')
                changed=[]
                for name,value in ckpt['encoder'].items():
                    if torch.equal(value,original[name]):
                        counts['unchanged_state_arrays']+=1
                    else:
                        assert spec!='frozen' and name.startswith(PREFIXES)
                        assert not name.endswith(('running_mean','running_var','num_batches_tracked'))
                        difference=float((value-original[name]).norm())
                        assert abs(difference-ckpt['changed'][name]) < 1e-5
                        changed.append(name)
                        counts['changed_state_arrays']+=1
                assert set(changed)==set(ckpt['changed'])
                assert bool(changed)==(spec!='frozen')
                counts['checkpoints']+=1
                encoder=make_encoder(args)
                encoder.load_state_dict(ckpt['encoder'],strict=True)
                for role,split in (('validation','eval'),('historical_test','dev')):
                    cache=root/f'points/{split}.npy'
                    assert file_hash(cache)==read(cache.with_suffix('.json'))['sha256']
                    if spec=='frozen':
                        embedding=np.load(args.reference/f'features/shape_{split}.npz')['features'].astype(float)
                    else:
                        points=np.load(cache,mmap_mode='r')
                        features=[]
                        with torch.inference_mode():
                            for start in range(0,len(points),32):
                                x=torch.from_numpy(np.array(points[start:start+32])).cuda()
                                features.append(F.normalize(encoder(x).float(),dim=-1).cpu().numpy())
                        embedding=np.concatenate(features).astype(float)
                    actual=embedding@ckpt['head']['weight'].numpy().astype(float).T+ckpt['head']['bias'].numpy()
                    expected=np.load(root/f'{role}/{spec}_e{epoch}_seed{seed}_raw.npy')
                    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
                    max_forward=max(max_forward,float(np.max(np.abs(actual-expected))))
                    counts['raw_arrays_replayed']+=1
                del encoder
    selection=read(root/'selection.json')
    grid=read(root/'validation_grid.json')
    bases={}
    for name in ('S0','r4_M'):
        bases[name]=np.load(root/f'validation/{name}_seed42.npy').astype(float)
    rebuilt=[]
    for row in grid:
        if row['name'] in bases:
            values=[bases[row['name']]]*3
        else:
            values=[(1-row['weight'])*bases[row['base']]+row['weight']*z(np.load(root/
                f'validation_raw/{row["name"]}_e{row["epoch"]}_seed{seed}.npy').astype(float)) for seed in SEEDS]
        measures=[recount(v,manifests['eval'])[0] for v in values]
        new={**row,'correct_by_seed':[m['correct'] for m in measures],
             'mean_correct':float(np.mean([m['correct'] for m in measures])),
             'mean_macro':float(np.mean([m['class_mean_accuracy'] for m in measures]))}
        assert new['correct_by_seed']==row['correct_by_seed']
        assert abs(new['mean_macro']-row['mean_macro']) < 1e-7
        rebuilt.append(new)
    reference=next(r for r in rebuilt if r['name']=='r4_M')
    baseline=next(r for r in rebuilt if r['name']=='S0')
    for mode in ('P','M'):
        chosen=pick(rebuilt,reference,mode)
        assert all(chosen[k]==selection['locked'][mode][k] for k in ('name','epoch','base','weight','mean_correct'))
    for name in SPECS:
        chosen=pick([r for r in rebuilt if r['name']==name],baseline,'P')
        assert all(chosen[k]==selection['per_candidate'][name][k] for k in ('name','epoch','base','weight','mean_correct'))
    text=np.load(args.reference/'features/text.npy').astype(float)
    records=read(root/'results.json')['records']
    comparisons={}
    for role,split in (('validation','eval'),('historical_test','dev')):
        shape=np.load(args.reference/f'features/shape_{split}.npz')['features'].astype(float)
        bases={'S0':z(10*shape@text.T),'r4_M':np.load(args.r4/f'{role}/locked_M_v0.npy').astype(float)}
        hits={}
        for row in [r for r in records if r['role']==role]:
            sel=row['selection']
            name,seed=row['method'],row['seed']
            saved=np.load(root/f'{role}/{name}_seed{seed}.npy')
            expected=bases[sel['name']] if sel['name'] in bases else (1-sel['weight'])*bases[sel['base']]+sel['weight']*z(
                np.load(root/f'{role}/{sel["name"]}_e{sel["epoch"]}_seed{seed}_raw.npy').astype(float))
            np.testing.assert_allclose(saved,expected,atol=2e-5,rtol=2e-5)
            summary,hit,_,_=recount(saved,manifests[split])
            assert all(abs(value-row['summary'][key]) < 1e-10 for key,value in summary.items())
            hits.setdefault(name,[]).append(hit)
        for mode in ('P','M'):
            for ref in ('S0','r4_M'):
                comparisons[f'{role}:locked_{mode}-{ref}']=paired_bootstrap(np.array(hits['locked_'+mode]),np.array(hits[ref]),manifests[split])
    write_json(root/'AUDIT.json',{'status':'passed','counts':counts,'max_forward_abs':max_forward,
               'validation_grid_rows':len(grid),'records_recounted':len(records),'selection':selection,
               'comparisons':comparisons,'scope':'Checkpoint changes and frozen state; all evaluation raw forward passes; validation selection and metrics. Not a full optimization trajectory replay.'})
    print(json.dumps({'status':'passed','counts':counts,'max_forward_abs':max_forward}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','reference','r4','checkpoint'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    audit(args)
