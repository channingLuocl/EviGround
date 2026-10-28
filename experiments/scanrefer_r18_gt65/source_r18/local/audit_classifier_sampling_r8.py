"""Replay frozen encoder passes and independently check r8 voting and validation choices."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from audit_classifier_balance_r4 import paired_bootstrap
from audit_pair_train_test import recount
from audit_pair_views import z
from category_common_v82 import file_hash, instance_seed, load_cloud
from classifier_backbone_r5 import make_encoder
from pointcloud_classifier_v82 import prepare_points
from run_rgb_experiment import check_frozen
from run_storage import write_json


def read(path):
    return json.loads(Path(path).read_text())


def pick(rows,ref,mode):
    if mode=='B':
        rows=[r for r in rows if r['macro']>=ref['macro']-1e-10]
    elif mode=='M':
        rows=[r for r in rows if r['correct']>=ref['correct']]
    else:
        assert mode=='P'
    key=lambda r:((-r['macro'],-r['correct']) if mode=='M' else (-r['correct'],-r['macro']))+(r['passes'],r['name'],r['rgb_weight'])
    return sorted(rows,key=key)[0]


def scores(embeddings,text,indices):
    # Sum separately projected cosine vectors, unlike the training script's mean embedding.
    return sum(10*embeddings[:,i].astype(float)@text.T for i in indices)/len(indices)


def audit(args):
    root=args.output
    for marker in ('PROTOCOL_FROZEN.json','SELECTION_FROZEN.json','TEST_FROZEN.json'):
        check_frozen(root,marker)
    for marker in ('FEATURES_FROZEN.json','EVAL_FEATURES_FROZEN.json'):
        check_frozen(args.reference,marker)
    check_frozen(args.r4,'TEST_FROZEN.json')
    config=read(root/'config.json')
    for path,digest in config['inputs'].items():
        assert file_hash(path)==digest
    for name,digest in config['source'].items():
        assert file_hash(root/'source_snapshot'/name)==digest
    assert read(root/'PREFLIGHT.json')['status']=='passed'
    manifests={s:read(root/f'manifests/{s}.json') for s in ('train','eval','dev')}
    rooms={s:{r['scene'].split('_')[0] for r in m['samples']} for s,m in manifests.items()}
    assert not rooms['train']&rooms['eval'] and not rooms['train']&rooms['dev'] and not rooms['eval']&rooms['dev']
    model=make_encoder(args)
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    replayed=0
    max_feature=0.
    max_rotation_input=0.
    feature_bank={}
    text=np.load(args.reference/'features/text.npy').astype(float)
    groups={'single':(0,), 'sample3':(0,1,2), 'sample7':tuple(range(7)), 'yaw3':(0,7,8), 'sample_yaw9':tuple(range(9))}
    for role,split in (('validation','eval'),('historical_test','dev')):
        path=root/f'features_{split}.npz'
        assert file_hash(path)==read(path.with_suffix('.json'))['sha256']
        embeddings=np.load(path)['features']
        assert embeddings.shape==(len(manifests[split]['samples']),9,512)
        current=None
        source_hashes={r['scene']:r['cloud_sha256'] for r in manifests[split]['sources']}
        for i,row in enumerate(manifests[split]['samples']):
            if row['scene']!=current:
                current=row['scene']
                assert file_hash(row['cloud'])==source_hashes[current]
                xyz,_,ids=load_cloud(row['cloud'])
            obj=xyz[ids==row['bbox_id']]
            prepared=[]
            for seed,angle in [(s,0.) for s in range(42,49)]+[(42,-10.),(42,10.)]:
                p=prepare_points(obj,None,'pointnext',instance_seed(seed,current,row['bbox_id']))
                if angle:
                    original=p.copy()
                    radians=angle*np.pi/180
                    matrix=np.array([[np.cos(radians),np.sin(radians)],[-np.sin(radians),np.cos(radians)]])
                    mathematical=p[:,:2].astype(float)@matrix
                    # Replay the sealed elementwise float32 rounding contract. The
                    # original matrix replay failed although inputs differed <6e-8.
                    x,y=original[:,0],original[:,1]
                    p[:,0]=x*np.cos(radians)-y*np.sin(radians)
                    p[:,1]=x*np.sin(radians)+y*np.cos(radians)
                    difference=float(np.max(np.abs(p[:,:2]-mathematical)))
                    assert difference<1e-7
                    max_rotation_input=max(max_rotation_input,difference)
                prepared.append(p)
            with torch.inference_mode():
                actual=F.normalize(model(torch.from_numpy(np.stack(prepared)).cuda()).float(),dim=-1).cpu().numpy()
            np.testing.assert_allclose(actual,embeddings[i],atol=2e-5,rtol=2e-4)
            max_feature=max(max_feature,float(np.max(np.abs(actual-embeddings[i]))))
            replayed+=9
        old=np.load(args.reference/f'features/shape_{split}.npz')['features']
        np.testing.assert_allclose(embeddings[:,0],old,atol=2e-5,rtol=2e-4)
        feature_bank[role]={name:z(scores(embeddings,text,indices)) for name,indices in groups.items()}
        print(json.dumps({'stage':'feature_replay','role':role,'passes':9*len(embeddings)}),flush=True)
    assert all(torch.equal(v,initial[k]) for k,v in model.state_dict().items())
    selection=read(root/'selection.json')
    grid=read(root/'validation_grid.json')
    def references(role):
        base=np.load(args.r4/f'{role}/S0_v0.npy').astype(float)
        ref=np.load(args.r4/f'{role}/locked_M_v0.npy').astype(float)
        rgb=ref/.6-(2/3)*base
        return base,ref,rgb
    base,ref,rgb=references('validation')
    rebuilt=[]
    for row in grid:
        values=base if row['name']=='S0' else ref if row['name']=='r4_M' else (
            (1-row['rgb_weight'])*feature_bank['validation'][row['name']]+row['rgb_weight']*rgb)
        summary=recount(values,manifests['eval'])[0]
        assert summary['correct']==row['correct'] and abs(summary['class_mean_accuracy']-row['macro'])<1e-12
        rebuilt.append({**row,'correct':summary['correct'],'macro':summary['class_mean_accuracy']})
    reference=next(r for r in rebuilt if r['name']=='r4_M')
    for mode in ('P','B','M'):
        assert pick(rebuilt,reference,mode)==selection['locked'][mode]
    for name in groups:
        assert pick([r for r in rebuilt if r['name']==name],reference,'P')==selection['per_candidate'][name]
    records=read(root/'results.json')['records']
    comparisons={}
    for role,split in (('validation','eval'),('historical_test','dev')):
        base,ref,rgb=references(role)
        hits={}
        for row in [r for r in records if r['role']==role]:
            sel=row['selection']
            expected=base if sel['name']=='S0' else ref if sel['name']=='r4_M' else (
                (1-sel['rgb_weight'])*feature_bank[role][sel['name']]+sel['rgb_weight']*rgb)
            actual=np.load(root/f'{role}/{row["method"]}.npy')
            np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-10)
            np.testing.assert_array_equal(np.argsort(-actual,axis=1,kind='stable')[:,:5],np.argsort(-expected,axis=1,kind='stable')[:,:5])
            summary,hit,_,_=recount(actual,manifests[split])
            assert summary==row['summary']==recount(expected,manifests[split])[0]
            hits[row['method']]=hit[None]
        for mode in ('P','B','M'):
            for reference in ('S0','r4_M'):
                comparisons[f'{role}:locked_{mode}-{reference}']=paired_bootstrap(hits['locked_'+mode],hits[reference],manifests[split])
    diagnosis=read(root/'ROTATION_DIAGNOSIS.json')
    assert diagnosis['original_replay_max_abs']==0 and diagnosis['input_max_abs']<1e-7
    result={'status':'passed_with_rotation_precision_limit','max_rotation_input_abs':max_rotation_input,
            'rotation_precision_limit':'Original matrix-arithmetic rotation replay failed at validation index 88; exact sealed arithmetic replay and all predictions checked. Alternative equivalent arithmetic may change encoder output; internal cause not isolated.','encoder_passes_replayed':replayed,'max_feature_abs':max_feature,
            'all_encoder_state':'unchanged','validation_grid_rows':len(grid),'predictions_recounted':len(records),
            'comparisons':comparisons,'scope':'All frozen feature forwards replayed with sealed input arithmetic; independent mathematical rotation input check; independent average-of-cosines voting, RGB recovery, validation selection, predictions and metrics.'}
    write_json(root/'AUDIT.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('output','reference','r4','checkpoint'):
        parser.add_argument('--'+n,type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    audit(args)
