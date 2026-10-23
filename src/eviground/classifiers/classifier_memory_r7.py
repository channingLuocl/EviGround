"""Train-only exemplar memory: stable nearest-neighbor and full-kernel classification."""
from eviground.paths import source_file
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from eviground.categories.category_common_v82 import file_hash, run_lock
from eviground.experiments.run_rgb_experiment import check_frozen, freeze
from eviground.evaluation.run_storage import write_json
from eviground.experiments.audit_pair_train_test import recount
import eviground.experiments.audit_classifier_balance_r4 as audit_classifier_balance_r4  # Cache only pure category normalization.
from eviground.experiments.audit_pair_views import z

MODALITIES=('shape','rgb','joint')
KERNELS=(('knn20',20,20.),('knn100',100,20.),('kernel10',0,10.),('kernel20',0,20.))
POWERS=(.5,1.)
WEIGHTS=(0.,.2,.4,.6,.8,1.)


def read(path):
    return json.loads(Path(path).read_text())


def unit(x):
    x=np.asarray(x,dtype=np.float64)
    norms=np.linalg.norm(x,axis=-1,keepdims=True)
    assert np.isfinite(x).all() and (norms>0).all()
    return x/norms


def embedding(shape,rgb,modality):
    if modality=='shape':
        return unit(shape)
    if modality=='rgb':
        return unit(rgb)
    assert modality=='joint'
    return unit(np.concatenate((unit(shape),unit(rgb)),axis=-1))


def kernel_scores(memory,labels,queries,classes,k,beta):
    """No query labels accepted; stable ties use memory row index."""
    assert memory.shape[1]==queries.shape[1] and len(memory)==len(labels)
    assert len(memory)>0 and np.min(labels)>=0 and np.max(labels)<classes
    result=[]
    for query in np.array_split(queries,max(1,int(np.ceil(len(queries)/128)))):
        similarity=np.clip(query@memory.T,-1.,1.)
        if k:
            order=np.argsort(-similarity,axis=1,kind='stable')[:,:min(k,len(memory))]
            values=np.exp(beta*(np.take_along_axis(similarity,order,axis=1)-1.))
            scores=np.zeros((len(query),classes),dtype=np.float64)
            np.add.at(scores,(np.arange(len(query))[:,None],labels[order]),values)
        else:
            values=np.exp(beta*(similarity-1.))
            # A class aggregation does not learn from queries or their labels.
            indicator=np.eye(classes,dtype=np.float64)[labels]
            scores=values@indicator
        result.append(scores)
    return np.concatenate(result)


def choose(rows,reference,mode):
    eligible=[r for r in rows if (r['macro']>=reference['macro']-1e-10 if mode=='P' else r['correct']>=reference['correct'])]
    def key(r):
        metrics=(-r['correct'],-r['macro']) if mode=='P' else (-r['macro'],-r['correct'])
        return (*metrics,r['memory_bytes'],r['name'],r['base'],r['weight'])
    return min(eligible,key=key)


def load(args,split):
    manifest=read(args.reference/f'manifests/{split}.json')
    shape=np.load(args.reference/f'features/shape_{split}.npz')['features'].astype(float)
    rgbpath=args.reference/('features/rgb_dev.npz' if split=='dev' else f'features/rgb_{split}/features.npz')
    with np.load(rgbpath) as data:
        assert data['valid'].all()
        rgb=data['features'].astype(float)
    assert len(shape)==len(rgb)==len(manifest['samples'])
    return manifest,shape,rgb


def setup(args):
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    manifests={s:read(args.reference/f'manifests/{s}.json') for s in ('train','eval','dev')}
    assert [len(manifests[s]['samples']) for s in manifests]==[28730,404,3413]
    rooms={s:{r['scene'].split('_')[0] for r in m['samples']} for s,m in manifests.items()}
    assert not rooms['train']&rooms['eval'] and not rooms['train']&rooms['dev'] and not rooms['eval']&rooms['dev']
    for marker in ('FEATURES_FROZEN.json','EVAL_FEATURES_FROZEN.json'):
        check_frozen(args.reference,marker)
    check_frozen(args.r4,'TEST_FROZEN.json')
    sources=('classifier_memory_r7.py','audit_pair_train_test.py','audit_pair_views.py',
             'audit_classifier_balance_r4.py','category_common_v82.py','run_rgb_experiment.py','run_storage.py')
    inputs=[args.design,args.reference/'features/text.npy',args.r4/'validation/locked_M_v0.npy']
    inputs += [args.reference/f'manifests/{s}.json' for s in manifests]
    inputs += [args.reference/f'features/{n}' for n in ('shape_train.npz','shape_eval.npz','rgb_train/features.npz','rgb_eval/features.npz')]
    config={'version':'train-exemplar-memory-r7','modalities':MODALITIES,'kernels':KERNELS,'powers':POWERS,'weights':WEIGHTS,
            'sources':{n:file_hash(source_file(n)) for n in sources},
            'inputs':{str(p):file_hash(p) for p in inputs},'objects':{'train':28730,'validation':404,'historical_test':3413},
            'dtype':'float64','tie':'stable original training index','random_seeds':None,
            'limitation':'historically revealed evaluation; original pretraining scene exposure unknown'}
    config=json.loads(json.dumps(config))
    if (out/'config.json').exists():
        assert read(out/'config.json')==config
    else:
        write_json(out/'config.json',config)
        for folder in ('source_snapshot','manifests'):
            (out/folder).mkdir()
        for n in sources:
            shutil.copyfile(source_file(n),out/'source_snapshot'/n)
        shutil.copyfile(args.design,out/'source_snapshot/design.md')
        for split in manifests:
            shutil.copyfile(args.reference/f'manifests/{split}.json',out/f'manifests/{split}.json')
        freeze(out,'PROTOCOL_FROZEN.json',['config.json']+[
            str(p.relative_to(out)) for d in ('source_snapshot','manifests') for p in (out/d).iterdir()])
    check_frozen(out,'PROTOCOL_FROZEN.json')
    return manifests


def candidates(args,split):
    train,s,r=load(args,'train')
    manifest,shape,rgb=load(args,split)
    labels=np.array([v['label'] for v in train['samples']],dtype=np.int64)
    assert (labels>=0).all()
    counts=np.bincount(labels,minlength=len(train['vocabulary']))
    results={}
    metadata={}
    for modality in MODALITIES:
        memory=embedding(s,r,modality)
        queries=embedding(shape,rgb,modality)
        for kernel,k,beta in KERNELS:
            started=time.time()
            raw=kernel_scores(memory,labels,queries,len(counts),k,beta)
            for power in POWERS:
                name=f'{modality}_{kernel}_p{power:g}'
                results[name]=raw/(counts+1.)**power
                metadata[name]={'modality':modality,'kernel':kernel,'k':k,'beta':beta,'power':power,
                                'memory_bytes':memory.nbytes+labels.nbytes,'objects':len(memory)}
            print(json.dumps({'stage':'retrieval','split':split,'modality':modality,'kernel':kernel,
                              'seconds':time.time()-started}),flush=True)
    return manifest,shape,results,metadata


def select(args):
    setup(args)
    assert not (args.output/'SELECTION_FROZEN.json').exists()
    manifest,shape,raw,metadata=candidates(args,'eval')
    text=np.load(args.reference/'features/text.npy').astype(float)
    bases={'S0':z(10*shape@text.T),'r4_M':np.load(args.r4/'validation/locked_M_v0.npy').astype(float)}
    directory=args.output/'validation_raw'
    directory.mkdir()
    rows=[]
    for name,v in raw.items():
        np.save(directory/f'{name}.npy',v)
        for base,b in bases.items():
            for weight in WEIGHTS:
                summary=recount((1-weight)*b+weight*z(v),manifest)[0]
                rows.append({'name':name,'base':base,'weight':weight,'memory_bytes':metadata[name]['memory_bytes'],
                             'correct':summary['correct'],'macro':summary['class_mean_accuracy']})
    for name,b in bases.items():
        summary=recount(b,manifest)[0]
        rows.append({'name':name,'base':name,'weight':0.,'memory_bytes':0,'correct':summary['correct'],'macro':summary['class_mean_accuracy']})
    reference=next(r for r in rows if r['name']=='r4_M')
    baseline=next(r for r in rows if r['name']=='S0')
    selection={'locked':{m:choose(rows,reference,m) for m in ('P','M')},
               'per_candidate':{name:choose([r for r in rows if r['name']==name],baseline,'P') for name in raw}}
    write_json(args.output/'validation_grid.json',rows)
    write_json(args.output/'selection.json',selection)
    write_json(args.output/'candidates.json',metadata)
    freeze(args.output,'SELECTION_FROZEN.json',['config.json','PROTOCOL_FROZEN.json','validation_grid.json','selection.json','candidates.json']+[
        str(p.relative_to(args.output)) for p in directory.iterdir()])
    print(json.dumps({'stage':'selection_frozen','selection':selection}),flush=True)


def evaluate(args):
    setup(args)
    check_frozen(args.output,'SELECTION_FROZEN.json')
    assert not (args.output/'TEST_FROZEN.json').exists()
    selection=read(args.output/'selection.json')
    text=np.load(args.reference/'features/text.npy').astype(float)
    records=[]
    for role,split in (('validation','eval'),('historical_test','dev')):
        manifest,shape,raw,metadata=candidates(args,split)
        assert metadata==read(args.output/'candidates.json')
        folder=args.output/role
        folder.mkdir()
        bases={'S0':z(10*shape@text.T),'r4_M':np.load(args.r4/f'{role}/locked_M_v0.npy').astype(float)}
        for name,v in raw.items():
            np.save(folder/f'{name}_raw.npy',v)
            if split=='eval':
                np.testing.assert_array_equal(v,np.load(args.output/f'validation_raw/{name}.npy'))
        chosen={**selection['per_candidate'],**{'locked_'+m:r for m,r in selection['locked'].items()}}
        chosen.update({name:{'name':name,'base':name,'weight':0.} for name in bases})
        for name,sel in chosen.items():
            values=bases[sel['name']] if sel['name'] in bases else (1-sel['weight'])*bases[sel['base']]+sel['weight']*z(raw[sel['name']])
            np.save(folder/f'{name}.npy',values)
            records.append({'role':role,'method':name,'selection':sel,'summary':recount(values,manifest)[0]})
        print(json.dumps({'stage':role,'locked':[r for r in records if r['role']==role and r['method'].startswith('locked_')]}),flush=True)
    write_json(args.output/'results.json',{'records':records})
    freeze(args.output,'TEST_FROZEN.json',['results.json','SELECTION_FROZEN.json']+[
        str(p.relative_to(args.output)) for d in ('validation','historical_test') for p in (args.output/d).iterdir()])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('select','evaluate'))
    for n in ('reference','r4','output','design'):
        parser.add_argument('--'+n,type=Path,required=True)
    args=parser.parse_args()
    with run_lock(args.output):
        (select if args.stage=='select' else evaluate)(args)
