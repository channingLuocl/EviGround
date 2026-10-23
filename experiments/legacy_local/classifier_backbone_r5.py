"""Adapt the actual PointNeXt backbone under the room-disjoint r5 protocol."""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from category_common_v82 import file_hash, instance_seed, load_cloud, run_lock
from classifier_sweep_r3 import fast_metric
from run_rgb_experiment import check_frozen, freeze
from run_storage import write_json
from train_pair_views import standardize

SEEDS = (42, 43, 44)
SPECS = ('frozen', 'last2', 'last2_aug')
EPOCHS = (4, 8, 12)
WEIGHTS = (0., .2, .4, .6, .8, 1.)
PREFIXES = ('obj_encoder.encoder.encoder.4.', 'obj_encoder.encoder.encoder.5.',
            'obj_encoder.prediction.', 'pc_projection')


def read(path):
    return json.loads(Path(path).read_text())


def make_encoder(args):
    # Torch must be imported before borrowing old Python-only dependencies.
    sys.path.insert(0, str(args.reference/'cuda_ops'))
    sys.path.append('/root/paper/reproductions/SeeGround/.venv-v82/lib/python3.12/site-packages')
    from pointcloud_classifier_v82 import load_pointnext
    return load_pointnext(args.checkpoint)[0].cuda().eval()


def configure(encoder, spec):
    for name, value in encoder.named_parameters():
        value.requires_grad_(spec != 'frozen' and name.startswith(PREFIXES))
    encoder.eval()  # Keep all original BN statistics and original dropout fixed.


def augment(points, generator):
    """Light training-only, Z-up geometry transforms; rebuild height afterwards."""
    xyz = points[..., :3].clone()
    batch, count, _ = xyz.shape
    rand = lambda shape: torch.rand(shape, generator=generator, device=xyz.device)
    xyz *= .9+.2*rand((batch, 1, 3))
    angle = (rand((batch, 1))-.5)*(math.pi/6)
    x, y = xyz[..., 0].clone(), xyz[..., 1].clone()
    xyz[..., 0] = x*angle.cos()-y*angle.sin()
    xyz[..., 1] = x*angle.sin()+y*angle.cos()
    xyz += (torch.randn(xyz.shape, generator=generator, device=xyz.device)*.005).clamp(-.02, .02)
    indices = torch.randint(count, (batch, count), generator=generator, device=xyz.device)
    replace = xyz.gather(1, indices[..., None].expand(-1, -1, 3))
    xyz = torch.where(rand((batch, count, 1)) < .1, replace, xyz)
    xyz -= xyz.mean(1, keepdim=True)
    xyz /= xyz.norm(dim=-1).amax(1)[:, None, None].clamp_min(1e-8)
    return torch.cat((xyz, xyz[..., 2:3]-xyz[..., 2:3].amin(1, keepdim=True)), -1).contiguous()


def head_for(text):
    head = nn.Linear(text.shape[1], len(text), device=text.device)
    with torch.no_grad():
        head.weight.copy_(10*text)
        head.bias.zero_()
    return head


def logits(encoder, head, points):
    return head(F.normalize(encoder(points).float(), dim=-1))


def cache_points(args, split):
    path = args.output/f'points/{split}.npy'
    seal = path.with_suffix('.json')
    manifest = read(args.output/f'manifests/{split}.json')
    if seal.exists():
        assert file_hash(path) == read(seal)['sha256']
        return path
    assert not path.exists(), 'Incomplete cache preserved; inspect before restarting'
    from pointcloud_classifier_v82 import prepare_points
    path.parent.mkdir(exist_ok=True)
    values = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                     shape=(len(manifest['samples']), 2048, 4))
    sources = {r['scene']: r['cloud_sha256'] for r in manifest['sources']}
    current = None
    for i, row in enumerate(manifest['samples']):
        if row['scene'] != current:
            current = row['scene']
            assert file_hash(row['cloud']) == sources[current]
            xyz, _, ids = load_cloud(row['cloud'])
        values[i] = prepare_points(xyz[ids == row['bbox_id']], None, 'pointnext',
                                   instance_seed(42, current, row['bbox_id']))
        if (i+1) % 2000 == 0:
            print(json.dumps({'stage':'cache', 'split':split, 'objects':i+1}), flush=True)
    values.flush()
    del values
    write_json(seal, {'sha256':file_hash(path), 'manifest_sha256':file_hash(args.output/f'manifests/{split}.json'),
                      'objects':len(manifest['samples']), 'dtype':'float32'})
    return path


def setup(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    sources = ('classifier_backbone_r5.py', 'pointcloud_classifier_v82.py', 'category_common_v82.py',
               'classifier_sweep_r3.py', 'train_pair_views.py', 'run_rgb_experiment.py', 'run_storage.py')
    manifests = {s:read(args.reference/f'manifests/{s}.json') for s in ('train','eval','dev')}
    assert [len(manifests[s]['samples']) for s in manifests] == [28730,404,3413]
    rooms = {s:{r['scene'].split('_')[0] for r in m['samples']} for s,m in manifests.items()}
    assert not rooms['train'] & rooms['eval'] and not rooms['train'] & rooms['dev'] and not rooms['eval'] & rooms['dev']
    assert all(m['vocabulary'] == manifests['train']['vocabulary'] for m in manifests.values())
    assert all(r['label'] >= 0 for r in manifests['train']['samples'])
    paths = [args.design,args.checkpoint,args.r4/'SELECTION_FROZEN.json',args.r4/'validation/locked_M_v0.npy']
    paths += [args.reference/f'manifests/{s}.json' for s in manifests]
    paths += [args.reference/f'features/{n}' for n in ('text.npy','shape_train.npz','shape_eval.npz')]
    config = {'version':'pointnext-backbone-r5', 'seeds':SEEDS, 'specs':SPECS, 'epochs':12, 'checkpoints':EPOCHS,
              'batch':32, 'body_lr':1e-5, 'head_lr':.001, 'weight_decay':.01, 'warmup_head_epochs':2,
              'loss':'inverse-sqrt-weighted CE', 'bn':'frozen', 'source':{n:file_hash(Path(__file__).parent/n) for n in sources},
              'inputs':{str(p):file_hash(p) for p in paths}, 'torch':str(torch.__version__),
              'limitation':'historically revealed evaluation; original pretraining exposure unknown'}
    config = json.loads(json.dumps(config))
    if (out/'config.json').exists():
        assert read(out/'config.json') == config
    else:
        write_json(out/'config.json', config)
        (out/'source_snapshot').mkdir()
        (out/'manifests').mkdir()
        for name in sources:
            shutil.copyfile(Path(__file__).parent/name, out/'source_snapshot'/name)
        shutil.copyfile(args.design, out/'source_snapshot/design.md')
        for split in manifests:
            shutil.copyfile(args.reference/f'manifests/{split}.json',out/f'manifests/{split}.json')
        freeze(out,'PROTOCOL_FROZEN.json',['config.json']+[
            str(p.relative_to(out)) for folder in ('source_snapshot','manifests') for p in (out/folder).iterdir()])
    check_frozen(out,'PROTOCOL_FROZEN.json')
    return manifests


def gpu_points(args, split):
    return torch.from_numpy(np.load(cache_points(args,split))).cuda()


def smoke(args):
    setup(args)
    assert not (args.output/'SMOKE.json').exists()
    encoder = make_encoder(args)
    points = gpu_points(args,'train')
    expected = np.load(args.reference/'features/shape_train.npz')['features']
    indices = [0,len(points)//2,len(points)-1]
    with torch.no_grad():
        actual = F.normalize(encoder(points[indices]),dim=-1).cpu().numpy()
    np.testing.assert_allclose(actual,expected[indices],atol=2e-5,rtol=2e-4)
    bridge = float(np.max(np.abs(actual-expected[indices])))
    configure(encoder,'last2')
    initial = {k:v.detach().clone() for k,v in encoder.state_dict().items()}
    text = torch.from_numpy(np.load(args.reference/'features/text.npy')).cuda()
    head = head_for(text)
    optimizer = torch.optim.AdamW([{'params':[p for p in encoder.parameters() if p.requires_grad], 'lr':1e-5},
                                  {'params':head.parameters(),'lr':.001}],weight_decay=.01)
    m = read(args.output/'manifests/train.json')
    x = points[:32]
    y = torch.tensor([r['label'] for r in m['samples'][:32]],device='cuda')
    torch.cuda.synchronize()
    started = time.time()
    losses = []
    for step in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits(encoder,head,x),y)
        loss.backward()
        if step == 0:
            gradients = {n:float(p.grad.norm()) for n,p in encoder.named_parameters() if p.requires_grad}
            assert gradients and all(np.isfinite(list(gradients.values()))) and sum(gradients.values())>0
        optimizer.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    assert losses[-1] < losses[0]
    changed = []
    for name,value in encoder.state_dict().items():
        if not torch.equal(value,initial[name]):
            assert name in gradients, name
            changed.append(name)
    assert any(n.startswith('obj_encoder.encoder.encoder.') for n in changed)
    write_json(args.output/'SMOKE.json',{'status':'passed','bridge_max_abs':bridge,'losses':losses,
              'steps':20,'batch':32,'seconds':time.time()-started,'nonzero_gradients':gradients,
              'changed':changed,'all_other_state_and_bn':'exactly_unchanged'})
    print(json.dumps({'stage':'smoke_passed','bridge':bridge,'first_loss':losses[0],
                      'last_loss':losses[-1],'seconds':time.time()-started}),flush=True)


@torch.inference_mode()
def predict(encoder, head, points):
    return torch.cat([logits(encoder,head,x) for x in points.split(32)])


def selector(rows, ref, mode):
    eligible = [r for r in rows if (r['mean_macro'] >= ref['mean_macro']-1e-7 if mode=='P'
                                   else r['mean_correct'] >= ref['mean_correct'])]
    def key(r):
        metrics = (-r['mean_correct'],-r['mean_macro']) if mode=='P' else (-r['mean_macro'],-r['mean_correct'])
        return (*metrics,r['parameters'],r['name'],r['epoch'],r['base'],r['weight'])
    return min(eligible,key=key)


def make_grid(args, scores, manifest):
    text = torch.from_numpy(np.load(args.reference/'features/text.npy')).cuda()
    shape = torch.from_numpy(np.load(args.reference/'features/shape_eval.npz')['features']).cuda()
    bases = {'S0':standardize(10*shape@text.T),
             'r4_M':torch.from_numpy(np.load(args.r4/'validation/locked_M_v0.npy')).cuda()}
    rows = []
    for (spec,epoch), (values,parameters) in scores.items():
        for base,b in bases.items():
            for w in WEIGHTS:
                metrics = [fast_metric((1-w)*b+w*standardize(v),manifest) for v in values]
                rows.append({'name':spec,'epoch':epoch,'base':base,'weight':w,'parameters':parameters,
                             'correct_by_seed':[v[0] for v in metrics],'mean_correct':float(np.mean([v[0] for v in metrics])),
                             'mean_macro':float(np.mean([v[1] for v in metrics]))})
    for name,b in bases.items():
        correct,macro = fast_metric(b,manifest)
        rows.append({'name':name,'epoch':0,'base':name,'weight':0.,'parameters':0,
                     'correct_by_seed':[correct]*3,'mean_correct':float(correct),'mean_macro':macro})
    ref = next(r for r in rows if r['name']=='r4_M')
    selection = {'locked':{mode:selector(rows,ref,mode) for mode in ('P','M')}}
    selection['per_candidate'] = {spec:selector([r for r in rows if r['name']==spec],
                   next(r for r in rows if r['name']=='S0'),'P') for spec in SPECS}
    return rows,selection


def train_select(args):
    manifests = setup(args)
    assert read(args.output/'SMOKE.json')['status']=='passed'
    assert not (args.output/'SELECTION_FROZEN.json').exists()
    text = torch.from_numpy(np.load(args.reference/'features/text.npy')).cuda()
    shape = torch.from_numpy(np.load(args.reference/'features/shape_train.npz')['features']).cuda()
    valshape = torch.from_numpy(np.load(args.reference/'features/shape_eval.npz')['features']).cuda()
    points,validation = gpu_points(args,'train'),gpu_points(args,'eval')
    y = torch.tensor([r['label'] for r in manifests['train']['samples']],device='cuda')
    weights = (torch.bincount(y,minlength=len(text)).float()+1).rsqrt()
    weights /= weights.mean()
    scores = {}
    folder = args.output/'validation_raw'
    folder.mkdir(exist_ok=True)
    for spec in SPECS:
        for seed in SEEDS:
            torch.manual_seed(seed)
            encoder = make_encoder(args)
            configure(encoder,spec)
            head = head_for(text)
            initial = {n:v.detach().clone() for n,v in encoder.state_dict().items()}
            trainable = [p for p in encoder.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW([{'params':trainable,'lr':1e-5},
                                          {'params':head.parameters(),'lr':.001}],weight_decay=.01)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=12)
            gen = torch.Generator(device='cuda').manual_seed(seed)
            auggen = torch.Generator(device='cuda').manual_seed(seed+1000)
            modeldir = args.output/f'models/{spec}/seed{seed}'
            modeldir.mkdir(parents=True,exist_ok=True)
            assert not list(modeldir.iterdir()), 'Preserve existing training; do not silently restart'
            history = []
            for epoch in range(1,13):
                started = time.time()
                total,seen = 0.,0
                for indices in torch.randperm(len(y),device='cuda',generator=gen).split(32):
                    optimizer.zero_grad(set_to_none=True)
                    if spec=='frozen':
                        embedding = shape[indices]
                    else:
                        x = points[indices]
                        if spec.endswith('aug'):
                            x = augment(x,auggen)
                        with torch.no_grad() if epoch<=2 else nullcontext():
                            embedding = F.normalize(encoder(x),dim=-1)
                    loss = F.cross_entropy(head(F.dropout(embedding,.3,True)),y[indices],weight=weights)
                    assert torch.isfinite(loss)
                    loss.backward()
                    if epoch==3 and spec!='frozen' and seen==0:
                        grad = {n:float(p.grad.norm()) for n,p in encoder.named_parameters() if p.requires_grad}
                        assert sum(grad.values())>0 and np.isfinite(list(grad.values())).all()
                        write_json(modeldir/'gradients.json',grad)
                    optimizer.step()
                    total += float(loss.detach())*len(indices)
                    seen += len(indices)
                schedule.step()
                assert seen==len(y)
                record = {'epoch':epoch,'objects':seen,'loss':total/seen,'seconds':time.time()-started}
                history.append(record)
                write_json(modeldir/'history.json',history)
                print(json.dumps({'stage':'epoch','spec':spec,'seed':seed,**record}),flush=True)
                if epoch in EPOCHS:
                    values = head(valshape) if spec=='frozen' else predict(encoder,head,validation)
                    values = values.detach()
                    np.save(folder/f'{spec}_e{epoch}_seed{seed}.npy',values.cpu().numpy())
                    parameters = sum(p.numel() for p in trainable)+sum(p.numel() for p in head.parameters())
                    scores.setdefault((spec,epoch),([],parameters))[0].append(values)
                    change = {}
                    for n,v in encoder.state_dict().items():
                        if not torch.equal(v,initial[n]):
                            assert n in dict(encoder.named_parameters()) and dict(encoder.named_parameters())[n].requires_grad,n
                            change[n] = float((v-initial[n]).norm())
                    assert bool(change)==(spec!='frozen')
                    torch.save({'encoder':{k:v.detach().cpu() for k,v in encoder.state_dict().items()},
                                'head':{k:v.detach().cpu() for k,v in head.state_dict().items()},
                                'epoch':epoch,'spec':spec,'seed':seed,'parameters':parameters,'changed':change,
                                'objects':len(y),'config_sha256':file_hash(args.output/'config.json')},modeldir/f'e{epoch}.pt')
            del encoder,head,initial,optimizer
    rows,selection = make_grid(args,scores,manifests['eval'])
    write_json(args.output/'validation_grid.json',rows)
    write_json(args.output/'selection.json',selection)
    files = ['config.json','SMOKE.json','validation_grid.json','selection.json','PROTOCOL_FROZEN.json']
    files += [str(p.relative_to(args.output)) for d in ('models','validation_raw') for p in (args.output/d).rglob('*') if p.is_file()]
    freeze(args.output,'SELECTION_FROZEN.json',files)
    print(json.dumps({'stage':'selection_frozen','selection':selection}),flush=True)


@torch.inference_mode()
def evaluate(args):
    manifests = setup(args)
    check_frozen(args.output,'SELECTION_FROZEN.json')
    assert not (args.output/'TEST_FROZEN.json').exists()
    from shape_scale import score
    selection = read(args.output/'selection.json')
    text = torch.from_numpy(np.load(args.reference/'features/text.npy')).cuda()
    records = []
    for role,split in (('validation','eval'),('historical_test','dev')):
        points = gpu_points(args,split)
        shape = torch.from_numpy(np.load(args.reference/f'features/shape_{split}.npz')['features']).cuda()
        bases = {'S0':standardize(10*shape@text.T),
                 'r4_M':torch.from_numpy(np.load(args.r4/f'{role}/locked_M_v0.npy')).cuda()}
        directory = args.output/role
        directory.mkdir()
        raw = {}
        for spec in SPECS:
            for epoch in EPOCHS:
                raw[spec,epoch] = []
                for seed in SEEDS:
                    path = args.output/f'models/{spec}/seed{seed}/e{epoch}.pt'
                    state = torch.load(path,map_location='cpu',weights_only=True)
                    assert state['config_sha256']==file_hash(args.output/'config.json')
                    encoder = make_encoder(args)
                    encoder.load_state_dict(state['encoder'],strict=True)
                    head = head_for(text)
                    head.load_state_dict(state['head'],strict=True)
                    values = head(shape) if spec=='frozen' else predict(encoder,head,points)
                    if split=='eval':
                        np.testing.assert_allclose(values.cpu().numpy(),np.load(args.output/f'validation_raw/{spec}_e{epoch}_seed{seed}.npy'),atol=1e-5,rtol=1e-5)
                    np.save(directory/f'{spec}_e{epoch}_seed{seed}_raw.npy',values.cpu().numpy())
                    raw[spec,epoch].append(values)
                    del encoder,head
        chosen = {**selection['per_candidate'],**{'locked_'+m:r for m,r in selection['locked'].items()}}
        chosen.update({name:{'name':name,'base':name,'epoch':0,'weight':0.} for name in bases})
        for name,sel in chosen.items():
            values = [bases[sel['name']]]*3 if sel['name'] in bases else [
                (1-sel['weight'])*bases[sel['base']]+sel['weight']*standardize(v) for v in raw[sel['name'],sel['epoch']]]
            for seed,v in zip(SEEDS,values):
                saved = v.cpu().numpy()
                np.save(directory/f'{name}_seed{seed}.npy',saved)
                records.append({'role':role,'method':name,'seed':seed,'selection':sel,'summary':score(saved,manifests[split])})
        print(json.dumps({'stage':role,'records':[r for r in records if r['role']==role and r['method'].startswith('locked_')]}),flush=True)
    write_json(args.output/'results.json',{'records':records})
    freeze(args.output,'TEST_FROZEN.json',['results.json','SELECTION_FROZEN.json']+[
        str(p.relative_to(args.output)) for d in ('validation','historical_test') for p in (args.output/d).iterdir()])


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('smoke','train-select','evaluate'))
    for name in ('reference','r4','output','checkpoint','design'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with run_lock(args.output):
        globals()[args.stage.replace('-','_')](args)
