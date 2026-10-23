"""Train on official training objects, select on revealed 404, then score historical 3413."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.nn import functional as F

from category_common_v82 import file_hash, run_lock
from run_rgb_experiment import check_frozen, freeze
from run_storage import write_json
from shape_scale import correctness, score
from train_pair_views import (SEEDS, MODES, SETTINGS, Gate, fit_head, fit_gate,
                             gate_features, prior, standardize)


METHODS = ('S0', 'A', 'B', 'C', 'D', 'E')


def read_json(path):
    return json.loads(path.read_text())


def load_split(reference, split, device):
    manifest = read_json(reference/f'manifests/{split}.json')
    shape = torch.from_numpy(np.load(reference/f'features/shape_{split}.npz')['features']).to(device)
    rgb_file = {'train':'rgb_train/features.npz', 'eval':'rgb_eval/features.npz', 'dev':'rgb_dev.npz'}[split]
    archive = np.load(reference/f'features/{rgb_file}')
    assert archive['valid'].all()
    rgb = torch.from_numpy(archive['features']).to(device)
    assert len(shape) == len(rgb) == len(manifest['samples'])
    return manifest, shape, rgb


def choose_validation(base, candidates, manifest, priors, settings):
    """Same setting across seeds; no historical test scores accepted here."""
    baseline = score(base.cpu().numpy(), manifest)['class_mean_accuracy']
    grid = []
    for kind, tau in settings:
        adjusted = [standardize(a-tau*priors.log()) for a in candidates[kind]]
        for weight in [i/10 for i in range(11)]:
            measured = [score(((1-weight)*base+weight*a).cpu().numpy(), manifest) for a in adjusted]
            grid.append({'kind':kind, 'tau':tau, 'weight':weight,
                         'total_correct':sum(v['correct'] for v in measured),
                         'mean_macro':float(np.mean([v['class_mean_accuracy'] for v in measured]))})
    eligible = [r for r in grid if r['mean_macro'] >= baseline-1e-7]
    selected = min(eligible, key=lambda r:(-r['total_correct'], -r['mean_macro'],
                                         r['kind'] != 'ce', r['tau'], r['weight']))
    return selected, grid


def validate_sources(args):
    reference, r1, output = args.reference, args.r1, args.output
    manifests = {s:read_json(reference/f'manifests/{s}.json') for s in ('train','eval','dev')}
    assert all(m['vocabulary'] == manifests['train']['vocabulary'] for m in manifests.values())
    assert len(manifests['train']['vocabulary']) == 197
    rooms = {s:{r['scene'].split('_')[0] for r in m['samples']} for s,m in manifests.items()}
    for a,b in (('train','eval'),('train','dev'),('eval','dev')):
        assert not rooms[a] & rooms[b], (a,b)
    assert [len(manifests[s]['samples']) for s in ('train','eval','dev')] == [28730,404,3413]
    for marker in ('EXPERTS_COMPLETE.json','RESULTS_COMPLETE.json'):
        seal = read_json(r1/'experiment'/marker)
        for name,digest in seal['files'].items():
            assert file_hash(r1/'experiment'/name) == digest, name
    names = ('run_pair_train_test.py','train_pair_views.py','cache_pair_evaluation.py',
             'rgb_classifier.py','shape_scale.py','category_common_v82.py','run_rgb_experiment.py','run_storage.py')
    sources = {n:file_hash(Path(__file__).parent/n) for n in names}
    inputs = {str(p):file_hash(p) for p in [args.design, r1/'experiment/config.json',
        r1/'experiment/EXPERTS_COMPLETE.json', r1/'experiment/RESULTS_COMPLETE.json',
        r1/'views/COMPLETE.json', r1/'views/features.npz']+
        [reference/f'manifests/{s}.json' for s in manifests]+
        [reference/f'features/{n}' for n in ('text.npy','shape_train.npz','rgb_train/features.npz',
                                            'shape_eval.npz','rgb_eval/features.npz')]}
    config = {'version':'train-validation-historical-test-r2','seeds':list(SEEDS), 'epochs':40,
              'roles':{'train':28730,'validation_revealed':404,'historical_test_previously_used':3413},
              'sources':sources,'inputs':inputs,'primary':'E with validation fallback B',
              'gate_parameters':289,'backbones_frozen':True,
              'limitation':'historically exposed evaluation sets; frozen 3D pretraining scope unknown'}
    output.mkdir(parents=True,exist_ok=True)
    if (output/'config.json').exists():
        assert read_json(output/'config.json') == config
    else:
        write_json(output/'config.json',config)
        snapshot = output/'source_snapshot'
        snapshot.mkdir()
        for n in names:
            shutil.copyfile(Path(__file__).parent/n,snapshot/n)
        shutil.copyfile(args.design,snapshot/'design.md')
    return manifests


def views_at(path, expected_rgb, device):
    seal = read_json(path/'COMPLETE.json')
    assert file_hash(path/'features.npz') == seal['features_sha256']
    views = torch.from_numpy(np.load(path/'features.npz')['features']).to(device)
    torch.testing.assert_close(F.normalize(views.mean(1),dim=-1), expected_rgb, atol=1e-5, rtol=0)
    return views


def head_logits(path, values):
    head = torch.load(path, map_location=values.device,weights_only=True)
    return F.linear(values,head['weight'],head['bias'])


def predict(args, split, manifest, shape, rgb, views, text, selected, priors):
    base = standardize(10*shape@text.T)
    folder = args.output/split
    folder.mkdir(exist_ok=True)
    arrays, rows = {}, []
    for seed in SEEDS:
        values = {'S0':base}
        for method in ('A','B'):
            s = selected[method]
            raw = head_logits(args.output/f'models/seed{seed}/{s["kind"]}.pt',rgb)-s['tau']*priors.log()
            values[method] = (1-s['weight'])*base+s['weight']*standardize(raw)
        s = selected['B']
        path = args.output/f'models/seed{seed}/{s["kind"]}.pt'
        raw = head_logits(path,rgb)-s['tau']*priors.log()
        view = head_logits(path,views)-s['tau']*priors.log()
        for method,mode in zip(('C','D','E'),MODES):
            features,different = gate_features(base,raw,view,mode)
            record = torch.load(args.output/f'models/seed{seed}/{method}.pt',map_location=shape.device,weights_only=True)
            model = Gate(record['state_dict']['mean'],record['state_dict']['std'],record['weight']).to(shape.device)
            model.load_state_dict(record['state_dict'],strict=True)
            with torch.inference_mode():
                w = model(features,different)
            values[method] = (1-w[:,None])*base+w[:,None]*standardize(raw)
            np.save(folder/f'{method}_seed{seed}_weights.npy',w.cpu().numpy())
        base_hit = correctness(base.cpu().numpy(),manifest)[0]
        for method,scores in values.items():
            name = f'{method}_seed{seed}'
            arrays[name] = scores.cpu().numpy()
            np.save(folder/f'{name}.npy',arrays[name])
            hit = correctness(arrays[name],manifest)[0]
            rows.append({'method':method,'seed':seed,'summary':score(arrays[name],manifest),
                         'corrected':int((hit & ~base_hit).sum()),'broken':int((~hit & base_hit).sum())})
    write_json(folder/'results.json',{'role':split,'records':rows})
    return rows


def train_select(args, manifests):
    device = args.device
    _,shape,rgb = load_split(args.reference,'train',device)
    validation,vshape,vrgb = load_split(args.reference,'eval',device)
    text = torch.from_numpy(np.load(args.reference/'features/text.npy')).to(device)
    y = torch.tensor([s['label'] for s in manifests['train']['samples']],device=device)
    assert (y>=0).all()
    priors = prior(y,len(text))
    vbase = standardize(10*vshape@text.T)
    candidates = {k:[] for k in ('ce','balanced')}
    for seed in SEEDS:
        for kind in candidates:
            path = args.output/f'models/seed{seed}/{kind}.pt'
            w,b = fit_head(rgb,y,text,kind,seed,path)
            candidates[kind].append(F.linear(vrgb,w,b))
    a,agrid = choose_validation(vbase,candidates,validation,priors,[('ce',0.)])
    b,bgrid = choose_validation(vbase,candidates,validation,priors,SETTINGS)
    selected = {'A':a,'B':b,'A_grid':agrid,'B_grid':bgrid}
    write_json(args.output/'fusion_selection.json',selected)
    base = standardize(10*shape@text.T)
    views = views_at(args.r1/'views',rgb,device)
    folds = np.array(read_json(args.r1/'experiment/folds.json')['outer'])
    for seed in SEEDS:
        raw = torch.zeros((len(y),197),device=device)
        view = torch.zeros((len(y),4,197),device=device)
        for fold in range(3):
            mask = folds==fold
            expert = args.r1/f'experiment/outer{fold}/seed{seed}/{b["kind"]}/outer.pt'
            raw[mask] = head_logits(expert,rgb[mask])
            view[mask] = head_logits(expert,views[mask])
        raw -= b['tau']*priors.log()
        view -= b['tau']*priors.log()
        for method,mode in zip(('C','D','E'),MODES):
            ft,dt = gate_features(base,raw,view,mode)
            fit_gate(ft,dt,base,standardize(raw),y,b['weight'],seed,args.output/f'models/seed{seed}/{method}.pt')
            print(json.dumps({'stage':'gate','seed':seed,'method':method}),flush=True)
    vviews = views_at(args.output/'validation_views',vrgb,device)
    rows = predict(args,'validation',validation,vshape,vrgb,vviews,text,selected,priors)
    grouped = {m:[r['summary'] for r in rows if r['method']==m] for m in METHODS}
    better = sum(v['correct'] for v in grouped['E']) > sum(v['correct'] for v in grouped['B'])
    macro_ok = np.mean([v['class_mean_accuracy'] for v in grouped['E']]) >= np.mean([v['class_mean_accuracy'] for v in grouped['B']])
    write_json(args.output/'deployment_selection.json',{'primary':'E','selected':'E' if better and macro_ok else 'B',
              'rule':'E strictly more validation True correct and macro no lower than B, otherwise B'})
    files = [str(p.relative_to(args.output)) for pattern in ('models/*/*','validation/*','source_snapshot/*')
             for p in args.output.glob(pattern) if p.is_file()]
    files += ['config.json','fusion_selection.json','deployment_selection.json',
              'validation_views/COMPLETE.json','validation_views/features.npz']
    freeze(args.output,'SELECTION_FROZEN.json',files)
    print(json.dumps({'stage':'selection_frozen','A':a,'B':b,'deployment':read_json(args.output/'deployment_selection.json')}),flush=True)


def test(args, manifests):
    check_frozen(args.output,'SELECTION_FROZEN.json')
    manifest,shape,rgb = load_split(args.reference,'dev',args.device)
    views = views_at(args.output/'historical_test_views',rgb,args.device)
    text = torch.from_numpy(np.load(args.reference/'features/text.npy')).to(args.device)
    y = torch.tensor([s['label'] for s in manifests['train']['samples']],device=args.device)
    rows = predict(args,'historical_test',manifest,shape,rgb,views,text,
                   read_json(args.output/'fusion_selection.json'),prior(y,len(text)))
    files = [str(p.relative_to(args.output)) for p in (args.output/'historical_test').glob('*') if p.is_file()]
    files += ['SELECTION_FROZEN.json','historical_test_views/COMPLETE.json']
    freeze(args.output,'TEST_FROZEN.json',files)
    print(json.dumps({'stage':'historical_test_complete','correct':[r['summary']['correct'] for r in rows]}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('train-select','test'))
    for name in ('reference','r1','output','design'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with run_lock(args.output):
        manifests = validate_sources(args)
        with torch.no_grad() if args.stage == 'test' else torch.enable_grad():
            (train_select if args.stage=='train-select' else test)(args,manifests)
