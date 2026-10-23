"""Recount revised-protocol selection, linear predictions and room-paired CI."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from seeground.experiments.audit_shape_scale import recount, normalized_category
from seeground.categories.category_common_v82 import file_hash
from seeground.classifiers.category_v10.data import assert_disjoint
from seeground.evaluation.run_storage import write_json


def z(scores):
    return (scores-scores.mean(1, keepdims=True))/np.maximum(scores.std(1, keepdims=True), 1e-6)


def fused(shape, image, valid, weight):
    scores = (1-weight)*z(shape)+weight*z(image)
    scores[~valid] = z(shape)[~valid]
    return scores


def audit(output, stage):
    for marker in ('PROTOCOL_FROZEN.json', 'FEATURES_FROZEN.json', 'MODEL_SELECTION_FROZEN.json'):
        for name, digest in json.loads((output / marker).read_text())['files'].items():
            assert file_hash(output / name) == digest, name
    if stage == 'full':
        for marker in ('EVAL_FEATURES_FROZEN.json', 'RESULTS_FROZEN.json'):
            for name, digest in json.loads((output / marker).read_text())['files'].items():
                assert file_hash(output / name) == digest, name
    cfg = json.loads((output / 'training_config.json').read_text())
    for name, digest in cfg['code_sha256'].items():
        assert file_hash(output / 'source_snapshot' / name) == digest, name
    assert file_hash(output / 'design.md') == cfg['design_sha256']
    manifests = {s: json.loads((output / f'manifests/{s}.json').read_text()) for s in
                 (('train', 'dev', 'eval') if stage == 'full' else ('train', 'dev'))}
    for first, second in (('train', 'dev'), *([('train', 'eval'), ('dev', 'eval')] if stage == 'full' else [])):
        a, b = manifests[first], manifests[second]
        assert_disjoint(a['scenes'], b['scenes'])
        assert not ({r['cloud_sha256'] for r in a['sources']} & {r['cloud_sha256'] for r in b['sources']})
    assert len(manifests['train']['samples']) == 28730
    assert len(manifests['dev']['samples']) == 3413
    protocol = json.loads((output / 'protocol.json').read_text())
    assert len(protocol['holdout_rooms']) == 12 and len(protocol['holdout_scenes']) == 22
    assert_disjoint(manifests['train']['scenes'] + manifests['dev']['scenes'], protocol['holdout_scenes'])
    if stage == 'full':
        assert manifests['eval']['scenes'] == protocol['holdout_scenes']
        seal = json.loads((output / 'HOLDOUT_FROZEN.json').read_text())
        assert file_hash(output / 'manifests/eval.json') == seal['manifest_sha256']
        from seeground.experiments.repair_scan_domain_annotations import unique_groups
        for record in json.loads((output / 'annotation_repairs.json').read_text()):
            scene = record['scene']
            original = output / 'annotation_evidence' / (scene+'.original.json')
            repaired = output / 'annotation_staging' / scene / (scene+'.aggregation.json')
            assert file_hash(original) == record['original_annotation_sha256']
            assert file_hash(repaired) == record['repaired_annotation_sha256']
            kept, duplicates = unique_groups(json.loads(original.read_text())['segGroups'])
            assert duplicates == record['duplicates']
            assert kept == json.loads(repaired.read_text())['segGroups']
            actual = [r for r in manifests['eval']['samples'] if r['scene']==scene]
            expected = [g for g in kept if normalized_category(g['label']) not in ('wall','floor','ceiling')]
            assert [(r['bbox_id'],r['category']) for r in actual] == [(g['objectId'],g['label']) for g in expected]
    selection = json.loads((output / 'selection.json').read_text())
    grids = json.loads((output / 'dev_grid.json').read_text())
    text = np.load(output / 'features/text.npy')
    # Independently apply W x + b from each stored head, not the training forward function.
    for split in ('dev', 'eval') if stage == 'full' else ('dev',):
        with np.load(output / f'features/shape_{split}.npz') as a:
            shape = a['features']
        rgb_path = output / ('features/rgb_dev.npz' if split == 'dev' else 'features/rgb_eval/features.npz')
        with np.load(rgb_path) as a:
            image, valid = a['features'], a['valid']
        with np.load(output / split / 'components.npz') as p:
            np.testing.assert_allclose(p['base'], 10*shape@text.T, atol=1e-5, rtol=1e-5)
            assert np.array_equal(p['valid'], valid)
            for mode, values in (('shape', shape), ('rgb', image)):
                for index, seed in enumerate((42, 43, 44)):
                    path = output / f'models/seed{seed}/{mode}/head.pt'
                    head = torch.load(path, map_location='cpu', weights_only=True)
                    history = json.loads(path.with_name('history.json').read_text())
                    best = sorted(history, key=lambda r: (-r['dev']['correct'], r['dev']['known_CE'], r['epoch']))[0]
                    assert len(history) == 40 and head['epoch'] == best['epoch']
                    assert head['config_sha256'] == file_hash(output / 'training_config.json')
                    state = head['state_dict']
                    expected = values@state['weight'].numpy().T+state['bias'].numpy()
                    if mode == 'rgb': expected[~valid] = p['base'][~valid]
                    np.testing.assert_allclose(p[mode][index], expected, atol=2e-5, rtol=1e-4)
                    if split == 'dev':
                        metrics, _ = recount(p[mode][index], manifests[split])
                        assert metrics['correct'] == best['dev']['correct']
            if split == 'dev':
                labels = np.array([r['label'] for r in manifests[split]['samples']])
                candidates = []
                for method, bases in (('S3', [p['base']]*3), ('S4', p['shape'])):
                    expected_weights = [i/10 for i in range(11)]
                    assert [r['weight'] for r in grids[method]] == expected_weights
                    for row in grids[method]:
                        counts = [recount(fused(a, b, valid, row['weight']), manifests['dev'])[0]['correct'] for a,b in zip(bases,p['rgb'])]
                        assert counts == row['correct_by_seed'] and sum(counts) == row['total_correct']
                        assert abs(sum(counts)/(3*len(labels))-row['mean_accuracy']) < 1e-12
                    best = min(grids[method], key=lambda r: (-r['total_correct'],r['weight']))
                    assert selection['fusion'][method] == best
                totals = {'S0':3*recount(p['base'], manifests['dev'])[0]['correct']}
                for method,key in (('S1','shape'),('S2','rgb')):
                    totals[method] = sum(recount(s,manifests['dev'])[0]['correct'] for s in p[key])
                totals.update({m:selection['fusion'][m]['total_correct'] for m in ('S3','S4')})
                primary = min(totals, key=lambda m:(-totals[m],cfg['primary_tie_order'].index(m)))
                assert primary == selection['primary_method']
            results = json.loads((output / split / 'results.json').read_text())
            base_metrics, base_hit = recount(p['base'], manifests[split])
            hit_lists = {}
            for row in results['records']:
                name = row['method']
                if name == 'S0': expected = p['base']
                else:
                    group, seed_name = name.split('_')
                    index = int(seed_name[4:])-42
                    if group in ('S1','S2'): expected = p['shape' if group=='S1' else 'rgb'][index]
                    else:
                        base = p['base'] if group=='S3' else p['shape'][index]
                        expected = fused(base, p['rgb'][index],valid, selection['fusion'][group]['weight'])
                logits = np.load(output / split / f'{name}.npy')
                np.testing.assert_array_equal(logits,expected)
                summary, hit = recount(logits, manifests[split])
                for key,value in summary.items(): assert abs(value-row['summary'][key]) < 1e-12
                assert int((~base_hit & hit).sum()) == row['corrected']
                assert int((base_hit & ~hit).sum()) == row['broken']
                hit_lists[name] = hit.astype(float)
            if split == 'eval':
                rooms = np.array([r['scene'].split('_')[0] for r in manifests[split]['samples']])
                unique = sorted(set(rooms)); assert len(unique)==12
                counts=np.array([(rooms==room).sum() for room in unique])
                rng=np.random.RandomState(20261019)
                weights=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=10000)
                intervals={}
                for method in ('S1','S2','S3','S4'):
                    hits=np.mean([hit_lists[f'{method}_seed{s}'] for s in (42,43,44)],axis=0)
                    delta=hits-base_hit
                    sums=np.array([delta[rooms==room].sum() for room in unique])
                    dist=(weights@sums)/(weights@counts)
                    intervals[method]={'accuracy_delta_pp':float(delta.mean()*100),
                        'paired_room_95CI_pp':(np.quantile(dist,[.025,.975])*100).tolist()}
    report={'status':'passed','stage':stage,'models':6,'epochs_each':40,'auditor_sha256':file_hash(Path(__file__)),
            'primary_method_locked':selection['primary_method'],'checks':['frozen sources and inputs','room and content separation',
                'checkpoint criterion','independent linear forwards','all dev fusion candidates and primary method',
                'all logits, metrics and paired corrections'], 'holdout_rooms':12,
            'limitation':'previously inspected research dev; small reserved validation cohort; backbone exposure uncertified'}
    if stage=='full': report['intervals']=intervals
    write_json(output / ('audit.json' if stage=='full' else 'dev_audit.json'), report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('dev','full'),required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    audit(args.output,args.stage)
