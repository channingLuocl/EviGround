"""Audit r11 provenance, real image-tail updates, predictions and locked selection."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from eviground.experiments.audit_classifier_balance_r4 import paired_bootstrap
from eviground.experiments.audit_classifier_residual_r9 import pick
from eviground.experiments.audit_pair_train_test import recount
from eviground.experiments.audit_pair_views import z
from eviground.categories.category_common_v82 import file_hash
from eviground.classifiers.classifier_clip_tail_r11 import (REFERENCES, SEEDS, SPECS, EPOCHS, bases, image_tensor,
    load_model, read, render_scene_rows, rows_by_scene, token_bank)
from eviground.experiments.run_rgb_experiment import check_frozen
from eviground.evaluation.run_storage import write_json


@torch.inference_mode()
def replay(visual, state, bank, reference, frozen):
    """Use the original visual pool and an independent head projection."""
    visual.transformer.resblocks[-1].load_state_dict({k[6:]: v for k, v in state.items() if k.startswith('block.')})
    visual.ln_post.load_state_dict({k[5:]: v for k, v in state.items() if k.startswith('norm.')})
    visual.proj.copy_(state['projection'])
    features = []
    for start in range(0, len(bank), 32):
        x = torch.from_numpy(np.array(bank[start:start+32], copy=True)).cuda()
        count, views, length, width = x.shape
        x = x.reshape(count*views, length, width)
        if not visual.transformer.batch_first:
            x = x.transpose(0, 1).contiguous()
        x = visual.transformer.resblocks[-1](x)
        if not visual.transformer.batch_first:
            x = x.transpose(0, 1)
        pooled, _ = visual._pool(x)
        embedded = F.normalize(pooled @ visual.proj, dim=-1)
        merged = F.normalize(embedded.reshape(count, views, -1).mean(1), dim=-1)
        features.append(merged)
    features = torch.cat(features)
    difference = 0.
    if frozen:
        torch.testing.assert_close(features, reference, atol=1e-5, rtol=1e-5)
        difference = float((features-reference).abs().max())
        # The control's sealed inference uses the original per-eight-object cache.
        features = reference
    weight, bias = state['head.weight'].cuda(), state['head.bias'].cuda()
    actual = F.linear(features, weight, bias).cpu().numpy() if frozen else np.concatenate([
        F.linear(batch, weight, bias).cpu().numpy() for batch in features.split(32)])
    independent = features.cpu().numpy().astype(float) @ state['head.weight'].numpy().astype(float).T + state['head.bias'].numpy().astype(float)
    np.testing.assert_allclose(independent, actual, atol=2e-5, rtol=2e-5)
    return actual, independent, difference


def audit(args):
    root = args.output
    for marker in ('PROTOCOL_FROZEN.json', 'SELECTION_FROZEN.json', 'TEST_FROZEN.json'):
        check_frozen(root, marker)
    config = read(root / 'config.json')
    for name, digest in config['inputs'].items():
        assert file_hash(name) == digest
    for name, digest in config['source'].items():
        assert file_hash(root / 'source_snapshot' / name) == digest
    assert read(root / 'PREFLIGHT.json')['status'] == 'passed'
    manifests = {s: read(root / f'manifests/{s}.json') for s in ('train', 'eval', 'dev')}
    assert [len(m['samples']) for m in manifests.values()] == [28730, 404, 3413]
    rooms = {s: {row['scene'].split('_')[0] for row in m['samples']} for s, m in manifests.items()}
    assert not rooms['train'] & rooms['eval'] and not rooms['train'] & rooms['dev'] and not rooms['eval'] & rooms['dev']
    original, text, _ = load_model(args, manifests['train']['vocabulary'])
    original_tail = {'block.'+k: v.detach().cpu().clone() for k, v in original.visual.transformer.resblocks[-1].state_dict().items()}
    original_tail.update({'norm.'+k: v.detach().cpu().clone() for k, v in original.visual.ln_post.state_dict().items()})
    original_tail['projection'] = original.visual.proj.detach().cpu().clone()
    counts = {'sampled_prefix_objects': 0, 'checkpoints': 0, 'tail_forwards': 0, 'frozen_tail_arrays': 0}
    maximum_prefix, maximum_forward, maximum_numpy, maximum_frozen_cache = 0., 0., 0., 0.
    numpy_internal_rank_differences = []
    banks = {}
    for split, manifest in manifests.items():
        bank, reference = token_bank(args, split); banks[split] = (bank, reference)
        metadata = read(root / f'tokens/{split}.json')
        assert bank.dtype == np.float32 and list(bank.shape) == metadata['shape']
        assert metadata['dtype'] == 'float32' and metadata['rgb_bridge_max_abs'] <= 1e-5
        assert metadata['manifest_sha256'] == file_hash(root / f'manifests/{split}.json')
        old_path = args.reference / ('features/rgb_dev.npz' if split == 'dev' else f'features/rgb_{split}/features.npz')
        assert file_hash(old_path) == metadata['old_rgb_sha256']
        np.testing.assert_allclose(reference.cpu().numpy(), np.load(old_path)['features'], atol=1e-5, rtol=0)
        for index in (0, len(bank)//2, len(bank)-1):
            scene_indices = rows_by_scene(manifest)[manifest['samples'][index]['scene']]
            chunk, images = next((c, images) for c, images in render_scene_rows(manifest, scene_indices) if index in c)
            captured = []
            def capture(module, inputs):
                captured.append(inputs[0].detach().clone())
            handle = original.visual.transformer.resblocks[-1].register_forward_pre_hook(capture)
            with torch.inference_mode():
                original.encode_image(image_tensor(images), normalize=True)
            handle.remove()
            actual = captured[0]
            if not original.visual.transformer.batch_first:
                actual = actual.transpose(0, 1)
            actual = actual.reshape(len(chunk), 4, 197, 768)[chunk.index(index)].cpu().numpy()
            np.testing.assert_allclose(actual, bank[index], atol=5e-5, rtol=2e-5)
            maximum_prefix = max(maximum_prefix, float(np.max(np.abs(actual-bank[index]))))
            counts['sampled_prefix_objects'] += 1
    warm_heads = {}
    for seed in SEEDS:
        for spec in SPECS:
            directory = root / f'models/{spec}/seed{seed}'
            head = torch.load(directory / 'warm_head.pt', weights_only=True, map_location='cpu')
            if seed in warm_heads:
                assert all(torch.equal(head[k], warm_heads[seed][k]) for k in head)
            else:
                warm_heads[seed] = head
            assert len(read(directory / 'warm_history.json')) == 40
            history = read(directory / 'history.json')
            assert [r['epoch'] for r in history] == list(range(1, 7))
            for row in history:
                assert row['objects'] == 28730 and np.isfinite([row['loss'], row['ce'], row['anchor']]).all()
                assert row['anchor'] >= 0
                assert abs(row['loss']-row['ce']-(row['anchor'] if spec == 'tail_anchor' else 0)) < 1e-6
                if spec == 'frozen_head':
                    assert all(k.startswith('head.') for k in row['changed_state'])
                else:
                    assert any(k.startswith('block.') for k in row['changed_state'])
            gradient = read(directory / 'gradients.json')
            assert np.isfinite(list(gradient.values())).all() and sum(gradient.values()) > 0
            assert any(k.startswith('block.') for k in gradient) == (spec != 'frozen_head')
            for epoch in EPOCHS:
                path = directory / f'e{epoch}.pt'
                assert file_hash(path) == read(path.with_suffix('.json'))['sha256']
                record = torch.load(path, weights_only=True, map_location='cpu')
                assert (record['spec'], record['seed'], record['epoch'], record['train_objects']) == (spec, seed, epoch, 28730)
                assert record['config_sha256'] == file_hash(root / 'config.json')
                state = record['state_dict']; assert set(state) == set(original_tail) | {'head.weight', 'head.bias'}
                if spec == 'frozen_head':
                    for k, expected in original_tail.items():
                        assert torch.equal(state[k], expected), k
                        counts['frozen_tail_arrays'] += 1
                else:
                    assert not torch.equal(state['block.attn.in_proj_weight'], original_tail['block.attn.in_proj_weight'])
                for role, split in (('validation', 'eval'), ('historical_test', 'dev')):
                    actual, independent, cache_difference = replay(original.visual, state, *banks[split], spec == 'frozen_head')
                    maximum_frozen_cache = max(maximum_frozen_cache, cache_difference)
                    saved = np.load(root / role / f'{spec}_e{epoch}_seed{seed}_raw.npy')
                    np.testing.assert_allclose(actual, saved, atol=2e-5, rtol=2e-5)
                    saved_order = np.argsort(-saved, axis=1, kind='stable')[:, :5]
                    np.testing.assert_array_equal(np.argsort(-actual, axis=1, kind='stable')[:, :5], saved_order)
                    independent_order = np.argsort(-independent, axis=1, kind='stable')[:, :5]
                    np.testing.assert_array_equal(independent_order[:, 0], saved_order[:, 0])
                    np.testing.assert_array_equal(np.sort(independent_order, axis=1), np.sort(saved_order, axis=1))
                    for index in np.flatnonzero((independent_order != saved_order).any(1)):
                        classes = saved_order[index]
                        numpy_internal_rank_differences.append({'spec': spec, 'seed': seed, 'epoch': epoch,
                            'split': split, 'index': int(index), 'saved_top5': classes.tolist(),
                            'numpy_top5': independent_order[index].tolist(),
                            'saved_scores': saved[index, classes].tolist(),
                            'numpy_scores': independent[index, classes].tolist()})
                    for alternative in (actual, independent):
                        assert recount(alternative, manifests[split])[0] == recount(saved, manifests[split])[0]
                    maximum_forward = max(maximum_forward, float(np.max(np.abs(actual-saved))))
                    maximum_numpy = max(maximum_numpy, float(np.max(np.abs(independent-saved))))
                    if split == 'eval':
                        np.testing.assert_array_equal(saved, np.load(root / f'validation_raw/{spec}_e{epoch}_seed{seed}.npy'))
                    counts['tail_forwards'] += 1
                counts['checkpoints'] += 1
                print(json.dumps({'stage': 'checkpoint_replayed', 'spec': spec, 'seed': seed, 'epoch': epoch}), flush=True)
    base = bases(args, 'validation', 'eval')
    grid = read(root / 'validation_grid.json'); assert len(grid) == 111
    for row in grid:
        arrays = [base[row['name']]]*3 if row['name'] in base else [
            (1-row['weight'])*base[row['base']]+row['weight']*z(np.load(root /
            f'validation_raw/{row["name"]}_e{row["epoch"]}_seed{seed}.npy').astype(float)) for seed in SEEDS]
        metrics = [recount(v, manifests['eval'])[0] for v in arrays]
        assert [v['correct'] for v in metrics] == row['correct_by_seed']
        assert np.mean([v['correct'] for v in metrics]) == row['mean_correct']
        assert abs(np.mean([v['class_mean_accuracy'] for v in metrics])-row['mean_macro']) < 1e-12
    selection = read(root / 'selection.json'); ref = next(r for r in grid if r['name'] == 'r8_M')
    for mode in ('P', 'B', 'M'):
        assert pick(grid, ref, mode) == selection['locked'][mode]
    for spec in SPECS:
        assert pick([r for r in grid if r['name'] == spec], ref, 'P') == selection['per_candidate'][spec]
    records = read(root / 'results.json')['records']; comparisons = {}
    for role, split in (('validation', 'eval'), ('historical_test', 'dev')):
        base = bases(args, role, split); hits = {}
        for row in [r for r in records if r['role'] == role]:
            selected, seed = row['selection'], row['seed']
            expected = base[selected['name']] if selected['name'] in base else (
                (1-selected['weight'])*base[selected['base']]+selected['weight']*z(np.load(root /
                role / f'{selected["name"]}_e{selected["epoch"]}_seed{seed}_raw.npy').astype(float)))
            saved = np.load(root / role / f'{row["method"]}_seed{seed}.npy')
            np.testing.assert_array_equal(saved, expected)
            summary, hit, _, _ = recount(saved, manifests[split]); assert summary == row['summary']
            hits.setdefault(row['method'], []).append(hit)
        for mode in ('P', 'B', 'M'):
            for reference in REFERENCES:
                comparisons[f'{role}:locked_{mode}-{reference}'] = paired_bootstrap(
                    np.array(hits['locked_'+mode]), np.array(hits[reference]), manifests[split])
    result = {'status': 'passed', 'counts': counts, 'maximum_prefix_abs': maximum_prefix,
              'maximum_forward_abs': maximum_forward, 'maximum_numpy_head_abs': maximum_numpy,
              'maximum_frozen_cache_abs': maximum_frozen_cache, 'validation_grid_rows': len(grid),
              'numpy_internal_rank_differences': numpy_internal_rank_differences,
              'prediction_arrays_recounted': len(records), 'comparisons': comparisons,
              'scope': 'All cache hashes, nine sampled original-batch full-prefix input replays, paired warm heads,18 checkpoints/36 tail forwards with exact original-precision Top5 order; independent float64 heads retain numeric tolerance,exact Top1/Top5 membership/all metrics,internal rank differences explicitly listed. Full validation selection/fusion/metrics/bootstrap; not every prefix or optimizer trajectory.',
              'inherited_reference_limit': 'r8 rotation precision limit'}
    write_json(root / 'AUDIT.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'reference', 'r4', 'r8', 'clip'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    audit(args)
