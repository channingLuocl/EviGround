"""Run the three frozen PointNeXt head controls and export the shared table metrics."""
from eviground.paths import source_file
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from eviground.categories.category_common_v82 import ROOT, category_vocabulary, file_hash, normalized_category, run_lock
from eviground.classifiers.category_v10 import run
from eviground.classifiers.category_v10.model import CategoryHead
from eviground.evaluation.run_storage import write_json
from eviground.experiments.train_pointbert_scannet import load_manifests, metrics


MODES = ('global', 'attention', 'conditioned')
NAMES = {'global': 'PointNeXt（冻结主干＋全局分类头）',
         'attention': 'PointNeXt（全局＋局部普通注意力）',
         'conditioned': 'PointNeXt（类别／竞争类别条件局部聚合）'}


def experiment(args):
    work = args.output.resolve()
    source = args.feature_root.resolve()
    with run_lock(work):
        manifests = load_manifests(source/'manifests')
        hashes = {s: file_hash(source/f'features/{s}/features.npz') for s in ('train', 'dev', 'eval')}
        config = {'version': 'pointnext-adaptation-table-r1', 'torch': str(torch.__version__),
                  'seeds': [42, 43, 44], 'epochs': 40, 'feature_root': str(source),
                  'cache_sha256': hashes, 'code_sha256': run.code_hashes(),
                  'runner_sha256': file_hash(Path(__file__)),
                  'selection': 'minimum internal-dev CE; evaluate only after all nine models finish',
                  'table_class_mean': 'all normalized target classes, including OOV',
                  'evaluation_status': 'previously inspected fixed validation set'}
        if (work/'config.json').exists() and json.loads((work/'config.json').read_text()) != config:
            raise ValueError('Experiment configuration changed; use a new output directory')
        write_json(work/'config.json', config)

        def status(stage, **values):
            value = {'stage': stage, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), **values}
            write_json(work/'status.json', value)
            print(json.dumps(value), flush=True)

        try:
            for split in ('train', 'dev', 'eval'):
                data, meta = run.read_cache(source/f'features/{split}/features.npz')
                rows = manifests[split]['samples']
                if (meta['code_sha256'] != config['code_sha256'] or
                        [(r['scene'], r['bbox_id'], r['category']) for r in meta['records']] !=
                        [(r['scene'], r['bbox_id'], r['category']) for r in rows] or
                        not torch.equal(data['labels'], torch.tensor([r['label'] for r in rows]))):
                    raise ValueError(f'Feature/manifest/source mismatch: {split}')
                del data
            status('preflight_passed', instances={s: len(m['samples']) for s, m in manifests.items()})
            for seed in config['seeds']:
                for mode in MODES:
                    model = work/f'models/seed{seed}/{mode}'
                    complete = model/'TRAINING_COMPLETE.json'
                    if complete.exists():
                        marker = json.loads(complete.read_text())
                        if marker['head_sha256'] != file_hash(model/'head.pt'):
                            raise ValueError('Completed checkpoint changed')
                        continue
                    status('training', seed=seed, mode=mode)
                    subprocess.run([sys.executable, str(source_file('category_v10/run.py')), 'train',
                                    '--train-cache', str(source/'features/train/features.npz'),
                                    '--dev-cache', str(source/'features/dev/features.npz'),
                                    '--mode', mode, '--seed', str(seed), '--epochs', '40',
                                    '--threads', '8', '--device', 'cuda', '--output', str(model)],
                                   cwd=ROOT, check=True)
                    history = json.loads((model/'history.json').read_text())
                    if len(history) != 40:
                        raise ValueError('Training budget incomplete')
                    write_json(complete, {'head_sha256': file_hash(model/'head.pt'), 'epochs': 40})

            data, meta = run.read_cache(source/'features/eval/features.npz')
            rows, vocabulary = manifests['eval']['samples'], list(category_vocabulary())
            comparisons = []
            torch.set_num_threads(8)
            for seed in config['seeds']:
                for mode in MODES:
                    status('evaluating', seed=seed, mode=mode)
                    model = work/f'models/seed{seed}/{mode}'
                    checkpoint = torch.load(model/'head.pt', map_location='cpu', weights_only=True)
                    training = checkpoint['config']
                    if (training['train_cache_sha256'] != hashes['train'] or
                            training['dev_cache_sha256'] != hashes['dev'] or
                            training['code_sha256'] != config['code_sha256'] or
                            training['extractor'] != meta['extractor'] or
                            training['text_sha256'] != meta['text_sha256']):
                        raise ValueError('Model/cache provenance differs')
                    run.check_sources(training['train_sources']+training['dev_sources'], meta['sources'])
                    history = json.loads((model/'history.json').read_text())
                    best = min(history, key=lambda r: r['dev']['category_loss'])['epoch']
                    if len(history) != 40 or best != checkpoint['best_epoch']:
                        raise ValueError('Internal-dev model selection differs from checkpoint')
                    head = CategoryHead(checkpoint['state_dict']['category_text'], mode).cuda()
                    head.load_state_dict(checkpoint['state_dict'], strict=True)
                    logits = run.evaluate_head(head, data, 64, 'cuda')
                    if logits.shape != (3413, 197) or not torch.isfinite(logits).all():
                        raise ValueError('Invalid evaluation logits')
                    order = logits.argsort(dim=-1, descending=True, stable=True)
                    predictions = [{'scene': r['scene'], 'bbox_id': r['bbox_id'], 'status': 'completed',
                                    'predicted_object': vocabulary[int(indices[0])],
                                    'top5': [vocabulary[int(i)] for i in indices[:5]]}
                                   for r, indices in zip(rows, order)]
                    summary = metrics(logits, manifests['eval'])
                    # Independently recompute lexical accuracy from exported names.
                    hits = [normalized_category(p['predicted_object']) == normalized_category(r['category'])
                            for p, r in zip(predictions, rows)]
                    if sum(hits) != summary['correct'] or summary['targets_outside_vocabulary'] != 157:
                        raise ValueError('Exported prediction scoring mismatch')
                    output = work/f'predictions/seed{seed}/{mode}'
                    output.mkdir(parents=True, exist_ok=True)
                    np.save(output/'logits.npy', logits.numpy(), allow_pickle=False)
                    write_json(output/'predictions.json', {'results': predictions})
                    record = {'seed': seed, 'mode': mode, 'name': NAMES[mode], 'summary': summary,
                              'best_internal_dev_epoch': best, 'epochs_completed': 40,
                              'head_sha256': file_hash(model/'head.pt'),
                              'logits_sha256': file_hash(output/'logits.npy')}
                    write_json(output/'results.json', record)
                    write_json(output/'FROZEN.json', {p: file_hash(output/p) for p in
                               ('logits.npy', 'predictions.json', 'results.json')})
                    comparisons.append(record)
                    del head
            aggregates = {mode: {key: {'mean': float(np.mean(values)), 'std': float(np.std(values, ddof=1))}
                                  for key in ('accuracy', 'exact_accuracy', 'top5_accuracy', 'class_mean_accuracy')
                                  for values in [[r['summary'][key] for r in comparisons if r['mode'] == mode]]}
                          for mode in MODES}
            write_json(work/'experiment_summary.json', {'config': config, 'comparison': comparisons,
                                                       'aggregate': aggregates, 'audit_passed': True})
            table = ['模型\t评估对象数\tTrue正确数\tTrue准确率\tExact Top-1\tTop-5\tClass Mean\tErrors']
            for record in comparisons:
                s = record['summary']
                table.append(f'{record["name"]} seed{record["seed"]}\t{s["count"]}\t{s["correct"]}'
                             f'\t{s["accuracy"]:.2%}\t{s["exact_accuracy"]:.2%}\t{s["top5_accuracy"]:.2%}'
                             f'\t{s["class_mean_accuracy"]:.2%}\t{s["errors"]}')
            (work/'table.tsv').write_text('\n'.join(table)+'\n')
            status('completed', models=len(comparisons), audit_passed=True)
        except BaseException as exc:
            status('failed', error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    os.environ.setdefault('OMP_NUM_THREADS', '8')
    os.environ.setdefault('MKL_NUM_THREADS', '8')
    experiment(parser.parse_args())
