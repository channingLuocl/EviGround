"""Server workflow: acquire minimal ScanNet files, prepare, cache, train and score."""
from eviground.paths import source_file
from eviground.paths import PACKAGE_ROOT, source_files
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen

from eviground.categories.category_common_v82 import ROOT, file_hash, run_lock
from eviground.classifiers.category_v10.data import SUFFIXES, convert_scene, scene_names
from eviground.evaluation.run_storage import write_json


def download(url, destination, reserve_bytes=5*1024**3):
    """Only complete downloads become final files; retries never append to partial bytes."""
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+'.part')
    for attempt in range(3):
        try:
            with urlopen(url, timeout=60) as response, temporary.open('wb') as handle:
                expected = response.headers.get('Content-Length')
                if expected and shutil.disk_usage(destination.parent).free < int(expected)+reserve_bytes:
                    raise ValueError('Insufficient disk space; reserving 5 GiB for features and models')
                while True:
                    chunk = response.read(1024*1024)
                    if not chunk:
                        break
                    if shutil.disk_usage(destination.parent).free < reserve_bytes:
                        raise ValueError('Data volume has less than 5 GiB free; stop before filling it')
                    handle.write(chunk)
            if expected is not None and temporary.stat().st_size != int(expected):
                raise OSError('Downloaded length does not match Content-Length')
            temporary.replace(destination)
            return
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def run_pipeline(args):
    work = args.work_root.resolve()
    with run_lock(work):
        config = {'version': 'v10-server-pipeline-r1', 'train_scenes_sha256': file_hash(args.train_scenes),
                  'evaluation_scenes_sha256': file_hash(args.eval_scenes),
                  'evaluation_scene_root': str(args.eval_scene_root.resolve()),
                  'base_url': args.base_url, 'epochs': args.epochs, 'seeds': args.seeds,
                  'terms_confirmed_by_user': True,
                  'code_sha256': {p.relative_to(PACKAGE_ROOT).as_posix(): file_hash(p) for p in source_files()
                                  if not p.name.startswith('test_')}}
        if (work/'pipeline_config.json').exists():
            if json.loads((work/'pipeline_config.json').read_text()) != config:
                raise ValueError('Pipeline configuration changed; use a fresh work root')
        write_json(work/'pipeline_config.json', config)
        started = time.time()

        def status(stage, **fields):
            value = {'stage': stage, 'elapsed_s': time.time()-started, **fields}
            write_json(work/'status.json', value)
            print(json.dumps(value), flush=True)

        def stage(name, arguments, completed_file):
            if completed_file.exists():
                print(f'Reusing completed {name}: {completed_file}', flush=True)
                return
            status(name, command=[sys.executable, *map(str, arguments)])
            subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, check=True)
            if not completed_file.exists():
                raise ValueError(f'Stage did not produce {completed_file}')

        try:
            scenes = scene_names(args.train_scenes)
            status('download_and_convert', completed=0, selected=len(scenes))

            def acquire(scene):
                output = work/'scenes'/scene
                if (output/'provenance.json').exists():
                    if not all((output/name).is_file() for name in ('scene.npz', 'objects_gt.json')):
                        raise ValueError(f'Incomplete converted scene: {output}')
                    return scene
                for suffix in SUFFIXES:
                    name = scene+suffix
                    download(f'{args.base_url.rstrip("/")}/{scene}/{name}', work/'raw'/scene/name)
                convert_scene(work/'raw', scene, output)
                return scene

            with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
                jobs = [pool.submit(acquire, s) for s in scenes]
                for index, future in enumerate(as_completed(jobs), 1):
                    scene = future.result()
                    status('download_and_convert', completed=index, selected=len(scenes), latest_scene=scene)

            # Link existing validation data without copying or changing the source files.
            for scene in scene_names(args.eval_scenes):
                source = (args.eval_scene_root/scene).resolve()
                if not all((source/name).is_file() for name in ('scene.npz', 'objects_gt.json')):
                    raise ValueError(f'Missing existing validation input: {source}')
                destination = work/'scenes'/scene
                if destination.is_symlink() and destination.resolve() == source:
                    continue
                if destination.exists():
                    raise ValueError(f'Validation destination already occupied: {destination}')
                destination.symlink_to(source, target_is_directory=True)

            run = source_file('category_v10/run.py')
            stage('prepare_manifests', [run, 'prepare', '--train-scenes', args.train_scenes,
                  '--eval-scenes', args.eval_scenes, '--scene-root', work/'scenes',
                  '--output', work/'manifests'], work/'manifests/split.json')
            for split in ('train', 'dev', 'eval'):
                stage(f'cache_{split}', [run, 'cache', '--manifest', work/f'manifests/{split}.json',
                      '--output', work/f'features/{split}'], work/f'features/{split}/features.npz')
            for seed in args.seeds:
                for mode in ('global', 'attention', 'conditioned'):
                    model = work/f'models/seed{seed}/{mode}'
                    # A checkpoint alone can represent an interrupted, unfinished training.
                    complete = model/'TRAINING_COMPLETE.json'
                    if not complete.exists():
                        if model.exists() and any(p.name != '.run.lock' for p in model.iterdir()):
                            raise ValueError(f'Interrupted training requires a fresh model directory: {model}')
                        stage(f'train_seed{seed}_{mode}', [run, 'train',
                              '--train-cache', work/'features/train/features.npz',
                              '--dev-cache', work/'features/dev/features.npz', '--mode', mode,
                              '--seed', seed, '--epochs', args.epochs, '--device', 'cuda',
                              '--output', model], model/'head.pt')
                        write_json(complete, {'epochs': args.epochs, 'seed': seed, 'mode': mode,
                                             'head_sha256': file_hash(model/'head.pt')})
            # All models are selected on internal dev before any final validation scoring.
            comparisons = []
            for seed in args.seeds:
                for mode in ('global', 'attention', 'conditioned'):
                    output = work/f'predictions/seed{seed}/{mode}'
                    stage(f'predict_seed{seed}_{mode}', [run, 'predict',
                          '--cache', work/'features/eval/features.npz',
                          '--head', work/f'models/seed{seed}/{mode}/head.pt', '--device', 'cuda',
                          '--output', output], output/'FROZEN.json')
                    result = json.loads((output/'results.json').read_text())
                    comparisons.append({'seed': seed, 'mode': mode, **result['summary']})
            write_json(work/'comparison.json', comparisons)
            status('complete', models=len(comparisons), comparison=str(work/'comparison.json'))
        except BaseException as exc:
            status('failed', error=f'{type(exc).__name__}: {exc}')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--train-scenes', type=Path, default=ROOT/'data/scannet/scannetv2_train.txt')
    parser.add_argument('--eval-scenes', type=Path, default=ROOT/'data/validation/nr3d_val_full.json')
    parser.add_argument('--eval-scene-root', type=Path, default=ROOT/'local_outputs')
    parser.add_argument('--base-url', default='https://kaldir.vc.in.tum.de/scannet/v2/scans')
    parser.add_argument('--download-workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--terms-confirmed', action='store_true',
                        help='Only set after the human user has agreed to the ScanNet terms')
    args = parser.parse_args()
    if not args.terms_confirmed:
        parser.error('User confirmation of ScanNet terms is required before acquisition')
    if args.download_workers < 1 or args.epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Workers and epochs must be positive; seeds must be distinct')
    run_pipeline(args)


if __name__ == '__main__':
    main()
