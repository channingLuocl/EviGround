"""Archive failed trajectories before retrying them in the original output directory."""
import argparse
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def checked_results(output, manifest):
    rows = read(manifest)['rows']
    results = read(output / 'results.json')['results']
    expected = {row['csv_row']: row for row in rows}
    observed = {row['csv_row']: row for row in results}
    if len(results) != len(observed) or set(expected) != set(observed):
        raise ValueError('Finish the current run before starting a new retry round')
    for key, row in expected.items():
        if any(observed[key].get(name) != value for name, value in row.items()):
            raise ValueError(f'Result does not match selection: {key}')
        evaluation = output / f'q{key:05d}' / 'evaluation.json'
        if not evaluation.is_file() or read(evaluation) != observed[key]:
            raise ValueError(f'Per-case evaluation differs from results: {key}')
    return results


def success(row):
    return not row.get("error") and row.get("predicted_id") is not None and row.get("iou_3d", 0) >= .25


def prepare(output, manifest, preview=False, max_rounds=0):
    for directory in (output.resolve(), *output.resolve().parents):
        if (directory / 'FROZEN.json').is_file():
            raise ValueError(f'Frozen run is read-only: {directory}')
    pending = output / 'retry_pending.json'
    if pending.exists():
        state = read(pending)
        print(f"Resuming retry round {state['round']} ({len(state['failed_ids'])} selected failures).", flush=True)
    else:
        results = checked_results(output, manifest)
        failed = [row['csv_row'] for row in results if not success(row)]
        if preview:
            print(f'Retry preview: preserve {len(results)-len(failed)} true, rerun {len(failed)} false; no results changed.', flush=True)
            return
        if not failed:
            print('All selected cases are true; nothing to retry.', flush=True)
            return
        history_path = output / 'retry_history.json'
        history = read(history_path) if history_path.exists() else {'rounds': []}
        number = len(history['rounds']) + 1
        if number > max_rounds:
            raise ValueError(f'Frozen maximum retry rounds exhausted: {max_rounds}')
        archive = output / 'retry_archive' / f'round{number:03d}'
        archive.mkdir(parents=True, exist_ok=False)
        for name in ('results.json', 'scanrefer_metrics.json', 'manifest.json', 'execution_sessions.jsonl'):
            if (output / name).exists():
                shutil.copyfile(output / name, archive / name)
        state = {'round': number, 'archive': str(archive), 'failed_ids': failed,
                 'total': len(results), 'correct_before': len(results)-len(failed),
                 'selection_metric': 'Acc@0.25', 'max_retry_rounds': max_rounds,
                 'status': 'archiving'}
        save(pending, state)
    if preview:
        return
    archive = Path(state['archive'])
    if state['status'] == 'archiving':
        for key in state['failed_ids']:
            name = f'q{key:05d}'
            if not (archive / name).exists():
                (output / name).rename(archive / name)
        state['status'] = 'running'
        save(pending, state)
    print(f"Retry round {state['round']}: original trajectories saved in {archive}; true cases remain cached.", flush=True)


def finalize(output, manifest):
    pending = output / 'retry_pending.json'
    if not pending.exists():
        return
    state = read(pending)
    if state['status'] != 'running':
        raise ValueError('Retry archival interrupted; resume with --retry-false')
    results = checked_results(output, manifest)
    correct = sum(success(row) for row in results)
    archive = Path(state['archive'])
    original = read(archive / 'results.json')['results']
    current = {row['csv_row']: row for row in results}
    for row in original:
        if row['csv_row'] not in state['failed_ids'] and current[row['csv_row']] != row:
            raise ValueError('A preserved true result changed')
    history_path = output / 'retry_history.json'
    history = read(history_path) if history_path.exists() else {'rounds': []}
    entry = {**state, 'status': 'complete', 'correct_after': correct,
             'newly_correct': correct-state['correct_before']}
    if len(history['rounds']) < state['round']:
        history['rounds'].append(entry)
        save(history_path, history)
    report_path = output / 'results.json'
    report = read(report_path)
    report['summary'].update(evaluation_policy='cumulative_retry_false',
                             retry_rounds=len(history['rounds']),
                             original_single_pass_accuracy=history['rounds'][0]['correct_before']/state['total'])
    report['summary']['cumulative_api_attempts'] = sum(row.get('api_attempts') or 0 for row in results) + sum(
        row.get('api_attempts') or 0
        for round_info in history['rounds']
        for row in read(Path(round_info['archive']) / 'results.json')['results']
        if row['csv_row'] in round_info['failed_ids'])
    save(report_path, report)
    pending.unlink()
    print(f"Cumulative retry result: {correct}/{len(results)}; newly correct this round: {entry['newly_correct']}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preview', 'prepare', 'finalize'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--max-rounds', type=int, required=True)
    args = parser.parse_args()
    if args.action == 'finalize':
        finalize(args.output, args.manifest)
    else:
        prepare(args.output, args.manifest, preview=args.action == 'preview', max_rounds=args.max_rounds)


if __name__ == '__main__':
    main()
