"""Evaluation-only cumulative accounting; never a source of inference inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from seeground.evaluation.run_storage import write_json


def read(path):
    return json.loads(Path(path).read_text())


def collect(selection, runs):
    canonical = {row['csv_row']: row for row in selection['rows']}
    if len(canonical) != len(selection['rows']):
        raise ValueError('Duplicate selection row')
    attempts, chosen, source_hash = {}, {}, None
    run_summaries = []
    seen_runs = set()
    for directory in runs:
        directory = Path(directory).resolve()
        if directory in seen_runs:
            raise ValueError('Duplicate source run')
        seen_runs.add(directory)
        manifest = read(directory / 'manifest.json')
        if not manifest['version'].startswith('predicted-category-grounding-v9-r8-'):
            raise ValueError('Only r8 predicted-category runs may enter this cumulative report')
        digest = manifest['categories']['source_predictions_sha256']
        if source_hash is not None and digest != source_hash:
            raise ValueError('Prediction export changed between attempts')
        source_hash = digest
        run_rows = {r['csv_row']: r for r in manifest['rows']}
        results = []
        for i in sorted(run_rows):
            if i not in canonical or any(run_rows[i].get(k) != canonical[i].get(k)
                    for k in ('scene', 'query', 'target_id')):
                raise ValueError('Run does not match canonical query/answer identity')
        def load_evaluation(i):
            path = directory / f'q{i:05d}/evaluation.json'
            return read(path) if path.exists() else None
        ordered_ids = sorted(run_rows)
        with ThreadPoolExecutor(max_workers=8) as pool:
            evaluations = list(pool.map(load_evaluation, ordered_ids))
        for i, row in zip(ordered_ids, evaluations):
            if row is None:
                continue
            if any(row.get(k) != canonical[i].get(k) for k in ('csv_row', 'scene', 'query', 'target_id')):
                raise ValueError('Evaluation does not match canonical identity')
            correct = row.get('predicted_id') == canonical[i]['target_id'] and not row.get('error')
            if bool(row.get('correct')) != correct:
                raise ValueError('Stored correctness disagrees with actual answer')
            results.append(row)
            attempts.setdefault(i, []).append({'run': str(directory), 'predicted_id': row.get('predicted_id'),
                'correct': correct, 'error': row.get('error'),
                'implementation_version': row.get('implementation_version'),
                'api_attempts': row.get('api_attempts', 0)})
            # This is explicitly an evaluation oracle: no selected ID is sent back to a worker.
            if i not in chosen or (not chosen[i]['correct'] and (correct or not row.get('error'))):
                chosen[i] = {**row, 'cumulative_source_run': str(directory)}
        api_attempts = sum(r.get('api_attempts', 0) for r in results)
        usage_records = sum(isinstance(u, dict) and 'total_tokens' in u
                            for r in results for u in r.get('call_usage', []))
        run_summaries.append({'run': str(directory), 'version': manifest['version'],
            'selected': len(run_rows), 'evaluated': len(results),
            'correct': sum(r['correct'] for r in results),
            'errors': sum(bool(r.get('error')) for r in results),
            'single_pass_accuracy_over_selected': sum(r['correct'] for r in results)/len(run_rows),
            'complete': len(results) == len(run_rows),
            'api_attempts': api_attempts,
            'unreported_token_usage_attempts': max(api_attempts-usage_records, 0),
            'token_totals_are_lower_bounds': api_attempts > usage_records,
            'tokens': {key: sum((u or {}).get(key, 0) for r in results for u in r.get('call_usage', []))
                      for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
            'manifest_sha256': hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest()})
    def group(predicate):
        ids = [i for i, r in canonical.items() if predicate(r)]
        correct = sum(chosen.get(i, {}).get('correct', False) for i in ids)
        return {'selected': len(ids), 'evaluated': sum(i in chosen for i in ids),
                'correct': correct, 'accuracy_over_selected': correct/len(ids) if ids else None}
    summary = group(lambda r: True)
    summary.update(evaluation_policy='cumulative_ever_correct_evaluation_oracle',
        deployment_prediction=False, complete=summary['evaluated'] == len(canonical),
        total_api_attempts=sum(r['api_attempts'] for r in run_summaries),
        unreported_token_usage_attempts=sum(r['unreported_token_usage_attempts'] for r in run_summaries),
        remaining_error_queries=[i for i, row in sorted(chosen.items()) if row.get('error')],
        source_predictions_sha256=source_hash)
    return {'summary': summary,
        'groups': {'easy': group(lambda r: r.get('easy') is True),
                   'hard': group(lambda r: r.get('easy') is False),
                   'view_dependent': group(lambda r: r.get('view_dep') is True),
                   'view_independent': group(lambda r: r.get('view_dep') is False)},
        'single_pass_runs': run_summaries,
        'results': [chosen[i] for i in sorted(chosen)],
        'attempt_history': {str(i): rows for i, rows in sorted(attempts.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = collect(read(args.selection), args.run)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/'results.json', report)
    print(json.dumps({'summary': report['summary'], 'groups': report['groups']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
