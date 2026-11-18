"""Export local sample costs without treating unprocessed queries as zero cost."""

import argparse
import csv
import json
from pathlib import Path

from eviground.evaluation.costs import sample_cost, summarize_costs
from eviground.evaluation.run_storage import write_json


def export(run_dir):
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    results_path = run_dir / 'results.json'
    outcomes = {}
    for row in json.loads(results_path.read_text())['results'] if results_path.exists() else []:
        if row['csv_row'] in outcomes:
            raise ValueError('Duplicate evaluator result rows')
        outcomes[row['csv_row']] = row
    rows = []
    for selected in manifest['rows']:
        sample = run_dir / f"q{selected['csv_row']:05d}"
        path = sample / 'run.json'
        if not path.exists():
            continue
        run = json.loads(path.read_text())
        outcome = outcomes.get(selected['csv_row'], {})
        if outcome and any(outcome.get(key) != selected.get(key) for key in ('scene', 'query')):
            raise ValueError('Evaluator result disagrees with the run manifest')
        cost = sample_cost(run)
        rows.append(dict(csv_row=selected['csv_row'], scene=selected['scene'],
            dataset=selected.get('dataset', 'nr3d'),
            model=manifest.get('models', {}).get('EVIGROUND_MODEL'),
            category_cache_sha256=run.get('category_input', {}).get('table_sha256'),
            correct=outcome.get('correct'), error=outcome.get('error'),
            cost=cost))
    summary = summarize_costs(rows, len(manifest['rows']))
    write_json(run_dir / 'cost_samples.json', dict(rows=rows))
    write_json(run_dir / 'cost_summary.json', summary)
    columns = ['csv_row', 'scene', 'dataset', 'model', 'category_cache_sha256', 'correct']
    columns += sorted({k for r in rows for k, v in r['cost'].items() if not isinstance(v, dict)})
    with (run_dir / 'cost_samples.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: v for k, v in row.items() if k != 'cost'},
                             **{k: v for k, v in row['cost'].items() if not isinstance(v, dict)}})
    lines = ['# Local inference cost', '',
             f"Processed: {len(rows)}/{len(manifest['rows'])}. Missing values remain unknown.", '',
             '| Field | Observed | Missing | Mean |', '| --- | ---: | ---: | ---: |']
    for key, value in summary.items():
        if isinstance(value, dict) and 'mean' in value:
            lines.append(f"| {key} | {value['observed']} | {value['missing']} | {value['mean']} |")
    lines += ['', '## Judgment revision', '',
        '| Trigger | Triggered / observed | Correct / evaluated triggered | Missing history | Unknown outcome |',
        '| --- | ---: | ---: | ---: | ---: |']
    for field in ('dag_reopened', 'candidates_restored', 'revision_triggered'):
        value = summary['judgment_revision'][field]
        trigger = f"{value['trigger_rate']:.2%}" if value['trigger_rate'] is not None else 'unknown'
        success = f"{value['correction_success_rate']:.2%}" if value['correction_success_rate'] is not None else 'unknown'
        lines.append(f"| {field} | {value['triggered_samples']}/{value['observed_samples']} ({trigger}) "
            f"| {value['correct_after_trigger']}/{value['evaluated_triggered_samples']} ({success}) "
            f"| {value['missing_history_samples']} | {value['unknown_outcome_samples']} |")
    lines += ['', 'Triggers count samples once, including repeated events and resumed executions.',
        'Reopened tasks are previously completed nodes actually reset by an accepted rollback; '
        'pending-node requests and deleted nodes do not count.',
        'Success is final evaluator correctness after a trigger, not proof of causal improvement. '
        'Missing histories/outcomes remain unknown; failed and abstained evaluated samples are unsuccessful.']
    lines += ['', 'P50/P95 latency are in cost_summary.json. Startup and preprocessing costs are separate.',
              'No GPU monetary cost or throughput is inferred from token counts or per-sample latencies.']
    (run_dir / 'cost_summary.md').write_text('\n'.join(lines) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    print(json.dumps(export(parser.parse_args().run), ensure_ascii=False))
