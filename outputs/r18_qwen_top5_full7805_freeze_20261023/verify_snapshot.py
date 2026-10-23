"""Verify the frozen full-cohort result without running inference or calling an API."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess


directory = Path(__file__).resolve().parent
root = directory.parents[1]
seal = json.loads((directory / 'freeze_manifest.json').read_text())
for name, digest in seal['artifact_sha256'].items():
    assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest, name
for name, digest in seal['reference_sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name

raw_artifacts = json.loads((directory / 'raw_artifacts.json').read_text())
for artifact in raw_artifacts['artifacts']:
    raw = gzip.decompress((directory / artifact['snapshot']).read_bytes())
    assert len(raw) == artifact['uncompressed_bytes']
    assert hashlib.sha256(raw).hexdigest() == artifact['uncompressed_sha256']

integrity = json.loads((directory / 'source_input_integrity.json').read_text())
assert not integrity['issues']
for entry in integrity['code']:
    assert hashlib.sha256((root / entry['path']).read_bytes()).hexdigest() == entry['sha256']
for entry in integrity['inputs']:
    blob = subprocess.run(
        ['git', 'cat-file', 'blob', 'HEAD:' + entry['git_path']],
        cwd=root, capture_output=True, check=True).stdout
    assert hashlib.sha256(blob).hexdigest() == entry['sha256'], entry['git_path']

manifest = json.loads(gzip.decompress((directory / 'original_manifest.json.gz').read_bytes()))
assert manifest['code_sha256'] == {Path(entry['path']).name: entry['sha256'] for entry in integrity['code']}
assert manifest['categories']['gate']['top_k'] == 5
assert manifest['models']['SEEGROUND_MODEL'] == 'qwen3.7-plus'
assert manifest['evaluation_policy'] == 'fixed_predictions_no_historical_correct_reuse'
data = json.loads(gzip.decompress((directory / 'results.json.gz').read_bytes()))
rows = data['results']
expected = {row['csv_row']: row for row in manifest['rows']}
assert len(rows) == len(expected) == seal['expected_total']
assert len({row['csv_row'] for row in rows}) == len(rows)
for row in rows:
    reference = expected[row['csv_row']]
    assert all(row.get(key) == value for key, value in reference.items())
    correct = type(row.get('predicted_id')) is int and row['predicted_id'] == reference['target_id']
    assert row['correct'] is correct

metrics = json.loads((directory / 'metrics.json').read_text())
for name, key, value in [('Easy', 'easy', True), ('Hard', 'easy', False),
                         ('Dep.', 'view_dep', True), ('Indep.', 'view_dep', False),
                         ('Overall', None, None)]:
    subset = [row for row in rows if key is None or row[key] is value]
    correct = sum(row['correct'] for row in subset)
    saved = metrics['metrics'][name]
    assert saved['total'] == len(subset) and saved['correct'] == correct
    assert saved['accuracy'] == correct / len(subset)
    assert saved['percent'] == round(100 * correct / len(subset), 2)
assert metrics['metrics']['Overall']['correct'] == seal['expected_correct']
assert metrics['null'] == sum(row.get('predicted_id') is None for row in rows)
assert metrics['execution_errors'] == sum(bool(row.get('error')) for row in rows)
assert metrics['independent_per_case_audit'] == 'passed'

with (directory / 'per_query_results.csv').open(newline='') as handle:
    exported = list(csv.DictReader(handle))
assert len(exported) == len(rows)
for item, row in zip(exported, sorted(rows, key=lambda item: item['csv_row'])):
    assert int(item['csv_row']) == row['csv_row']
    assert item['predicted_id'] == ('' if row.get('predicted_id') is None else str(row['predicted_id']))
    assert item['correct'] == str(row['correct'])

audit = json.loads((directory / 'final_audit.json').read_text())
assert audit['verified_complete'] and not audit['structural_issues'] and not audit['missing_rows']
assert audit['recorded'] == len(rows) and audit['correct'] == seal['expected_correct']
assert audit['cumulative'] is False
preservation = json.loads((directory / 'preservation.json').read_text())
assert not preservation['issues']
assert preservation['all_initial_evaluations_unchanged']
assert preservation['all_initial_prediction_metadata_unchanged']
assert preservation['original_manifest_unchanged']
print(json.dumps({'verified': True, 'recorded': len(rows),
                  'correct': seal['expected_correct'],
                  'metrics_percent': {key: value['percent'] for key, value in metrics['metrics'].items()},
                  'inference_files': len(integrity['code']), 'input_files': len(integrity['inputs'])},
                 ensure_ascii=False, indent=2))
