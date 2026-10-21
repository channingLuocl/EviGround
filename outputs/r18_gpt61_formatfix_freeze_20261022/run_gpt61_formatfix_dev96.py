"""Run the fixed development cohort with the isolated R18 relay-format source."""
import json
from pathlib import Path
import subprocess

import gpt61_continuation as setup
from validate_gpt61_promotion import audit_run


BASE = Path(__file__).resolve().parent
SOURCE = BASE / 'source_r18_gpt61_formatfix'
SELECTION = BASE / 'dev96_manifest.json'
OUTPUT = setup.RUNS / 'v74_r18_gpt61_formatfix_dev96_20261021'
START_AUDIT = BASE / 'r18_formatfix_dev96_start_audit.json'
FINAL_AUDIT = BASE / 'r18_formatfix_dev96_final_audit.json'


def main():
    setup.SOURCE = SOURCE
    selection = setup.read(SELECTION)
    validation = setup.read(BASE / 'validation192_manifest.json')
    selected_ids = {row['csv_row'] for row in selection['rows']}
    validation_ids = {row['csv_row'] for row in validation['rows']}
    assert len(selected_ids) == len(selection['rows']) == 96
    assert len(validation_ids) == len(validation['rows']) == 192
    assert not selected_ids & validation_ids
    assert not OUTPUT.exists(), 'Inspect an existing output before starting inference'

    source_hashes = setup.source_hashes()
    setup.save(START_AUDIT, {
        'source': SOURCE.name,
        'source_code_sha256': source_hashes,
        'selection_sha256': setup.digest(SELECTION),
        'queries': len(selected_ids),
        'disjoint_from_validation192': True,
        'category_top_k': 5,
        'historical_predictions_reused': False,
        'workers': 4,
        'api_concurrency': 4,
        'format_attempt_limit': 4,
        'transient_transport_retries': 1,
        'read_timeout_seconds': 240,
        'minimum_correct_for_validation': 68,
        'GT_input': False,
    })

    env = setup.environment(relay=True)
    env['PYTHONPATH'] = str(SOURCE / 'local')
    log_path = BASE / 'r18_formatfix_dev96.log'
    command = setup.command(SELECTION, OUTPUT, workers=4)
    with log_path.open('w') as log:
        returncode = subprocess.run(command, cwd=SOURCE, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, check=False).returncode
    if returncode:
        raise SystemExit(f'Format-fix dev96 failed with exit {returncode}; inspect {log_path}')
    assert setup.source_hashes() == source_hashes, 'Source changed during inference'

    audit = audit_run(OUTPUT, SELECTION)
    audit['source'] = SOURCE.name
    audit['source_code_sha256'] = source_hashes
    setup.save(FINAL_AUDIT, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
