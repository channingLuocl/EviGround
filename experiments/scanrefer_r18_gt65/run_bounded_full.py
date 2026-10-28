"""Resume a frozen full first pass, then use only its predeclared retry limit."""
import argparse
import json
import os
import subprocess

from retry_state import save
from verify import ROOT, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    args = parser.parse_args()
    frozen, _ = verify('full', args.version)
    output = ROOT / 'outputs' / args.version / 'full'
    environment = {**os.environ, 'GT65_VERSION': args.version}

    def run(retry=False):
        command = ['bash', str(ROOT / 'run.sh'), 'run', 'full']
        if retry:
            command.append('--retry-false')
        subprocess.run(command, cwd=ROOT, env=environment, check=True)

    # This also resumes a pending retry round, without selecting another round.
    run()
    while True:
        metrics = json.loads((output / 'scanrefer_metrics.json').read_text())
        assert metrics['complete'] and metrics['overall']['processed'] == 9508
        assert metrics['overall']['total'] == 9508 and metrics['pending'] == 0
        rounds = metrics.get('retry_rounds', 0)
        correct = metrics['overall']['correct25']
        achieved = correct / 9508 >= .65
        exhausted = rounds >= frozen['max_retry_rounds']
        state = {'version': args.version, 'correct25': correct, 'total': 9508,
                 'retry_rounds': rounds, 'max_retry_rounds': frozen['max_retry_rounds'],
                 'target_reached': achieved, 'retry_limit_exhausted': exhausted,
                 'single_pass_acc25': metrics.get('first_pass', metrics)['overall']['acc25']}
        save(output / 'bounded_execution.json', state)
        print(json.dumps(state), flush=True)
        if achieved:
            return
        if exhausted:
            raise SystemExit('Frozen retry limit exhausted; target remains unmet.')
        run(retry=True)


if __name__ == '__main__':
    main()
