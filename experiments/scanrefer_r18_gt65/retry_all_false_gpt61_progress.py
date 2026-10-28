"""Cumulative progress for the independent all-false model retry."""
import argparse
from functools import partial
import json
from pathlib import Path
import runpy
import sys

from retry_null_progress import CumulativeProgress
from retry_state import success


class VisualProgress(CumulativeProgress):
    def update(self, result):
        self.replace(result)
        if self.tty:
            self.stderr.write('\r\033[2K')
            self.stderr.flush()
        print(json.dumps({'csv_row': result['csv_row'], 'predicted_id': result.get('predicted_id'),
            'correct25': success(result), 'null_reason': result.get('null_reason')},
            ensure_ascii=False), file=self.stdout, flush=True)
        self.done += 1
        self.draw()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--baseline', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    baseline = json.loads(args.baseline.read_text())['results']
    if len(baseline) != 9508:
        raise ValueError('Cumulative display requires all 9508 baseline queries')
    import batch_progress
    batch_progress.BatchProgress = partial(VisualProgress, baseline=baseline)
    runner = Path(__file__).resolve().parent / 'visual109_runtime/run_grounding_v9.py'
    sys.argv = [str(runner), *remaining]
    runpy.run_path(str(runner), run_name='__main__')


if __name__ == '__main__':
    main()
