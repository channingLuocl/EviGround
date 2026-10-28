"""Display full-validation cumulative metrics around the unchanged batch runner."""
import argparse
from functools import partial
import json
from pathlib import Path
import runpy
import sys
import time

from retry_state import success


class CumulativeProgress:
    def __init__(self, total, stdout=None, stderr=None, initial_results=(), *, baseline):
        self.rows = {r['csv_row']: r for r in baseline}
        if len(self.rows) != len(baseline):
            raise ValueError('Duplicate query in cumulative baseline')
        self.total, self.done = total, len(initial_results)
        self.seen = set()
        for row in initial_results:
            self.replace(row)
        self.initial_done = self.done
        self.stdout, self.stderr = stdout or sys.stdout, stderr or sys.stderr
        self.tty = self.stderr.isatty()
        self.started = time.monotonic()
        self.draw()

    def replace(self, row):
        key = row['csv_row']
        if key not in self.rows or key in self.seen:
            raise ValueError(f'Unexpected or repeated retry result: {key}')
        self.rows[key] = row
        self.seen.add(key)

    def stats(self):
        rows = list(self.rows.values())
        valid = [r for r in rows if not r.get('error') and r.get('predicted_id') is not None]
        return {'total': len(rows), 'correct25': sum(success(r) for r in rows),
                'correct50': sum(r.get('iou_3d', 0) >= .5 for r in valid),
                'errors': sum(bool(r.get('error')) for r in rows),
                'abstentions': sum(not r.get('error') and r.get('predicted_id') is None for r in rows)}

    def draw(self):
        stats = self.stats()
        fraction = self.done / self.total if self.total else 1
        filled = int(24 * fraction)
        elapsed = time.monotonic() - self.started
        processed = self.done - self.initial_done
        eta = elapsed * (self.total - self.done) / processed if processed else 0
        denominator = stats['total']
        acc25 = stats['correct25'] / denominator if denominator else 0
        acc50 = stats['correct50'] / denominator if denominator else 0
        message = (f'Cumulative Acc@0.25 {stats["correct25"]}/{denominator} = {acc25:.2%}'
                   f' | Acc@0.5 {stats["correct50"]}/{denominator} = {acc50:.2%}'
                   f' | Retry [{"#" * filled}{"-" * (24-filled)}] {self.done}/{self.total}'
                   f' | {elapsed:.0f}s | ETA {eta:.0f}s')
        self.stderr.write(('\r\033[2K' if self.tty else '') + message +
                          ('\n' if not self.tty or self.done == self.total else ''))
        self.stderr.flush()

    def update(self, result):
        self.replace(result)
        if self.tty:
            self.stderr.write('\r\033[2K')
            self.stderr.flush()
        print(json.dumps({'predicted_id': result.get('predicted_id'), 'correct25': success(result)},
                         ensure_ascii=False), file=self.stdout, flush=True)
        self.done += 1
        self.draw()


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--baseline', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    baseline = json.loads(args.baseline.read_text())['results']
    if len(baseline) != 9508:
        raise ValueError('Full cumulative display requires all 9508 baseline queries')
    # Only the evaluation parent's console reporter changes. Inference workers
    # still launch from the original file with the original worker arguments.
    import batch_progress
    batch_progress.BatchProgress = partial(CumulativeProgress, baseline=baseline)
    runner = Path(__file__).resolve().parent / 'source_r18/local/run_grounding_v9.py'
    sys.argv = [str(runner), *remaining]
    runpy.run_path(str(runner), run_name='__main__')


if __name__ == '__main__':
    main()
