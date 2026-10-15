"""Minimal batch console output; full results remain in JSON artifacts."""
import json
import sys
import time


class BatchProgress:
    def __init__(self, total, stdout=None, stderr=None, initial_results=()):
        self.total, self.done = total, len(initial_results)
        self.correct = sum(row.get('correct') is True for row in initial_results)
        self.initial_done = self.done
        self.stdout, self.stderr = stdout or sys.stdout, stderr or sys.stderr
        self.started = time.monotonic()
        self.tty = self.stderr.isatty()
        self.draw()

    def draw(self):
        fraction = self.done / self.total if self.total else 1
        filled = int(24 * fraction)
        elapsed = time.monotonic() - self.started
        processed = self.done - self.initial_done
        eta = elapsed * (self.total - self.done) / processed if processed else 0
        accuracy = self.correct / self.done if self.done else 0
        message = (f'[{"#" * filled}{"-" * (24 - filled)}] {self.done}/{self.total} {fraction:.0%}'
                   f' | {elapsed:.0f}s | ETA {eta:.0f}s'
                   f' | True {self.correct}/{self.done} = {accuracy:.2%}')
        self.stderr.write(('\r\033[2K' if self.tty else '') + message + ('\n' if not self.tty or self.done == self.total else ''))
        self.stderr.flush()

    def update(self, result):
        if self.tty:
            self.stderr.write('\r\033[2K')
            self.stderr.flush()
        print(json.dumps({'predicted_id': result.get('predicted_id'),
                          'correct': result.get('correct', False)}, ensure_ascii=False),
              file=self.stdout, flush=True)
        self.done += 1
        self.correct += result.get('correct') is True
        self.draw()
