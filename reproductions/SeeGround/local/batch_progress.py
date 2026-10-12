"""Minimal batch console output; full results remain in JSON artifacts."""
import json
import sys
import time


class BatchProgress:
    def __init__(self, total, stdout=None, stderr=None):
        self.total, self.done = total, 0
        self.stdout, self.stderr = stdout or sys.stdout, stderr or sys.stderr
        self.started = time.monotonic()
        self.tty = self.stderr.isatty()
        self.draw()

    def draw(self):
        fraction = self.done / self.total if self.total else 1
        filled = int(24 * fraction)
        elapsed = time.monotonic() - self.started
        eta = elapsed * (self.total - self.done) / self.done if self.done else 0
        message = f'[{"#" * filled}{"-" * (24 - filled)}] {self.done}/{self.total} {fraction:.0%} | {elapsed:.0f}s | ETA {eta:.0f}s'
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
        self.draw()
