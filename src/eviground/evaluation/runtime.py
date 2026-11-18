"""Observe batch wall time and GPUs without changing worker inputs or budgets."""

from contextlib import contextmanager
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid

from eviground.evaluation.run_storage import write_json


def gpu_sample():
    fields = ('index', 'uuid', 'name', 'utilization.gpu', 'memory.used', 'memory.total')
    record = dict(time_unix=time.time(), interval_s=10)
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=' + ','.join(fields),
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
        record['gpus'] = [dict(zip(fields, (v.strip() for v in row)))
                          for row in csv.reader(io.StringIO(result.stdout)) if row]
    except (OSError, subprocess.SubprocessError) as exc:
        record['error'] = str(exc)
    return record


@contextmanager
def observe_batch(output, completed):
    identity = uuid.uuid4().hex
    directory = Path(output) / 'batch_executions'
    directory.mkdir(exist_ok=True)
    path = directory / (identity + '.json')
    record = dict(execution_id=identity, started_unix=time.time(), complete=False,
                  initial_processed=completed(), processed_increment=0,
                  throughput_samples_per_s=None, wall_time_s=None,
                  scope='inference coordinator only; excludes cache preparation and service startup')
    write_json(path, record)
    started = time.perf_counter()
    stop = threading.Event()

    def sample():
        with (directory / (identity + '-gpu.jsonl')).open('x') as stream:
            while True:
                sample_started = time.perf_counter()
                stream.write(json.dumps(gpu_sample()) + '\n')
                stream.flush()
                if stop.wait(max(0, 10 - (time.perf_counter() - sample_started))):
                    return

    thread = None
    if os.getenv('EVIGROUND_MONITOR_GPU') == '1':
        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
    try:
        yield record
        record['complete'] = True
    except BaseException as exc:
        record['error_type'] = type(exc).__name__
        raise
    finally:
        elapsed = time.perf_counter() - started
        record.update(finished_unix=time.time(), wall_time_s=elapsed,
            processed_increment=completed() - record['initial_processed'])
        record['throughput_samples_per_s'] = record['processed_increment'] / elapsed
        stop.set()
        if thread is not None:
            thread.join(timeout=6)
        write_json(path, record)
