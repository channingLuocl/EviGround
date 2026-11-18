"""Append local attempt evidence outside all model-visible requests."""

import json
import os
from pathlib import Path
import time
import uuid


def save_progress():
    from eviground.grounding.recovery import sample_recovery

    budget = sample_recovery.get()
    if budget and budget.get('save_progress'):
        budget['save_progress']()


def read_attempts(path):
    """Keep started-only attempts unknown, including a truncated final journal line."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines()
    attempts = {}
    for index, line in enumerate(lines):
        try:
            event = json.loads(line)
        except ValueError:
            if index == len(lines) - 1:
                break
            raise
        identity = event['local_attempt_id']
        if event['event'] == 'started':
            attempts[identity] = event
        elif identity in attempts:
            attempts[identity] = {**attempts[identity], **event}
    return list(attempts.values())


def log_attempt(attempt_id, event, **fields):
    from eviground.grounding.recovery import sample_recovery

    budget = sample_recovery.get()
    if not budget or not budget.get('api_checkpoint'):
        return
    path = Path(budget['api_checkpoint']).parent / 'http_attempts.jsonl'
    if path.exists():
        # Validate the existing prefix before changing it. A crash can leave a
        # partial final event; preserve those bytes before repairing the tail.
        read_attempts(path)
        data = path.read_bytes()
        if data:
            lines = data.splitlines(keepends=True)
            offset = sum(map(len, lines[:-1]))
            try:
                json.loads(lines[-1])
            except ValueError:
                archive = path.with_name(path.name + '.truncated-' + uuid.uuid4().hex)
                with archive.open('xb') as stream:
                    stream.write(data[offset:])
                    stream.flush()
                    os.fsync(stream.fileno())
                with path.open('r+b') as stream:
                    stream.truncate(offset)
                    stream.flush()
                    os.fsync(stream.fileno())
            else:
                if not data.endswith(b'\n'):
                    with path.open('ab') as stream:
                        stream.write(b'\n')
                        stream.flush()
                        os.fsync(stream.fileno())
    with path.open('a') as stream:
        stream.write(json.dumps(dict(local_attempt_id=attempt_id, event=event,
                                      stage=budget.get('call_stage'),
                                      execution_id=budget.get('execution_id'),
                                      time=time.time(), **fields), ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    save_progress()
