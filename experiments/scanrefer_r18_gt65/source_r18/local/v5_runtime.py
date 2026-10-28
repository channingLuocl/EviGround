"""Per-sample recovery budget and auditable pending-review checkpoints."""
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

sample_recovery = ContextVar('see_ground_recovery', default=None)


def ledger_attempt(event, attempt_id=None, **fields):
    """Persist actual HTTP attempts independently of replayed stage checkpoints."""
    budget = sample_recovery.get()
    if not budget or not budget.get('api_checkpoint'):
        return None
    attempt_id = attempt_id or uuid.uuid4().hex
    path = Path(budget['api_checkpoint']).with_name('api_attempts.jsonl')
    row = {'event': event, 'attempt_id': attempt_id, 'time_unix': time.time(), **fields}
    with path.open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return attempt_id


def input_identity(query, objects, config):
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py')) if not p.name.startswith(('test_', '._'))}
    raw = json.dumps({'query': query, 'objects': objects, 'config': config, 'code': code}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def save_checkpoint(path, value):
    from qwen_api import _redact
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(_redact(value, limit=None), ensure_ascii=False, indent=2))
    tmp.replace(path)


def retryable(diagnostic):
    return diagnostic.get('failure_kind') in ('timeout', 'network') or diagnostic.get('http_status') in (502, 503, 504)


def file_identity(paths):
    result = {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        result[path.name] = digest.hexdigest()
    return result


def persist_api_recovery(budget):
    if budget.get('api_checkpoint'):
        save_checkpoint(budget['api_checkpoint'], {k: budget[k] for k in
            ('used', 'attempts', 'completed_requests', 'checkpoint_identity', 'failed_request') if k in budget})


def actual_attempts(call):
    if call.get('cache_hit'):
        return 0
    attempts = call.get('api_attempts') or call.get('api_diagnostic', {}).get('api_attempts')
    return len(attempts) if attempts else 1
