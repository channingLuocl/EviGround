"""Configured Chat Completions client for Qwen and OpenAI-compatible providers."""
from __future__ import annotations

import base64
from contextlib import contextmanager
import fcntl
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
from PIL import Image


class QwenAPIError(ValueError):
    """Shared API failure; the legacy name preserves existing caller contracts."""

    def __init__(self, message: str, *, diagnostic: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostic = diagnostic or {}


def api_concurrency() -> int:
    """Maximum number of simultaneous outbound model requests per host."""
    raw = os.getenv('SEEGROUND_API_CONCURRENCY', '4').strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise QwenAPIError('SEEGROUND_API_CONCURRENCY must be a positive integer') from exc
    if value < 1:
        raise QwenAPIError('SEEGROUND_API_CONCURRENCY must be a positive integer')
    return value


def api_slot_dir() -> Path:
    configured = os.getenv('SEEGROUND_API_SLOT_DIR', '').strip()
    if configured:
        return Path(configured).expanduser()
    identity = f"{os.getuid()}-{api_provider()}"
    return Path(tempfile.gettempdir()) / f'seeground-api-slots-{identity}'


@contextmanager
def api_request_slot():
    """Acquire one cross-process slot before sending an outbound API request."""
    from run_storage import ensure_run_writable
    slot_dir = api_slot_dir()
    ensure_run_writable(slot_dir)
    slot_dir.mkdir(parents=True, exist_ok=True)
    limit = api_concurrency()
    started = time.perf_counter()
    while True:
        for index in range(limit):
            path = slot_dir / f'slot-{index}.lock'
            handle = path.open('a+')
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            try:
                yield round(time.perf_counter() - started, 4)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
            return
        time.sleep(0.05)


def api_provider() -> str:
    provider = os.getenv('SEEGROUND_API_PROVIDER', 'qwen').strip().lower()
    if provider not in ('qwen', 'openai'):
        raise QwenAPIError('SEEGROUND_API_PROVIDER must be qwen or openai')
    return provider


def thinking_enabled() -> bool:
    raw = os.getenv('SEEGROUND_ENABLE_THINKING', '0').strip()
    if raw not in ('0', '1'):
        raise QwenAPIError('SEEGROUND_ENABLE_THINKING must be 0 or 1')
    return raw == '1'


def thinking_token_floor() -> int:
    raw = os.getenv('SEEGROUND_THINKING_TOKEN_FLOOR', '8192').strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise QwenAPIError('SEEGROUND_THINKING_TOKEN_FLOOR must be a positive integer') from exc
    if value < 1:
        raise QwenAPIError('SEEGROUND_THINKING_TOKEN_FLOOR must be a positive integer')
    return value


def request_options(max_tokens: int, temperature: float | None, json_mode: bool,
                    response_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    provider = api_provider()
    if type(max_tokens) is not int or max_tokens < 1:
        raise QwenAPIError('max_tokens must be a positive integer')
    thinking = json_mode and thinking_enabled()
    if thinking:
        max_tokens = max(max_tokens, thinking_token_floor())
    if provider == 'qwen':
        options = {'max_tokens': max_tokens, 'temperature': 0.0 if temperature is None else temperature}
        if json_mode:
            options['enable_thinking'] = thinking
    else:
        token_parameter = os.getenv('SEEGROUND_OPENAI_TOKEN_PARAMETER', 'max_completion_tokens').strip()
        if token_parameter not in ('max_completion_tokens', 'max_tokens'):
            raise QwenAPIError('SEEGROUND_OPENAI_TOKEN_PARAMETER must be max_completion_tokens or max_tokens')
        options = {token_parameter: max_tokens, 'reasoning_effort': 'medium'}
    if 'temperature' in options and not 0 <= options['temperature'] <= 2:
        raise QwenAPIError('temperature must be finite and between 0 and 2')
    if response_schema is not None:
        if not json_mode or response_schema.get('strict') is not True:
            raise QwenAPIError('Strict response_schema requires json_mode=True')
        from jsonschema import Draft202012Validator
        Draft202012Validator.check_schema(response_schema['schema'])
        # Both providers use JSON mode; the full schema is enforced locally.
        options['response_format'] = {'type': 'json_object'}
    elif json_mode:
        options['response_format'] = {'type': 'json_object'}
    return options


def effective_timeout(timeout: tuple[int, int]) -> tuple[int, int]:
    if api_provider() != 'openai':
        return timeout
    raw = os.getenv('SEEGROUND_OPENAI_READ_TIMEOUT_SECONDS', '240').strip()
    try:
        read_timeout = int(raw)
    except ValueError as exc:
        raise QwenAPIError('SEEGROUND_OPENAI_READ_TIMEOUT_SECONDS must be a positive integer') from exc
    if read_timeout < 1:
        raise QwenAPIError('SEEGROUND_OPENAI_READ_TIMEOUT_SECONDS must be a positive integer')
    return timeout[0], max(timeout[1], read_timeout)


def api_config_summary() -> dict[str, Any]:
    """Include parameter choices in batch provenance without including secrets."""
    options = request_options(512, None, True)
    options.pop('max_tokens', None)
    options.pop('max_completion_tokens', None)
    from v4_policy import v62_enabled
    structured_outputs = False
    return {'provider': api_provider(), 'request_options': options,
            **({'thinking_token_floor': thinking_token_floor()} if thinking_enabled() else {}),
            'structured_outputs': structured_outputs,
            'local_schema_validation': v62_enabled(),
            'format_attempt_limit': 4,
            'token_parameter': ('max_tokens' if api_provider() == 'qwen' else
                os.getenv('SEEGROUND_OPENAI_TOKEN_PARAMETER', 'max_completion_tokens').strip()),
            'allow_http': os.getenv('SEEGROUND_ALLOW_HTTP', '') == '1',
            'api_concurrency': api_concurrency(),
            'api_slot_dir': str(api_slot_dir())}


def _settings() -> tuple[str, str, str]:
    base = os.getenv("SEEGROUND_BASE_URL", "").strip().rstrip("/")
    model = os.getenv("SEEGROUND_MODEL", "").strip()
    key = os.getenv("SEEGROUND_API_KEY", "").strip()
    if not base or not model or not key:
        raise QwenAPIError("请配置 SEEGROUND_BASE_URL、SEEGROUND_MODEL 和 SEEGROUND_API_KEY")
    api_provider()
    address = urlsplit(base)
    if (address.scheme not in ('https', 'http') or not address.hostname or
            address.username or address.password or address.query or address.fragment or
            not address.path.rstrip('/').endswith('/v1')):
        raise QwenAPIError('SEEGROUND_BASE_URL must be an API base ending in /v1, without credentials or query')
    if (address.scheme == 'http' and address.hostname not in ('localhost', '127.0.0.1', '::1')
            and os.getenv('SEEGROUND_ALLOW_HTTP', '') != '1'):
        raise QwenAPIError('HTTP API requires explicit SEEGROUND_ALLOW_HTTP=1; HTTPS is preferred')
    return base, model, key


def image_part(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with Image.open(path) as im:
        mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}.get(im.format)
        im.verify()
    if mime is None:
        raise QwenAPIError("图片必须是 PNG、JPEG 或 WEBP")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def _redact(value: Any, *, limit: int | None = 4000) -> Any:
    """Keep diagnostics useful without persisting credentials or image payloads."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            key_text = re.sub(r'[^a-z0-9]', '', str(key).lower())
            if key_text in {'authorization', 'proxyauthorization', 'key', 'token',
                            'secret', 'password', 'passwd', 'credential', 'credentials',
                            'privatekey'} or key_text.endswith((
                                'apikey', 'authtoken', 'accesstoken', 'refreshtoken',
                                'apitoken', 'apisecret', 'idtoken', 'sessiontoken', 'bearertoken', 'clientsecret',
                                'subscriptionkey', 'secretkey')):
                result[str(key)] = '[REDACTED]'
            elif key_text in {'imageurl', 'image', 'b64json', 'imagebase64',
                              'base64image', 'base64', 'b64', 'imagedata', 'imagebytes',
                              'imagecontent', 'encodedimage'}:
                result[str(key)] = '[IMAGE_REDACTED]'
            else:
                result[str(key)] = _redact(item, limit=limit)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact(item, limit=limit) for item in value]
    if isinstance(value, str):
        try:
            structured = json.loads(value)
        except (ValueError, TypeError):
            structured = None
        if isinstance(structured, (dict, list)):
            sanitized = _redact(structured, limit=limit)
            if sanitized != structured:
                value = json.dumps(sanitized, ensure_ascii=False)
        text = re.sub(r'data:image/[^,\s]+;base64,\s*[A-Za-z0-9+/_=-]+'
                      r'(?:[ \t]*[\r\n]+[ \t]*[A-Za-z0-9+/_=-]+)*',
                      '[IMAGE_REDACTED]', value, flags=re.I)
        # Upstream error bodies may echo an image without its field name or
        # data URI.  Redact long encoded tokens, including folded MIME lines,
        # before truncation so no retained prefix contains image content.
        text = re.sub(r'(?<![A-Za-z0-9+/_=-])(?:[A-Za-z0-9+/_-]{128,}={0,2}|'
                      r'(?:[A-Za-z0-9+/_-]{64,}={0,2}[ \t]*[\r\n]+[ \t]*)+'
                      r'[A-Za-z0-9+/_-]+={0,2})(?![A-Za-z0-9+/_=-])',
                      '[IMAGE_REDACTED]', text)
        text = re.sub(r'(?i)(bearer\s+)[^\s,;]+', r'\1[REDACTED]', text)
        text = re.sub(r'(?i)(["\']?(?:(?:x[_-])?(?:api[_-]?(?:key|token|secret)|auth[_-]?token)|'
                      r'(?:proxy[_-]?)?authorization|(?:access|refresh|id|session|bearer)[_-]?token|'
                      r'token|(?:client[_-]?)?secret|password|passwd|credentials?|'
                      r'private[_-]?key|subscription[_-]?key|secret[_-]?key)["\']?\s*[:=]\s*)'
                      r'(?:"[^"\r\n]*"|\'[^\'\r\n]*\'|[^\s,;}]+)',
                      r'\1[REDACTED]', text)
        for variable in ('SEEGROUND_API_KEY', 'TYPESAFE_API_KEY'):
            secret = os.getenv(variable, '').strip()
            if secret:
                text = text.replace(secret, '[REDACTED]')
        return (text[:limit] + ('...[TRUNCATED]' if len(text) > limit else '')
                if limit is not None else text)
    return value


def _request_summary(base: str, model: str, messages: list[dict[str, Any]],
                     options: dict[str, Any]) -> dict[str, Any]:
    return {
        'endpoint': _redact(base + '/chat/completions'),
        'model': model,
        'provider': api_provider(),
        'message_roles': [item.get('role') for item in messages if isinstance(item, dict)],
        'message_count': len(messages),
        'message_chars': len(json.dumps(_redact(messages), ensure_ascii=False)),
        'contains_images': 'image_url' in json.dumps(messages, ensure_ascii=False),
        **options,
    }


def _request_id(response: Any) -> str | None:
    headers = getattr(response, 'headers', {}) or {}
    for name in ('x-request-id', 'request-id', 'x-amzn-requestid', 'x-correlation-id'):
        value = headers.get(name) or headers.get(name.title())
        if value:
            return _redact(str(value), limit=256)
    try:
        body = response.json()
    except (ValueError, TypeError, AttributeError):
        body = None
    if isinstance(body, dict):
        value = body.get('request_id') or body.get('requestId')
        if value:
            return _redact(str(value), limit=256)
    return None


def repair_missing_nullable_fields(payload: dict[str, Any], schema: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Recursively fill required fields the gateway dropped when the model
    returns a partial object. Nullable fields get None; array fields get []
    (matching the prompt instruction to use empty arrays for unused fields);
    nested objects are repaired recursively. No decision content invented."""
    repaired = []
    def fill(value: Any, sch: dict[str, Any]) -> tuple[Any, bool]:
        if not isinstance(value, dict):
            return value, False
        changed = False
        for field in sch.get('required', []):
            if field in value:
                nested = sch.get('properties', {}).get(field)
                if nested and nested.get('type') == 'object' and isinstance(value[field], dict):
                    value[field], sub = fill(value[field], nested)
                    changed = changed or sub
                elif nested and nested.get('type') == 'array' and isinstance(value[field], list):
                    item_schema = nested.get('items', {})
                    if item_schema.get('type') == 'object':
                        for item in value[field]:
                            if isinstance(item, dict):
                                fill(item, item_schema)
                continue
            field_schema = sch.get('properties', {}).get(field)
            if field_schema is None:
                continue
            allows_null = field_schema.get('nullable') is True or 'null' in field_schema.get('type', []) \
                or any(sub.get('type') == 'null' for sub in field_schema.get('anyOf', []))
            if allows_null:
                value[field] = None
                repaired.append(field)
                changed = True
            elif field_schema.get('type') == 'array':
                value[field] = []
                repaired.append(field)
                changed = True
        return value, changed
    fill(payload, schema)
    return payload, repaired


def normalize_schema_payload(name: str, payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Map equivalent provider action fields into the local ReAct contract."""
    if not isinstance(payload, dict):
        return payload, []

    value = dict(payload)
    repaired = []
    if name in ('scene_grounding', 'scene_consistency_review'):
        if 'predicted_id' not in value and 'target_id' in value:
            value['predicted_id'] = value['target_id']
            repaired.append('target_id->predicted_id')
        if 'reason' not in value and isinstance(value.get('reasoning'), str):
            value['reason'] = value['reasoning']
            repaired.append('reasoning->reason')
        if repaired:
            value.pop('target_id', None)
            value.pop('reasoning', None)
        return value, repaired
    if name != 'r18_react_action':
        return payload, []
    parameters = value.get('parameters')
    if not isinstance(parameters, dict) and isinstance(value.get('params'), dict):
        parameters = value['params']
        repaired.append('params->parameters')
    if isinstance(parameters, dict):
        for field in ('reason', 'focus_ids', 'context', 'azimuth_deg', 'elevation_deg',
                      'frame', 'predicted_id', 'evidence', 'unresolved'):
            if field not in value and field in parameters:
                value[field] = parameters[field]
                repaired.append(f'parameters.{field}->{field}')
        for alias in ('ids', 'object_ids'):
            if 'focus_ids' not in value and alias in parameters:
                value['focus_ids'] = parameters[alias]
                repaired.append(f'parameters.{alias}->focus_ids')
                break
        for alias in ('render_mode', 'view', 'mode'):
            if 'context' not in value and alias in parameters:
                value[alias] = parameters[alias]
                repaired.append(f'parameters.{alias}->context')
                break
        if 'context' not in value and isinstance(parameters.get('isolated'), bool):
            value['context'] = 'isolated' if parameters['isolated'] else 'scene'
            repaired.append('parameters.isolated->context')

    if 'focus_ids' not in value:
        for alias in ('ids', 'object_ids'):
            if alias in value:
                value['focus_ids'] = value[alias]
                repaired.append(f'{alias}->focus_ids')
                break

    isolated = value.get('isolated')
    if isinstance(isolated, bool):
        context = 'isolated' if isolated else 'scene'
        if value.get('context') is None:
            value['context'] = context
            repaired.append('isolated->context')
        elif value.get('context') == context:
            repaired.append('duplicate isolated context removed')

    context_aliases_to_remove = []
    for alias in ('view', 'mode', 'render_mode'):
        context = value.get(alias)
        if isinstance(context, str):
            context = context.strip().lower().strip('`\"\'')
        if context not in ('scene', 'isolated'):
            continue
        if value.get('context') is None:
            value['context'] = context
            repaired.append(f'{alias}->context')
            context_aliases_to_remove.append(alias)
        elif value.get('context') == context:
            repaired.append(f'duplicate {alias} context removed')
            context_aliases_to_remove.append(alias)

    if 'action' not in value:
        for alias in ('action_type', 'tool'):
            if isinstance(value.get(alias), str):
                action = value[alias].strip().lower()
                action = {'render': 'inspect', 'view': 'inspect', 'observe': 'inspect',
                          'measurement': 'measure', 'measurements': 'measure',
                          'answer': 'finish', 'select': 'finish', 'choose': 'finish'}.get(action, action)
                value['action'] = action
                repaired.append(f'{alias}->action')
                break

    if isinstance(value.get('action'), str):
        action = value['action'].strip().lower()
        canonical = {'render': 'inspect', 'view': 'inspect', 'observe': 'inspect',
                     'measurement': 'measure', 'measurements': 'measure',
                     'answer': 'finish', 'select': 'finish', 'choose': 'finish'}.get(action, action)
        if canonical != value['action']:
            value['action'] = canonical
            repaired.append('action alias canonicalized')

    reason = value.get('reason')
    if value.get('action') in ('inspect', 'measure', 'finish') and (
            not isinstance(reason, str) or not reason.strip()):
        value['reason'] = 'Required reason omitted; the model-selected action and parameters were retained.'
        repaired.append('missing reason filled with format-repair note')

    for alias in context_aliases_to_remove:
        value.pop(alias, None)

    if repaired:
        for alias in ('action_type', 'tool', 'parameters', 'params', 'isolated', 'ids', 'object_ids'):
            value.pop(alias, None)
    return value, repaired


def _balanced_json_objects(content: str) -> list[dict[str, Any]]:
    """Extract JSON objects embedded in prose or tool-like text."""
    objects = []
    for start, char in enumerate(content):
        if char != '{':
            continue
        depth = 0
        quoted = False
        escaped = False
        for end in range(start, len(content)):
            current = content[end]
            if quoted:
                if escaped:
                    escaped = False
                elif current == '\\':
                    escaped = True
                elif current == '"':
                    quoted = False
                continue
            if current == '"':
                quoted = True
            elif current == '{':
                depth += 1
            elif current == '}':
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(content[start:end + 1])
                    except (ValueError, TypeError):
                        break
                    if isinstance(value, dict):
                        objects.append(value)
                    break
    return objects


def _number_list(value: str) -> list[int]:
    return [int(item) for item in re.findall(r'(?<![\w.])-?\d+(?![\w.])', value)]


def parse_r18_action_content(content: str) -> dict[str, Any] | None:
    """Recover an r18 action when an OpenAI-compatible relay ignores JSON mode."""
    if not isinstance(content, str) or not content.strip():
        return None
    objects = _balanced_json_objects(content)
    action_match = re.search(r'\b(inspect|measure|finish|render|view|observe|select|choose|answer)\b',
                             content, re.IGNORECASE)
    action = action_match.group(1).lower() if action_match else None
    action = {'render': 'inspect', 'view': 'inspect', 'observe': 'inspect',
              'select': 'finish', 'choose': 'finish', 'answer': 'finish'}.get(action, action)
    if objects:
        value = dict(next((item for item in reversed(objects)
                           if any(key in item for key in ('action', 'action_type', 'tool'))),
                          max(objects, key=lambda item: len(item))))
        if action and not any(key in value for key in ('action', 'action_type', 'tool')):
            value['action'] = action
        return value

    if not action:
        return None
    value: dict[str, Any] = {'action': action}
    focus = None
    bracket = re.search(r'\b(?:focus_ids|object_ids|ids)\s*[:=]?\s*\[([^\]]*)\]', content, re.IGNORECASE)
    if bracket:
        focus = _number_list(bracket.group(1))
    elif action in ('inspect', 'measure'):
        call = re.search(r'\b(?:inspect|measure)\s*\(([^)]*)\)', content, re.IGNORECASE)
        if call:
            positional = call.group(1).split(',')
            numbers = []
            for item in positional:
                if '=' in item:
                    break
                numbers.extend(_number_list(item))
            focus = numbers
    if focus is not None:
        value['focus_ids'] = focus
    if action in ('inspect', 'measure') and not focus:
        return None

    context = re.search(r'\b(?:context|render_mode|mode|view)\s*[:=]\s*[\"\']?\s*(scene|isolated)\b',
                        content, re.IGNORECASE)
    if context:
        value['context'] = context.group(1).lower()
    else:
        isolated = re.search(r'\bisolated\s*[:=]\s*(true|false)\b', content, re.IGNORECASE)
        scene = re.search(r'\bscene\s*[:=]\s*(true|false)\b', content, re.IGNORECASE)
        if isolated:
            value['context'] = 'isolated' if isolated.group(1).lower() == 'true' else 'scene'
        elif scene:
            value['context'] = 'scene' if scene.group(1).lower() == 'true' else 'isolated'

    for field in ('azimuth_deg', 'elevation_deg'):
        match = re.search(rf'\b{field}\s*[:=]\s*[\"\']?(-?\d+(?:\.\d+)?)', content, re.IGNORECASE)
        if match:
            value[field] = float(match.group(1))
            if value[field].is_integer():
                value[field] = int(value[field])

    if action == 'finish':
        match = re.search(r'\b(?:predicted[_ ]?id|object[_ ]?id)\s*[:=]?\s*[\"\']?(\d+)', content, re.IGNORECASE)
        if match is None:
            match = re.search(r'\b(?:answer|choose|select|final)\b[^\n]{0,40}?\b(?:id|object)\s*[#:=]?\s*(\d+)',
                              content, re.IGNORECASE)
        if match:
            value['predicted_id'] = int(match.group(1))
    return value


def parse_scene_grounding_content(content: str) -> dict[str, Any] | None:
    """Recover a conservative scene decision from a non-JSON relay response."""
    if not isinstance(content, str) or not content.strip():
        return None
    text = content.strip()
    match = re.search(r'\b(?:predicted[_ ]?id|target[_ ]?id|object[_ ]?id|answer|id)\b[^\d]{0,24}(\d+)',
                      text, re.IGNORECASE)
    if match is None:
        match = re.search(r'\b(?:target|answer|choice|selected)\b\s*[:#-]?\s*(?:object\s*)?(\d+)',
                          text, re.IGNORECASE)
    if match is None and re.fullmatch(r'\s*\d+\s*', text):
        match = re.search(r'\d+', text)
    if match is None:
        return None
    number = match.group(1) if match.lastindex else match.group(0)
    return {'predicted_id': int(number),
            'reason': text[:1000], 'evidence': [],
            'unresolved': ['Relay returned no structured evidence; final validation may abstain.']}


def _schema_messages(messages: list[dict[str, Any]], schema: dict[str, Any],
                     error: str | None = None) -> list[dict[str, Any]]:
    """Include the output contract in text for compatible gateways."""
    instruction = ('Return ONLY one JSON object matching this JSON Schema. '
        'No markdown fences or prose. Include every required field. '
        'Use null only where allowed. Do not invent evidence. JSON Schema: '
        + json.dumps(schema['schema'], ensure_ascii=False, separators=(',', ':')))
    if error is not None:
        instruction += '\nThe previous response failed validation: ' + error[:1500]
        instruction += '\nAnswer the original task again using the schema above.'
    return [*messages, {'role': 'user', 'content': instruction}]


def _chat_schema_retry(messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Allow four format attempts and one relay transport retry."""
    schema = kwargs.get('response_schema')
    request_messages = _schema_messages(messages, schema) if schema is not None else messages
    attempts = []
    started = time.perf_counter()
    schema_attempt = 0
    transport_retries = 0
    while True:
        attempt_start = time.perf_counter()
        try:
            data = _chat_once(request_messages, **kwargs)
            attempts.append({'status': 'success', 'latency_s': round(time.perf_counter() - attempt_start, 4),
                             'usage': data.get('usage')})
            data['_baseline'].update(api_attempts=attempts, extra_attempts=len(attempts) - 1,
                schema_retry=schema_attempt, transport_retries=transport_retries,
                latency_s=round(time.perf_counter() - started, 4))
            return data
        except QwenAPIError as exc:
            diagnostic = exc.diagnostic
            attempts.append({'status': 'failed', 'latency_s': round(time.perf_counter() - attempt_start, 4),
                             'diagnostic': dict(diagnostic)})
            diagnostic.update(api_attempts=attempts, extra_attempts=len(attempts) - 1,
                              transport_retries=transport_retries)
            failure_kind = diagnostic.get('failure_kind')
            if (api_provider() == 'openai' and transport_retries < 1
                    and failure_kind in {'timeout', 'network', 'http_4xx', 'http_5xx'}
                    and 'insufficient_quota' not in str(diagnostic.get('response_body', ''))
                    and diagnostic.get('http_status') in (None, 408, 429, 500, 502, 503, 504)):
                transport_retries += 1
                if diagnostic.get('http_status') == 429:
                    time.sleep(30)  # Relay rate limits are per-minute; wait out the window.
                continue
            if (schema is None or failure_kind != 'response_schema' or schema_attempt >= 3
                    or diagnostic.get('validation_error') == 'Model refused'):
                raise
            schema_attempt += 1
            request_messages = _schema_messages(messages, schema,
                str(diagnostic.get('validation_error', 'Expected a JSON object')))


def _chat_once(messages: list[dict[str, Any]], *, max_tokens: int = 512, temperature: float | None = None,
         timeout: tuple[int, int] = (15, 120), json_mode: bool = False,
         response_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    base, model, key = _settings()
    timeout = effective_timeout(timeout)
    options = request_options(max_tokens, temperature, json_mode, response_schema)
    payload = {"model": model, "messages": messages, **options}
    summary = _request_summary(base, model, messages, options)
    summary['timeout_seconds'] = {'connect': timeout[0], 'read': timeout[1]}
    label = 'Qwen' if api_provider() == 'qwen' else 'OpenAI-compatible'
    started = time.perf_counter()
    try:
        with api_request_slot() as slot_wait_s:
            response = requests.post(
                base + "/chat/completions",
                headers={"Authorization": "Bearer " + key},
                json=payload,
                timeout=timeout,
            )
    except requests.Timeout as exc:
        raise QwenAPIError(f"{label} API 请求超时", diagnostic={
            'failure_kind': 'timeout', 'http_status': None, 'response_body': None,
            'request_id': None, 'model': model, 'request_summary': summary, 'stage': 'http_request',
        }) from exc
    except requests.RequestException as exc:
        raise QwenAPIError(f"{label} API 网络连接失败", diagnostic={
            'failure_kind': 'network', 'http_status': None, 'response_body': None,
            'request_id': None, 'model': model, 'request_summary': summary, 'stage': 'http_request',
        }) from exc
    if not response.ok:
        status = int(getattr(response, 'status_code', 0) or 0)
        kind = 'http_4xx' if 400 <= status < 500 else 'http_5xx' if status >= 500 else 'http_error'
        body = getattr(response, 'text', '') or ''
        raise QwenAPIError(f"{label} API 返回 HTTP {status}", diagnostic={
            'failure_kind': kind, 'http_status': status,
            'response_body': _redact(body), 'request_id': _request_id(response),
            'model': model, 'request_summary': summary, 'stage': 'http_response',
        })
    try:
        data = response.json()
        content = data["choices"][0]["message"].get("content") or ""
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        body = getattr(response, 'text', '') or ''
        raise QwenAPIError(f"{label} API 返回格式不正确", diagnostic={
            'failure_kind': 'response_parse', 'http_status': int(getattr(response, 'status_code', 200) or 200),
            'response_body': _redact(body), 'request_id': _request_id(response),
            'model': model, 'request_summary': summary, 'stage': 'response_parse',
        }) from exc
    if not isinstance(content, str):
        raise QwenAPIError(f"{label} API 的回答必须是文本", diagnostic={
            'failure_kind': 'response_parse', 'http_status': int(getattr(response, 'status_code', 200) or 200),
            'response_body': _redact(getattr(response, 'text', '') or ''),
            'request_id': _request_id(response), 'model': model, 'request_summary': summary,
            'stage': 'response_parse',
        })
    if response_schema is not None:
        from jsonschema import Draft202012Validator, ValidationError
        payload = None
        response_normalization = []
        parse_error = None
        choice = data['choices'][0]
        if choice['message'].get('refusal') or choice.get('finish_reason') != 'stop':
            raise QwenAPIError(f'{label} API 结构化回答被拒绝或未完整结束', diagnostic={
                'failure_kind': 'response_schema', 'http_status': response.status_code,
                'response_body': _redact(getattr(response, 'text', '') or content),
                'request_id': _request_id(response), 'model': model, 'request_summary': summary,
                'schema_name': response_schema['name'], 'finish_reason': choice.get('finish_reason'),
                'validation_error': 'Model refused' if choice['message'].get('refusal') else 'Incomplete response',
                'stage': 'response_schema_complete',
            })
        try:
            payload = json.loads(content)
        except ValueError as exc:
            payload = None
            parse_error = str(exc)
        if not isinstance(payload, dict) and response_schema.get('name') == 'r18_react_action':
            payload = parse_r18_action_content(content)
        if not isinstance(payload, dict) and response_schema.get('name') in ('scene_grounding', 'scene_consistency_review'):
            payload = parse_scene_grounding_content(content)
        if payload is None:
            import re as _re
            _match = _re.search(r'\{.*\}', content, _re.DOTALL)
            if _match:
                try:
                    payload = json.loads(_match.group(0))
                    if not isinstance(payload, dict):
                        payload = None
                except ValueError:
                    try:
                        from json_repair import loads as json_repair_loads
                        payload = json_repair_loads(_match.group(0))
                        if not isinstance(payload, dict):
                            payload = None
                    except (ImportError, ValueError, TypeError):
                        payload = None
        if payload is None:
            raise QwenAPIError(f'{label} API 回答不符合严格 schema', diagnostic={
                'failure_kind': 'response_schema', 'http_status': response.status_code,
                'response_body': _redact(getattr(response, 'text', '') or content),
                'request_id': _request_id(response), 'model': model, 'request_summary': summary,
                'schema_name': response_schema['name'],
                'validation_error': parse_error or 'content is not a JSON object',
                'stage': 'response_schema_parse',
            })
        try:
            if isinstance(payload, dict):
                payload, response_normalization = normalize_schema_payload(
                    response_schema['name'], payload)
                payload, _repaired = repair_missing_nullable_fields(payload, response_schema['schema'])
                content = json.dumps(payload, ensure_ascii=False)
            Draft202012Validator(response_schema['schema']).validate(payload)
            if (response_schema['name'] == 'r18_react_action'
                    and payload.get('action') in ('inspect', 'measure')
                    and not payload.get('focus_ids')):
                raise ValidationError('inspect/measure require explicit nonempty focus_ids')
            if (response_schema['name'] == 'r18_react_action'
                    and payload.get('action') == 'finish' and payload.get('predicted_id') is not None
                    and not any(payload['predicted_id'] in row.get('object_ids', [])
                                for row in payload.get('evidence', []))):
                raise ValidationError('Final choice requires recorded evidence about the selected object')
        except (ValueError, ValidationError) as exc:
            raise QwenAPIError(f'{label} API 回答不符合严格 schema', diagnostic={
                'failure_kind': 'response_schema', 'http_status': response.status_code,
                'response_body': _redact(getattr(response, 'text', '') or content),
                'request_id': _request_id(response), 'model': model, 'request_summary': summary,
                'schema_name': response_schema['name'], 'validation_error': _redact(str(exc)),
                'stage': 'response_schema_validate',
            }) from exc
    if api_provider() == 'openai' and json_mode:
        try:
            if not isinstance(json.loads(content), dict):
                raise ValueError('JSON response must be an object')
        except ValueError as exc:
            raise QwenAPIError(f"{label} API 的 JSON 回答无法解析为对象", diagnostic={
                'failure_kind': 'response_parse',
                'http_status': int(getattr(response, 'status_code', 200) or 200),
                'response_body': _redact(getattr(response, 'text', '') or content),
                'request_id': _request_id(response), 'model': model, 'request_summary': summary,
                'stage': 'response_json_parse',
            }) from exc
    data["_baseline"] = {"latency_s": round(time.perf_counter() - started, 4), "api_slot_wait_s": slot_wait_s,
                         "api_concurrency": api_concurrency(), "model": model,
                         'provider': api_provider(), 'request_summary': summary,
                         **({'response_normalization': response_normalization}
                            if response_schema is not None else {}),
                         **({'response_format': options['response_format']} if 'response_format' in options else {}),
                         'http_status': getattr(response, 'status_code', 200),
                         'request_id': _request_id(response)}
    data["_content"] = content
    data['_finish_reason'] = data['choices'][0].get('finish_reason')
    return data


def chat(messages: list[dict[str, Any]], *, max_tokens: int = 512, temperature: float | None = None,
         timeout: tuple[int, int] = (15, 120), json_mode: bool = False,
         response_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    from v4_policy import v5_enabled
    if not v5_enabled():
        return _chat_schema_retry(messages, max_tokens=max_tokens, temperature=temperature, timeout=timeout, json_mode=json_mode,
                          response_schema=response_schema)
    from v5_runtime import sample_recovery, retryable, persist_api_recovery
    budget = sample_recovery.get()
    if budget is None:
        budget = {'used': False, 'attempts': []}
    import hashlib
    request_hash = hashlib.sha256(json.dumps({'messages': messages, 'max_tokens': max_tokens,
        'temperature': temperature, 'json_mode': json_mode,
        'model': _settings()[1],
        'request_options': request_options(max_tokens, temperature, json_mode, response_schema),
        **({'response_schema': response_schema} if response_schema is not None else {})}, sort_keys=True).encode()).hexdigest()
    if budget.get('resuming') and request_hash in budget.get('completed_requests', {}):
        from copy import deepcopy
        cached = deepcopy(budget['completed_requests'][request_hash])
        cached['usage'] = None
        cached['_baseline'].update(cache_hit=True, api_attempts=[], extra_attempts=0, latency_s=0)
        return cached
    if budget.pop('manual_resume_pending', False):
        failure = budget.get('failed_request', {})
        if failure.get('hash') != request_hash or not retryable(failure.get('diagnostic', {})):
            raise QwenAPIError('Saved failed request is not eligible for stage recovery')
        budget['used'] = True
        budget['attempts'].append({'reason': 'stage_resume', 'context_compressed': False})
        persist_api_recovery(budget)
    attempts = []
    started = time.perf_counter()
    while True:
        attempt_start = time.perf_counter()
        try:
            data = _chat_schema_retry(messages, max_tokens=max_tokens, temperature=temperature, timeout=timeout, json_mode=json_mode,
                              response_schema=response_schema)
            attempts.extend(data['_baseline']['api_attempts'])
            data['_baseline'].update(api_attempts=attempts, extra_attempts=len(attempts) - 1,
                                      latency_s=round(time.perf_counter() - started, 4))
            if budget.get('api_checkpoint'):
                budget.setdefault('completed_requests', {})[request_hash] = data
                persist_api_recovery(budget)
            return data
        except QwenAPIError as exc:
            diagnostic = exc.diagnostic
            attempts.extend(diagnostic.pop('api_attempts'))
            context_error = diagnostic.get('http_status') == 400 and any(marker in
                str(diagnostic.get('response_body', '')).lower() for marker in
                ('context window', 'context_length_exceeded', 'maximum context length'))
            retry_messages = messages
            if context_error:
                from react_decision import compress_planner_messages
                retry_messages = compress_planner_messages(messages)
            changed = retry_messages != messages
            if not budget['used'] and (retryable(diagnostic) or context_error and changed):
                budget['used'] = True
                budget['attempts'].append({'reason': diagnostic.get('failure_kind'), 'context_compressed': changed})
                persist_api_recovery(budget)
                if diagnostic.get('http_status') == 429:
                    time.sleep(20)  # Relay rate limits are per-minute; back off before the single retry.
                messages = retry_messages
                continue
            diagnostic.update(api_attempts=attempts, extra_attempts=len(attempts) - 1,
                              latency_s=round(time.perf_counter() - started, 4))
            budget['failed_request'] = {'hash': request_hash, 'diagnostic': dict(diagnostic)}
            persist_api_recovery(budget)
            raise


def parse_predicted_id(content: str, valid_ids: set[int]) -> int:
    patterns = [r"predicted\s*id\s*[:：]\s*(\d+)", r"object\s*id\s*[:：]\s*(\d+)", r"\bid\s*[:：]\s*(\d+)"]
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            object_id = int(match.group(1))
            if object_id not in valid_ids:
                raise QwenAPIError("Qwen 返回了不在候选清单中的编号")
            return object_id
    raise QwenAPIError("Qwen 未返回可解析的候选编号")


def parse_json_object(content: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise QwenAPIError("Qwen 未返回 JSON 对象")
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise QwenAPIError("Qwen 返回的 JSON 无法解析") from exc
    if not isinstance(value, dict):
        raise QwenAPIError("Qwen 返回的解析结果不是对象")
    return value
