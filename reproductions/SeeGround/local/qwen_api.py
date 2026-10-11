"""Configured Chat Completions client for Qwen and OpenAI-compatible providers."""
from __future__ import annotations

import base64
import json
import os
import re
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


def api_provider() -> str:
    provider = os.getenv('SEEGROUND_API_PROVIDER', 'qwen').strip().lower()
    if provider not in ('qwen', 'openai'):
        raise QwenAPIError('SEEGROUND_API_PROVIDER must be qwen or openai')
    return provider


def request_options(max_tokens: int, temperature: float | None, json_mode: bool) -> dict[str, Any]:
    provider = api_provider()
    if type(max_tokens) is not int or max_tokens < 1:
        raise QwenAPIError('max_tokens must be a positive integer')
    if provider == 'qwen':
        options = {'max_tokens': max_tokens, 'temperature': 0.0 if temperature is None else temperature}
        if json_mode:
            options['enable_thinking'] = False
    else:
        token_parameter = os.getenv('SEEGROUND_OPENAI_TOKEN_PARAMETER', 'max_completion_tokens').strip()
        if token_parameter not in ('max_completion_tokens', 'max_tokens'):
            raise QwenAPIError('SEEGROUND_OPENAI_TOKEN_PARAMETER must be max_completion_tokens or max_tokens')
        options = {token_parameter: max_tokens}
        configured_temperature = os.getenv('SEEGROUND_OPENAI_TEMPERATURE', '').strip()
        if temperature is None and configured_temperature:
            try:
                temperature = float(configured_temperature)
            except ValueError as exc:
                raise QwenAPIError('SEEGROUND_OPENAI_TEMPERATURE must be a number') from exc
        if temperature is not None:
            options['temperature'] = temperature
    if 'temperature' in options and not 0 <= options['temperature'] <= 2:
        raise QwenAPIError('temperature must be finite and between 0 and 2')
    if json_mode:
        options['response_format'] = {'type': 'json_object'}
    return options


def api_config_summary() -> dict[str, Any]:
    """Include parameter choices in batch provenance without including secrets."""
    options = request_options(512, None, True)
    options.pop('max_tokens', None)
    options.pop('max_completion_tokens', None)
    return {'provider': api_provider(), 'request_options': options,
            'token_parameter': ('max_tokens' if api_provider() == 'qwen' else
                os.getenv('SEEGROUND_OPENAI_TOKEN_PARAMETER', 'max_completion_tokens').strip()),
            'allow_http': os.getenv('SEEGROUND_ALLOW_HTTP', '') == '1'}


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


def chat(messages: list[dict[str, Any]], *, max_tokens: int = 512, temperature: float | None = None,
         timeout: tuple[int, int] = (15, 120), json_mode: bool = False) -> dict[str, Any]:
    base, model, key = _settings()
    options = request_options(max_tokens, temperature, json_mode)
    payload = {"model": model, "messages": messages, **options}
    summary = _request_summary(base, model, messages, options)
    label = 'Qwen' if api_provider() == 'qwen' else 'OpenAI-compatible'
    started = time.perf_counter()
    try:
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
    data["_baseline"] = {"latency_s": round(time.perf_counter() - started, 4), "model": model,
                         'provider': api_provider(), 'request_summary': summary,
                         'http_status': getattr(response, 'status_code', 200),
                         'request_id': _request_id(response)}
    data["_content"] = content
    data['_finish_reason'] = data['choices'][0].get('finish_reason')
    return data


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
