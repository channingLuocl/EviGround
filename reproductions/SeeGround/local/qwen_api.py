"""Small OpenAI-compatible Qwen client used by the local baseline."""
from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests


class QwenAPIError(ValueError):
    """Raised when the configured Qwen endpoint cannot produce a valid answer."""


def _settings() -> tuple[str, str, str]:
    base = os.getenv("SEEGROUND_BASE_URL", "").strip().rstrip("/")
    model = os.getenv("SEEGROUND_MODEL", "").strip()
    key = os.getenv("SEEGROUND_API_KEY", "").strip()
    if not base or not model or not key:
        raise QwenAPIError("请配置 SEEGROUND_BASE_URL、SEEGROUND_MODEL 和 SEEGROUND_API_KEY")
    if not base.startswith(("https://", "http://localhost:")):
        raise QwenAPIError("云端 API 地址必须使用 HTTPS")
    return base, model, key


def image_part(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(path.suffix.lower())
    if mime is None:
        raise QwenAPIError("图片必须是 PNG、JPEG 或 WEBP")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def chat(messages: list[dict[str, Any]], *, max_tokens: int = 512, temperature: float = 0.0, timeout: tuple[int, int] = (15, 120)) -> dict[str, Any]:
    base, model, key = _settings()
    started = time.perf_counter()
    try:
        response = requests.post(
            base + "/chat/completions",
            headers={"Authorization": "Bearer " + key},
            json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise QwenAPIError("Qwen API 网络连接失败") from exc
    if not response.ok:
        raise QwenAPIError(f"Qwen API 返回 HTTP {response.status_code}")
    try:
        data = response.json()
        content = data["choices"][0]["message"].get("content") or ""
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise QwenAPIError("Qwen API 返回格式不正确") from exc
    data["_baseline"] = {"latency_s": round(time.perf_counter() - started, 4), "model": model}
    data["_content"] = content
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
