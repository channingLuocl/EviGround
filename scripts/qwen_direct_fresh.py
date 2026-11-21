"""Qwen native ReAct episode with optional text-only visual reporting."""
from __future__ import annotations
import argparse
import base64
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import distributions
from graphlib import CycleError, TopologicalSorter
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path
import shlex
import sys
import time
from urllib.parse import urlsplit
import requests
from qwen_step2_context import original_resolution_pack, compact_call, compact_wire_schema
from qwen_evidence_selector import EvidenceSelectionError, registry as evidence_registry, select_context, validate_visible_references
from qwen_visual_reports import COORDINATOR, WORKER, VisualReports, obj, request_schema
from PIL import Image
from jsonschema import Draft202012Validator, ValidationError, validate

PROJECT = Path(__file__).resolve().parents[1]
INPUTS = PROJECT / "data/qwen_direct"
MODEL = "qwen3.8-max"
BASE_URL = "https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1"
OUTPUT_LIMIT = 32768  # strict local engineering gate; provider usage audited separately
GLM_MODEL = "GLM-4.5V"
GLM_CONTEXT = 65536
# Deployed processor maximum: 9,633,792 pixels / (14 * 2)^2, plus template markers.
GLM_IMAGE_BOUND = 12288 + 16
OUTPUT_RESERVE = 4096
OUTPUT_OVERRUN_ALLOWANCE = 10  # explicit engineering cushion for approximate provider output limits
IMAGE_MAX_PIXELS = 2621440
API_KEY = ""
sys.path.insert(0, str(PROJECT / "src"))
from eviground.evaluation.run_storage import write_json


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(packed(value).encode()).hexdigest()


def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field: " + key)
        result[key] = value
    return result


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def private_qwen_config():
    path = Path.home() / ".config/seeground/laya-joint-20261120/token-plan.env"
    values = {}
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        key, sep, raw = text.removeprefix("export ").partition("=")
        if sep:
            parsed = shlex.split(raw, comments=True)
            values[key.strip()] = parsed[0] if parsed else ""
    if values.get("SEEGROUND_BASE_URL", "").rstrip("/") != BASE_URL:
        raise ValueError("Existing Token Plan configuration has an unexpected base URL")
    token = os.environ.get("EVIGROUND_DIRECT_QWEN_TOKEN_PLAN_KEY") or values.get("SEEGROUND_API_KEY")
    if not token:
        raise ValueError("Existing Token Plan API key is missing")
    return token


def configure_glm(endpoint):
    global MODEL, BASE_URL, OUTPUT_LIMIT
    url = urlsplit(endpoint)
    if (url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost', '::1')
            or url.username is not None or url.query or url.fragment or url.path != '/v1'):
        raise ValueError('GLM requires a loopback vLLM API base ending in /v1')
    MODEL, BASE_URL, OUTPUT_LIMIT = GLM_MODEL, endpoint, GLM_CONTEXT


def check_glm_service():
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(BASE_URL + '/models', timeout=10, allow_redirects=False)
        response.raise_for_status()
        models = response.json()['data']
    if not any(row.get('id') == GLM_MODEL and row.get('max_model_len') == GLM_CONTEXT for row in models):
        raise ValueError(f'vLLM must advertise {GLM_MODEL} with a {GLM_CONTEXT}-token context')


def glm_prompt_count(session, payload):
    data = {key: payload[key] for key in ('model', 'messages', 'chat_template_kwargs')}
    reply = session.post(BASE_URL.removesuffix('/v1') + '/tokenize', json=data,
                         timeout=(10, 45), allow_redirects=False)
    reply.raise_for_status()
    result = reply.json()
    count = result.get('count')
    if type(count) is not int or count < 1 or result.get('max_model_len') != OUTPUT_LIMIT:
        raise ValueError('Invalid GLM tokenizer count or service context changed')
    return count

def redact(value):
    if isinstance(value, dict):
        return {
            key: ("[IMAGE_REDACTED]" if key == "url" and isinstance(item, str)
                  and item.startswith("data:image/") else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return value.replace(API_KEY, "[REDACTED]") if API_KEY else value
    return value

def qwen_preflight(messages, response_format, reserve, allow_legacy_image_fields=False):
    if type(reserve) is not int or reserve <= 0:
        raise ValueError('Positive integer output reservation required')
    clean_messages = deepcopy(messages)
    images = []
    for message in clean_messages:
        if set(message) - {'role', 'content'} or message.get('role') not in ('system', 'user', 'assistant'):
            raise ValueError('Unsupported chat framing fields/role; no uncounted tool/name inputs')
        content = message.get("content")
        if not isinstance(content, (str, list)):
            raise ValueError('Unsupported message content')
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get('type') not in ('text', 'image_url'):
                raise ValueError('Unsupported uncounted media/content part')
            if part['type'] == 'text' and (set(part) != {'type', 'text'} or not isinstance(part['text'], str)):
                raise ValueError('Unsupported text part fields')
            if part.get("type") != "image_url":
                continue
            url_keys = {'url', 'max_pixels'} if allow_legacy_image_fields else {'url'}
            if set(part) - {'type', 'image_url', 'max_pixels'} or set(part['image_url']) - url_keys:
                raise ValueError('Unsupported vision preprocessing fields; bound does not cover them')
            url = part["image_url"]["url"]
            if not url.startswith("data:image/") or ";base64," not in url:
                raise ValueError("Only frozen local image attachments are allowed")
            raw = base64.b64decode(url.split(",", 1)[1], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                width, height = image.size
                if width < 1 or height < 1:
                    raise ValueError("Invalid image dimensions")
                if getattr(image, 'n_frames', 1) != 1 or max(width, height) / min(width, height) > 200:
                    raise ValueError('Only static images with aspect ratio<=200 have a supported local vision bound')
            images.append({
                "sha256": hashlib.sha256(raw).hexdigest(),
                "width": width,
                "height": height,
                "visual_tokens_upper_bound": image_token_bound(width, height),
            })
            part["image_url"]["url"] = "[actual image bytes counted separately]"
    # Count the actual text strings, not their doubly escaped HTTP JSON envelope.
    text_bytes = len(packed(response_format).encode("utf-8"))
    for message in clean_messages:
        content = message.get("content")
        if isinstance(content, str):
            text_bytes += len(content.encode("utf-8"))
        elif isinstance(content, list):
            text_bytes += sum(len(part["text"].encode("utf-8"))
                              for part in content if part.get("type") == "text")
    image_bound = sum(row["visual_tokens_upper_bound"] for row in images)
    parts = sum(len(m['content']) if isinstance(m.get('content'), list) else 1 for m in clean_messages)
    framing = 1024 + 128 * len(messages) + 16 * parts
    maximum = text_bytes + framing + image_bound + reserve + OUTPUT_OVERRUN_ALLOWANCE
    return {
        "method": "UTF-8 byte text upper bound + full per-image patch ceiling",
        "text_bytes": text_bytes,
        "framing_allowance": framing,
        "framing_policy": "1024 base +128 per message +16 per content part; explicit engineering allowance, not a private provider template proof",
        "image_tokens_upper_bound": image_bound,
        "images": images,
        "requested_max_tokens": reserve,
        "output_overrun_allowance": OUTPUT_OVERRUN_ALLOWANCE,
        "output_reserved": reserve + OUTPUT_OVERRUN_ALLOWANCE,
        "maximum": maximum,
        "limit": OUTPUT_LIMIT,
        "image_bound_source": ("Deployed GLM-4.5V-AWQ preprocessor_config.json" if MODEL == GLM_MODEL
                               else "https://help.aliyun.com/zh/model-studio/vision"),
        "image_bound_contract": ("GLM processor maximum 12288 image patches plus 16 markers" if MODEL == GLM_MODEL else
                                 "Conditional Qwen-style smart resize bound, factor32, conservative minimum512*512, max_pixels2621440 and two image markers; exact private endpoint preprocessing not verified"),
        "provider_count_contract_verified": False,
        'historical_legacy_image_param_audit': allow_legacy_image_fields,
        "counting_assumptions": ["one text token consumes at least one original UTF8 byte (private tokenizer not supplied)",
            "response Schema counted once plus explicit framing allowance (private chat template/Schema linearization not supplied)",
            ("Deployed GLM processor uses its configured pixel ceiling and 28-pixel spatial merge"
             if MODEL == GLM_MODEL else "documented Qwen-family vision preprocessing applies to this private endpoint")],
    }

def image_token_bound(width, height):
    if MODEL == GLM_MODEL:
        return GLM_IMAGE_BOUND
    factor, minimum, maximum = 32, 512 * 512, IMAGE_MAX_PIXELS
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > maximum:
        beta = math.sqrt(height * width / maximum)
        h = math.floor(height / beta / factor) * factor
        w = math.floor(width / beta / factor) * factor
    elif h * w < minimum:
        beta = math.sqrt(minimum / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h * w // (factor * factor) + 2

def factor_schema(schema):
    """Share repeated JSON Schema objects while retaining existing local references."""
    counts, names, definitions = {}, {}, {}
    reserved = set(schema.get("$defs", {}))

    def visit(value, replace=False, root=False):
        if isinstance(value, list):
            return [visit(item, replace) for item in value]
        if not isinstance(value, dict):
            return value
        key = packed(value)
        if not replace:
            counts[key] = counts.get(key, 0) + 1
        schema_node = (isinstance(value.get("type"), (str, list))
                       or isinstance(value.get("anyOf"), list)
                       or isinstance(value.get("enum"), list))
        if replace and schema_node and not root and len(key) >= 80 and counts.get(key, 0) > 1:
            if key not in names:
                name = f"s{len(names)}"
                while name in reserved:
                    name += "_"
                reserved.add(name)
                names[key] = name
                definitions[name] = visit(value, True, True)
            return {"$ref": "#/$defs/" + names[key]}
        return {k: v if k in ("enum", "const", "default", "examples")
                else visit(v, replace) for k, v in value.items()}

    visit(schema)
    result = visit(schema, True, True)
    result["$defs"] = {**result.get("$defs", {}), **definitions}
    def references(value):
        if isinstance(value, dict):
            if "$ref" in value:
                yield value["$ref"]
            for key, child in value.items():
                if key not in ("enum", "const", "default", "examples"):
                    yield from references(child)
        elif isinstance(value, list):
            for child in value:
                yield from references(child)

    prefix = "#/$defs/"
    counts = Counter(references(result))
    inline = {prefix + name: node for name, node in definitions.items() if counts[prefix + name] == 1}
    for name in inline:
        result["$defs"].pop(name.removeprefix(prefix))
    def inline_once(value):
        if isinstance(value, dict):
            if set(value) == {"$ref"} and value["$ref"] in inline:
                return inline_once(inline[value["$ref"]])
            return {key: child if key in ("enum", "const", "default", "examples") else inline_once(child)
                    for key, child in value.items()}
        return [inline_once(child) for child in value] if isinstance(value, list) else value

    result = inline_once(result)
    return result if len(packed(result)) < len(packed(schema)) else schema

def provider_schema(value):
    """Token Plan rejects uniqueItems; preserve that check in original local validation."""
    if isinstance(value, dict):
        return {key: provider_schema(child) for key, child in value.items() if key != "uniqueItems"}
    if isinstance(value, list):
        return [provider_schema(child) for child in value]
    return value

def canonical_monitor_updates(monitor):
    result = deepcopy(monitor)
    updates = result.get("node_updates")
    if updates is None:
        return result
    if not isinstance(updates, list) or any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in updates):
        raise ValueError("Explicit node_updates entries with IDs are required")
    ids = [row["id"] for row in updates]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate explicit node_updates IDs")
    current = result.get("current_node_id")
    if current not in ids:
        raise ValueError("Explicit current_node_id update is missing; no checkpoint will be invented")
    result["node_updates"] = [row for row in updates if row["id"] == current] + [
        row for row in updates if row["id"] != current]
    return result


def canonical_initial_monitor(monitor):
    """Serialize the model's explicit graph in the native topological ID order."""
    result = deepcopy(monitor)
    nodes = result['nodes']
    graph = {node['id']: node['depends_on'] for node in nodes}
    if len(graph) != len(nodes) or any(dep not in graph for deps in graph.values() for dep in deps):
        raise ValueError('Initial DAG must contain distinct task IDs and only dependencies on existing tasks')
    ordered = list(graph)
    if all(node_id == f'n{index}' and set(graph[node_id]) <= set(ordered[:index])
           for index, node_id in enumerate(ordered)):
        return result
    try:
        ordered = list(TopologicalSorter(graph).static_order())
    except CycleError as exc:
        raise ValueError('Initial DAG contains a cycle; propose acyclic task dependencies') from exc
    ids = {node_id: f'n{index}' for index, node_id in enumerate(ordered)}
    by_id = {node['id']: node for node in nodes}
    for node in nodes:
        node['id'] = ids[node['id']]
        node['depends_on'] = [ids[dep] for dep in node['depends_on']]
    result['nodes'] = [by_id[node_id] for node_id in ordered]
    result['verification']['depends_on'] = [ids.get(dep, dep) for dep in result['verification']['depends_on']]
    if set(result['verification']['depends_on']) == set(ids.values()):
        result['verification']['depends_on'].sort(key=lambda node_id: int(node_id[1:]))
    result['current_node_id'] = ids.get(result['current_node_id'], result['current_node_id'])
    result['reopen_node_ids'] = [ids.get(node_id, node_id) for node_id in result['reopen_node_ids']]
    for requirement, task_ids in result['requirement_nodes'].items():
        result['requirement_nodes'][requirement] = [ids.get(node_id, node_id) for node_id in task_ids]
    return result

def extract_state(messages):
    state = {}
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "text":
                continue
            try:
                value = json.loads(part["text"])
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                state.update(value)
    return decode_prompt_tables(state)

def decode_prompt_tables(value):
    if isinstance(value, list):
        return [decode_prompt_tables(child) for child in value]
    if not isinstance(value, dict):
        return value
    if {"columns", "rows"} <= set(value) <= {"columns", "rows", "constants"}:
        columns, constants = value["columns"], value.get("constants", {})
        if len(set(columns)) != len(columns) or set(columns) & set(constants):
            raise ValueError("Ambiguous native table fields")
        if any(len(row) != len(columns) for row in value["rows"]):
            raise ValueError("Native table row arity mismatch")
        return [decode_prompt_tables({**dict(zip(columns, row)), **constants}) for row in value["rows"]]
    return {key: decode_prompt_tables(child) for key, child in value.items()}

def complete_validation_feedback(error, *, concise=False):
    if not isinstance(error, ValidationError):
        return [{"path": [], "error": str(error)}]
    if error.context:
        children = error.context
        if concise and error.validator == 'anyOf' and isinstance(error.instance, dict):
            instance = error.instance
            choices = list(enumerate(error.validator_value))
            for field in ('action', 'status', 'predicted_id', 'current_node_id'):
                matching = [(index, branch) for index, branch in choices
                            if field in instance and field in branch.get('properties', {}) and
                            Draft202012Validator(branch['properties'][field]).is_valid(instance[field])]
                choices = matching or choices
            nodes = instance.get('nodes')
            if isinstance(nodes, list):
                matching = [(index, branch) for index, branch in choices
                            if branch.get('properties', {}).get('nodes', {}).get('maxItems') == len(nodes)]
                choices = matching or choices
            matching = [(index, branch) for index, branch in choices
                        if {'node_updates', 'replace_plan'}.intersection(branch.get('required', ()))
                        and set(branch.get('required', ())) <= instance.keys()]
            choices = matching or choices
            if len(choices) == 1:
                index = choices[0][0]
                children = [child for child in children if child.schema_path[0] == index] or children
        return [row for child in children for row in complete_validation_feedback(child, concise=concise)]
    message = error.message
    if concise and len(message) > 120:
        constraint = packed(error.validator_value)
        message = f'{error.validator} constraint: {constraint[:160]}'
    return [{"path": list(error.absolute_path), "schema_path": list(error.absolute_schema_path),
             "validator": error.validator, "error": message}]

def expand_history_references(state):
    """Decode existing exact history references before model-selected row delivery."""
    observations = {row["evidence_id"]: row for row in state.get("observations", []) if "evidence_id" in row}

    def expand(item, field=None):
        if isinstance(item, list):
            return [expand(child, field) for child in item]
        if not isinstance(item, dict):
            return item
        if set(item) == {"same_as"}:
            return expand(observations[item["same_as"]][field], field)
        if set(item) == {"same_as_view"}:
            evidence, index = item["same_as_view"]
            return expand(observations[evidence]["views"][index][field], field)
        if set(item) == {"same_as_action"}:
            return expand(state["previous_actions"][item["same_as_action"]]["monitor"][field], field)
        if set(item) == {"same_as_task_dag"}:
            return expand(state["task_dag"][item["same_as_task_dag"]], field)
        return {key: expand(child, key) for key, child in item.items()}

    return expand(state)

def read_response(response, model, preflight, reserve, enforce_32k=True):
    if response.status_code != 200:
        raise RuntimeError(f"Qwen HTTP {response.status_code}; stopped without retry")
    value = response.json()
    if value.get("model") != model:
        raise RuntimeError(f"Unexpected Qwen response model: {value.get('model')}")
    choice = value["choices"][0]
    usage = value.get("usage") or {}
    if not isinstance(usage, dict):
        raise ProviderCapacityViolation('Provider returned a non-object usage record')
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if type(prompt_tokens) is not int or prompt_tokens < 0:
        raise ProviderCapacityViolation("Provider did not report valid integer prompt_tokens")
    if type(completion_tokens) is not int or completion_tokens < 0 or completion_tokens > reserve:
        raise ProviderCapacityViolation("Provider completion_tokens missing/invalid or beyond output reservation")
    if enforce_32k and prompt_tokens + reserve >= OUTPUT_LIMIT:
        raise ProviderCapacityViolation("Provider usage exceeded the reserved context window")
    if enforce_32k and prompt_tokens + completion_tokens >= OUTPUT_LIMIT:
        raise ProviderCapacityViolation("Provider actual input plus output reached strict context boundary")
    if prompt_tokens > preflight["maximum"] - reserve:
        raise ProviderCapacityViolation("Provider usage exceeded the preflight input upper bound")
    if choice.get("finish_reason") == "length":
        raise ValueError("Output reached max_tokens before completing JSON. Return a compact answer within the supplied Schema.")
    if choice.get("finish_reason") != "stop":
        raise RuntimeError(f"Qwen response incomplete: {choice.get('finish_reason')}")
    return value

def progress_identity(state, accepted_plan):
    plan = deepcopy(accepted_plan)
    if plan:
        plan.pop("revision_reason", None)
    return digest(progress_canonical({"active_target_ids": state.get("active_target_ids"), "accepted_plan": plan}))


def progress_canonical(value, field=""):
    if isinstance(value, dict):
        return {key: progress_canonical(item, key) for key, item in value.items()}
    if isinstance(value, list):
        items = [progress_canonical(item) for item in value]
        return sorted(items, key=packed) if field.endswith("_ids") or field in (
            "nodes", "allowed_actions", "source_quotes", "objects", "directed_pair_measurements",
            "conditional_observer_frames") else items
    return value


def evidence_progress_keys(row):
    """Historical acquisition coverage, not a semantic evidence-sufficiency test."""
    if row.get("no_evidence_acquired") or row.get("tool") in ("monitor_check", "finish_check"):
        return set()
    if row.get("tool") == "render":
        keys = set()
        for view in row.get("views", []):
            scope = {"tool": "render", "focus_ids": sorted(view.get("focus_ids", row.get("focus_ids", []))),
                     "context": view.get("context")}
            # Engineering novelty proxy only; 15 degrees does not guarantee information gain.
            azimuth = float(view.get("azimuth_deg", 315)) % 360
            elevation = float(view.get("elevation_deg", 30))
            keys.add(packed({**scope, "azimuth_bin": int(((azimuth + 7.5) % 360) // 15),
                             "elevation_bin": int((elevation + 7.5) // 15)}))
            for object_id, pixels in view.get("visible_instance_pixels", {}).items():
                if pixels > 0:
                    keys.add(packed({**scope, "visible_object_id": int(object_id)}))
        return keys
    # Deterministic raw measurements: ignore receipt/node IDs, retain numeric
    # facts and frames. Sorting object rows/groups does not create new evidence.
    return {packed(progress_canonical({key: value for key, value in row.items()
            if key not in ("evidence_id", "monitor_node_id")}))}


def candidate_contract(state, accepted_plan, revision):
    allowed = state.get("allowed_target_ids", [])
    active = state.get("active_target_ids", allowed)
    feedback = [row for row in state.get("observations", []) if row.get("no_evidence_acquired")]
    rollback = [row for row in feedback if row.get("error", "").startswith("Rollback accepted;")]
    return {"fixed_allowed_targets": {"ids": allowed, "meaning": "Fixed legal target domain; not all known objects."},
            "current_active_targets": {"ids": active, "meaning": "Not validly excluded by CURRENT accepted Monitor; multiple active IDs do NOT make the proposed ID illegal."},
            "legal_current_prediction_ids": sorted(set(allowed) & set(active)),
            "known_object_scope": "All objects remain available as references/comparison members for evidence; known references do not automatically become legal targets. Exclusion never deletes physical comparison members.",
            "accepted_native_plan": accepted_plan, "accepted_native_revision": revision,
            "original_evidence_ids": state.get("supporting_evidence_ids", []),
            "evidence_coverage": state.get("supporting_evidence_objects", {}),
            "discarded_exclusions": state.get("discarded_exclusions", []),
            "latest_native_rejection_verbatim": feedback[-1] if feedback else None,
            "latest_native_rollback_verbatim": rollback[-1] if rollback else None,
            "monitor_progression": {
                "rule": "A node's allowed_actions restricts the NEXT tool at that node, not the evidence needed to complete it. Read existing evidence and explicitly complete supported bindings yourself. You may complete an old node and select another ready pending tool node in the SAME proposal; current tool node stays pending and verification stays pending.",
                "replanning": "If no suitable pending node allows the needed tool, use monitor={replace_plan: COMPLETE_COMPACT_DAG} with the exact supplied Monitor schema. Delta node_updates cannot change allowed_actions, goals or dependencies. Explicitly add a relation/comparison task or revise permitted tools as justified; include verification and ALL requirement_nodes assignments. This is model planning, not permission to weaken evidence or force completion.",
                "uncertainty": "If binding evidence is insufficient, keep it pending and request discriminating evidence. Repeating a view does not complete it. Never invent binding, conclusion or exclusion to unlock a tool."},
            "instruction": "Read the literal fixed/active sets and original evidence. A uniqueness rejection is not an illegal-ID rejection. Do not narrow the physical comparison group to make a different target win; acquire covering evidence, explicitly roll back mistaken exclusions, or acknowledge uncertainty."}


class TentativeFinal(Exception):
    def __init__(self, answer, state_version, monitor_snapshot):
        super().__init__("Model-explicit tentative terminal evaluation")
        self.answer, self.state_version, self.monitor_snapshot = answer, state_version, monitor_snapshot


class BootstrapReady(Exception):
    """Offline real-worker startup reached the first actual API boundary."""


class ContextCapacityExceeded(RuntimeError):
    pass


class TentativeRivalOverlap(ValueError):
    pass


class ProviderCapacityViolation(RuntimeError):
    pass


class PreparedCall(Exception):
    """Offline-only final payload capture, before journal/HTTP; no model response."""
    def __init__(self, payload, preflight, packing):
        self.payload, self.preflight, self.packing = payload, preflight, packing


def prepare_qwen_payload(messages, schema, role, reserve, state=None, wire_schema=None, *,
                         text_visual_reports=False, recheck_ids=(), readback_ids=(), visual_report_rows=False):
    visible, packing = compact_call(messages, state if role == 'semantic_consultant' else None,
                                   text_visual_reports=text_visual_reports, visual_report_rows=visual_report_rows)
    for message in visible:
        if not isinstance(message.get('content'), list): continue
        for part in message['content']:
            if part.get('type') == 'image_url':
                part['image_url'].pop('max_pixels', None)
                if MODEL == GLM_MODEL:
                    part.pop('max_pixels', None)
                else:
                    part['max_pixels'] = IMAGE_MAX_PIXELS
    generated = compact_wire_schema(wire_schema if wire_schema is not None else schema,
                                    state if role == 'semantic_consultant' else None,
                                    text_visual_reports=text_visual_reports,
                                    compact_replacement=visual_report_rows)
    if MODEL == GLM_MODEL and role == 'semantic_consultant' and 'main_rivals' in generated.get('properties', {}):
        fields = generated['properties']
        targets = [target for target in fields['predicted_id'].get('enum', []) if target is not None]
        selected = fields['predicted_id'].get('const')
        if selected is not None:
            targets = [selected]
        no_target = deepcopy(generated)
        no_target['properties']['predicted_id'] = {'type': 'null'}
        no_target['properties']['unresolved']['minItems'] = 1
        if targets:
            fields['predicted_id'] = {'enum': targets}
            fields['evidence']['minItems'] = 1
            rival_ids = fields['main_rivals']['items']['properties']['object_id'].get('enum', targets)
            if fields['main_rivals'].get('maxItems') != 0 and (len(rival_ids) > 1 or selected is not None):
                fields['main_rivals']['minItems'] = 1
            if selected is None:
                generated = {'anyOf': [generated, no_target]}
        else:
            generated = no_target
    if recheck_ids:
        generated['properties']['action']['anyOf'].append(recheck_action_schema(recheck_ids))
    if readback_ids:
        generated['properties']['action']['anyOf'].append(readback_action_schema(readback_ids))
    generated = provider_schema(factor_schema(generated))
    Draft202012Validator.check_schema(generated)
    response_format = {'type': 'json_schema', 'json_schema': {'name': role + '_answer', 'strict': True, 'schema': generated}}
    preflight = qwen_preflight(visible, response_format, reserve)
    visible, packing, preflight = original_resolution_pack(visible, response_format, reserve, packing,
        preflight, qwen_preflight, OUTPUT_LIMIT)
    preflight.update(enforced=True, context_policy='strict-local-64k' if MODEL == GLM_MODEL else 'strict-local-32k',
                     provider_context_limit=GLM_CONTEXT if MODEL == GLM_MODEL else None)
    payload = {'model': MODEL, 'messages': visible, 'max_tokens': reserve, 'temperature': 0,
               'response_format': response_format}
    if MODEL == GLM_MODEL:
        payload['chat_template_kwargs'] = {'enable_thinking': False}
    else:
        payload.update(enable_thinking=False, vl_high_resolution_images=False)
    packing['visible_native_evidence_ids'] = ['initial'] + packing.get('all_raw_evidence_ids', []) if state is not None else []
    packing['readback_ids'] = list(readback_ids)
    return PreparedCall(payload, preflight, packing)


def recheck_action_schema(image_ids):
    return obj({'evidence_request': request_schema(image_ids)})


def readback_action_schema(source_ids):
    return obj({'read_evidence_page': {'enum': list(source_ids)}})


class DirectCalls:
    def __init__(self, root, *, visual_reports=False, visual_levels=3):
        self.root = root
        self.key = API_KEY
        self.record_only = True
        self.qwen_context_policy = "strict-local-64k" if MODEL == GLM_MODEL else "strict-local-32k"
        self.context_state = None
        self.session = requests.Session()
        self.session.trust_env = False
        self.ledger = []
        self.r_used = self.t_used = 0
        self.repair_observations_charged = set()
        self.monitor = None
        self.decisions = []
        self.rejected_native_actions = {}
        self.last_progress_identity = None
        self.stagnant_transitions = 0
        self.covered_monitor_states = set()
        self.covered_evidence_keys = set()
        self.tentative_evaluation_started = False
        self.pending_tentative_progress = None
        self.console_progress = False
        self.trace = None
        self.bootstrap_only = False
        self.preview_only = False
        self.selector_preview = None
        self.selector_sessions = 0
        self.visual = VisualReports(self._visual_call, self._save_visual_report,
                                    coordinated=visual_levels == 3) if visual_reports else None

    def progress(self, stage, **fields):
        b_used = sum(bool(step.get('observation')) and not step['observation'].get('no_evidence_acquired')
                     and step['observation'].get('tool') not in ('monitor_check', 'finish_check')
                     for step in (self.trace or {}).get('steps', []))
        save(self.root / "live-progress.json", {"stage": stage, "B": b_used, "R": self.r_used, "T": self.t_used,
             "calls": len(self.ledger), "utc": datetime.now(timezone.utc).isoformat(), **fields})
        if self.console_progress:
            print(packed({"event": "sample_stage", "stage": stage, "B": b_used, "R": self.r_used,
                          "T": self.t_used, "calls": len(self.ledger), **fields}), flush=True)

    def observe_progress(self, state):
        monitor_key = progress_identity(state, self.monitor.plan)
        new_monitor = monitor_key not in self.covered_monitor_states
        self.covered_monitor_states.add(monitor_key)
        rows = []
        for row in expand_history_references(state).get("observations", []):
            keys = evidence_progress_keys(row)
            new_keys = keys - self.covered_evidence_keys
            self.covered_evidence_keys.update(keys)
            rows.append({"evidence_id": row.get("evidence_id"), "tool": row.get("tool"),
                         "new_coverage_keys": sorted(new_keys)})
        new_evidence = any(row["new_coverage_keys"] for row in rows)
        self.stagnant_transitions = 0 if new_monitor or new_evidence else self.stagnant_transitions + 1
        self.last_progress_identity = digest({"monitor_states": sorted(self.covered_monitor_states),
                                              "evidence_keys": sorted(self.covered_evidence_keys)})
        save(self.root / "progress-checks" / f"t{self.t_used:02d}.json", {
            "new_accepted_monitor_state": new_monitor, "evidence": rows,
            "stagnant_transitions": self.stagnant_transitions, "threshold": 3,
            "policy": "Historical sorted groups/context/15-degree camera coverage and newly visible IDs; raw measurement facts/frames. Pixel counts, file paths and receipt IDs are not novelty. Engineering proxy, not evidence sufficiency."})
        return self.stagnant_transitions >= 3

    def tentative_final(self, messages, state, trigger):
        self.context_state = deepcopy(state)
        if self.tentative_evaluation_started:
            raise RuntimeError("Only one tentative final evaluation is permitted")
        self.tentative_evaluation_started = True
        version = digest(state)
        targets = sorted(set(state["allowed_target_ids"]) & set(self.monitor.active_ids()))
        known = [row["id"] for row in state["objects"]]
        evidence_ids = ["initial"] + [row["evidence_id"] for row in state.get("observations", [])
            if row.get("evidence_id") and not row.get("no_evidence_acquired")]
        text = {"type": "string", "minLength": 1}
        schema = {"type": "object", "properties": {
            "state_version": {"const": version},
            "predicted_id": {"type": ["integer", "null"], "enum": targets + [None]},
            "reason": text,
            "evidence": {"type": "array", "items": {"type": "object", "properties": {
                "source_quote": {"const": state["description"]},
                "object_ids": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "integer", "enum": known}},
                "evidence_ids": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "string", "enum": evidence_ids}},
                "observation": text}, "required": ["source_quote", "object_ids", "evidence_ids", "observation"], "additionalProperties": False}},
            "main_rivals": {"type": "array", "items": {"type": "object", "properties": {
                "object_id": {"type": "integer", **({"enum": targets} if targets else {})},
                "evidence_ids": {"type": "array", "items": {"type": "string", "enum": evidence_ids}},
                "comparison": text}, "required": ["object_id", "evidence_ids", "comparison"], "additionalProperties": False}},
            "unresolved": {"type": "array", "items": text}},
            "required": ["state_version", "predicted_id", "reason", "evidence", "main_rivals", "unresolved"], "additionalProperties": False}
        folder = self.root / "tentative-final"
        self.decisions.append({"T": self.t_used, "state_version": version, "mode": "tentative_final",
                               "trigger": trigger, "R_before": self.r_used})
        save(self.root / "decision-ledger.json", self.decisions)
        snapshot = {"plan": deepcopy(self.monitor.plan), "history": deepcopy(self.monitor.history),
                    "active_target_ids": self.monitor.active_ids()}
        save(folder / "trigger-and-unchanged-native-state.json", {"trigger": trigger, "T": self.t_used,
            "stagnant_transitions": self.stagnant_transitions, "threshold_engineering_choice": 3,
            "state_version": version, "native_snapshot": snapshot, "strict_monitor_accepted": False})
        visible = deepcopy(messages)
        visible[0] = {"role": "system", "content":
            "You are the same grounding model, performing ONE final tentative evaluation after no progress. "
            "This is an explicitly user-authorized TENTATIVE exit, not a strict Monitor finish. "
            "Do not emit or complete Monitor nodes, exclusions or rollback. The accepted plan and all "
            "other active candidates remain unchanged. Read the full original description, raw "
            "evidence and actual attached images. Choose the best-supported target ONLY from "
            "legal_current_prediction_ids, compare your strongest remaining rival(s), cite original "
            "evidence covering each claim, and state unresolved assumptions/uncertainty. Every "
            "main_rivals object_id must differ from predicted_id. If no "
            "usable basis exists, choose null and explain missing evidence. No forced target. "
            "Do not infer certainty from classifier priors or absence from a view. Negative right "
            "coordinates are less than positive coordinates; honor the SAME frame rows/order. "
            "Explain any frame assumption explicitly, never reverse numeric order. Fixed allowed "
            "and current active IDs are distinct from known references. Multiple active IDs do NOT "
            "make a proposed ID illegal and must remain recorded. Return only the supplied Schema."}
        visible.append({"role": "user", "content": packed({"state_version": version,
            "trigger": trigger, "candidate_contract": candidate_contract(state, self.monitor.plan, len(self.monitor.history)),
            "terminal_status": "tentative; strict_monitor_accepted=false; evidence_sufficiency independently audited"})})
        original_visible = deepcopy(visible)
        for attempt in range(2):
            try:
                response = self._qwen(f"t{self.t_used:02d}-tentative-final-{attempt}", "semantic_consultant",
                                      visible, schema, reserve=OUTPUT_RESERVE)
                answer = json.loads(response["_content"], object_pairs_hook=unique_json_object)
                validate_tentative_answer(answer, schema, targets)
                if snapshot != {"plan": self.monitor.plan, "history": self.monitor.history, "active_target_ids": self.monitor.active_ids()}:
                    raise RuntimeError("Tentative evaluator mutated native Monitor")
                self.decisions[-1].update(status="legal_tentative_terminal", R_after=self.r_used,
                                          source_call_index=self.ledger[-1]["index"])
                save(self.root / "decision-ledger.json", self.decisions)
                save(folder / "frozen-model-prediction.json", {"terminal_mode": "tentative", "accepted": False,
                    "strict_monitor_accepted": False, "source_call_index": self.ledger[-1]["index"],
                    "answer": answer, "unchanged_native_snapshot": snapshot})
                raise TentativeFinal(answer, version, snapshot)
            except (ValueError, ValidationError) as exc:
                self.r_used += 1
                feedback = {"errors": complete_validation_feedback(exc), "instruction": "ONE exact structural repair only. Same evidence and candidates; fix your own fields. No Monitor changes or semantic executor repair."}
                save(folder / f"structural-rejection-{attempt}.json", feedback)
                if attempt:
                    raise RuntimeError("Tentative evaluation still structurally invalid after one model repair") from exc
                call = self.ledger[-1]
                raw = self.root / "calls" / f"{call['index']:03d}-{call['tag']}-{call['role']}" / "provider-response.json"
                previous = json.loads(raw.read_text())["choices"][0]["message"]["content"]
                try:
                    proposal = json.loads(previous or "", object_pairs_hook=unique_json_object)
                except ValueError:
                    proposal = {}
                selected = proposal.get('predicted_id') if isinstance(proposal, dict) else None
                proposed_rivals = proposal.get('main_rivals') if isinstance(proposal, dict) else None
                if (type(selected) is int and selected in targets and proposal.get('state_version') == version
                        and isinstance(proposed_rivals, list)
                        and any(isinstance(row, dict) and row.get('object_id') == selected for row in proposed_rivals)):
                    # Keep the model's explicit target and constrain the rival IDs on its repair.
                    schema = deepcopy(schema)
                    fields = schema['properties']
                    fields['predicted_id'] = {'const': selected}
                    rivals = [target for target in targets if target != selected]
                    if rivals:
                        fields['main_rivals']['items']['properties']['object_id']['enum'] = rivals
                    else:
                        fields['main_rivals']['maxItems'] = 0
                    feedback['instruction'] += ' Keep predicted_id and give distinct rival IDs.'
                    if not isinstance(exc, TentativeRivalOverlap):
                        feedback['errors'].append({'path': ['main_rivals'], 'error': 'Rival list contains predicted target'})
                    save(folder / f"structural-rejection-{attempt}.json", feedback)
                feedback["previous_raw_answer"] = previous
                visible = deepcopy(original_visible) + [{"role": "user", "content": packed(feedback)}]
    def _selected_payload(self, tag, messages, schema, reserve, wire_schema):
        self.selector_sessions += 1
        session = self.root / 'evidence-selections' / f's{self.selector_sessions:04d}'
        before = deepcopy(self.context_state)
        native_version = digest(before)
        full, units = evidence_registry(before, messages)
        save(session/'full-native-state.json', before)
        manifest_units = []
        for unit in units:
            save(session/'units'/f"{unit['id']}.json", unit['raw'])
            originals = []
            for part in unit['images']:
                header, encoded = part['image_url']['url'].split(',', 1)
                raw = base64.b64decode(encoded, validate=True)
                sha = hashlib.sha256(raw).hexdigest()
                mime = header.split(';', 1)[0].removeprefix('data:')
                suffix = {'image/png': '.png', 'image/jpeg': '.jpg', 'image/webp': '.webp'}.get(mime, '.bin')
                path = session/'original-images'/(sha + suffix); path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists(): path.write_bytes(raw)
                originals.append({'path': str(path.relative_to(session)), 'sha256': sha})
            manifest_units.append({**{k: v for k, v in unit.items() if k not in ('raw', 'images')}, 'original_images': originals})
        save(session/'registry.json', {'native_state_version': native_version, 'units': manifest_units})
        def prepare_selector(batch_messages, batch_schema):
            return prepare_qwen_payload(batch_messages, batch_schema, 'evidence_selector', 768)
        def request_selector(batch_messages, batch_schema, metadata):
            folder = session/f"batch-{metadata['batch_index']:03d}"
            probe = prepare_selector(batch_messages, batch_schema)
            save(folder/'request-redacted.json', redact(probe.payload))
            save(folder/'preflight.json', probe.preflight)
            save(folder/'allowed-and-required-ids.json', metadata)
            if self.preview_only:
                if self.selector_preview is None:
                    raise EvidenceSelectionError('Offline selector requires an explicit simulated ID selection; no model reply fabricated')
                selected = self.selector_preview(metadata)
                answer = {'batch_id': metadata['batch_id'], 'selected_evidence_ids': selected}
                provenance = {'simulated_selection': True, 'actual_model_requests': 0, 'provider_usage': None}
                save(folder/'SIMULATED-selection-not-model-response.json', {**provenance, 'answer': answer})
                return answer, provenance
            try:
                response = self._qwen(f"selector-s{self.selector_sessions:04d}-b{metadata['batch_index']:03d}",
                                      'evidence_selector', batch_messages, batch_schema, reserve=768)
                answer = json.loads(response['_content'], object_pairs_hook=unique_json_object)
            except ProviderCapacityViolation:
                raise
            except Exception as exc:
                raise EvidenceSelectionError('Selector request/validation failed after journal saved: ' + str(exc)) from exc
            receipt = self.ledger[-1]
            return answer, {'simulated_selection': False, 'model': MODEL, 'call_index': receipt['index'],
                            'usage': receipt.get('usage'), 'preflight': receipt['preflight']}
        def prepare_main(main_messages, model_state):
            return prepare_qwen_payload(main_messages, schema, 'semantic_consultant', reserve, model_state, wire_schema)
        repair_ids = set()
        def requested_ids(value):
            if isinstance(value, list):
                for row in value: requested_ids(row)
            elif isinstance(value, dict):
                for key, child in value.items():
                    if key == 'evidence_ids' and isinstance(child, list): repair_ids.update(e for e in child if isinstance(e, str))
                    else: requested_ids(child)
        for message in messages:
            if not isinstance(message.get('content'), str) or message.get('role') == 'system': continue
            try:
                content = json.loads(message['content'])
                if isinstance(content, dict) and isinstance(content.get('previous_raw_answer'), str):
                    requested_ids(json.loads(content['previous_raw_answer']))
            except ValueError: pass
        repair_ids &= {unit['id'] for unit in units}
        try:
            prepared, audit, _ = select_context(before, messages, prepare_selector, request_selector,
                prepare_main, repair_ids)
        except EvidenceSelectionError as exc:
            save(session/'selection-failure.json', {'error': str(exc), 'audit': exc.audit,
                'journal_entries_to_date': len(self.ledger), 'no_default_or_semantic_selection': True})
            raise
        except ProviderCapacityViolation as exc:
            save(session/'provider-capacity-violation.json', {'error': str(exc), 'journal_entries_to_date': len(self.ledger),
                                                            'no_further_selector_or_main_sent': True})
            raise
        assert self.context_state == before, 'Selector altered full native state'
        audit.update(native_state_unchanged=True, canonical_native_state_version=native_version,
                     current_main_tag=tag, original_registry_path=str(session/'registry.json'))
        save(session/'selection-and-delivery.json', audit)
        prepared.packing['evidence_selection'] = audit
        prepared.packing['visible_native_evidence_ids'] = audit['selected_ids']
        return prepared

    def _save_visual_report(self, run_id, name, value):
        save(self.root / 'visual-reports' / run_id / name, value)

    def _visual_call(self, tag, role, messages, schema, reserve):
        try:
            answer = self._qwen(tag, role, messages, schema, reserve=reserve)
            return json.loads(answer['_content'], object_pairs_hook=unique_json_object)
        except (ValueError, ValidationError) as exc:
            raise EvidenceSelectionError(f'{role} returned an invalid visual report: {exc.message if isinstance(exc, ValidationError) else exc}') from exc

    def _reported_payload(self, messages, schema, reserve, wire_schema=None, *,
                          allow_recheck=False, allow_readback=False):
        from qwen_evidence_selector import required_evidence
        from eviground.grounding.dag_monitor import usable_evidence_observation

        state = self.context_state
        units = self.visual.collect_new(state, messages)
        expanded = expand_history_references(state)
        self._save_visual_report('registry', 'images.json', self.visual.catalog())
        all_ids = {row['id'] for row in units}
        mandatory = required_evidence(expanded, units)
        # Keep the latest rendered evidence and any raw facts cited by a
        # proposed repair. Original evidence remains in the native Monitor.
        latest_render = next((row['id'] for row in reversed(units) if row['raw']['tool'] == 'render'), None)
        if latest_render:
            mandatory.add(latest_render)
        measurements = {row['id']: row['object_group'] for row in units if row['raw']['tool'] == 'measure'}
        latest_measure = next(reversed(measurements), None)
        readback = self.visual.current_readback
        current_measurements = {readback} if readback in measurements else set()
        current_reports = self.visual.current_recheck_reports.copy()
        if readback in self.visual.reports:
            current_reports.add(readback)
        for msg in messages:
            content = msg.get('content')
            if msg.get('role') != 'system' and isinstance(content, str):
                try:
                    control = json.loads(content)
                except ValueError:
                    control = {}
                if isinstance(control, dict) and {'candidate_contract', 'image_recheck_completed', 'repair'} & control.keys():
                    continue
                references = all_ids.intersection(re.findall(r'\bobservation_[0-9]+\b', content))
                mandatory.update(references)
                current_measurements.update(measurements.keys() & references)
                current_reports.update(self.visual.reports_for_citations(content))
        mandatory.update(current_measurements)
        pinned_reports = current_reports | self.visual.reports_for_citations(packed(expanded.get('task_dag')))
        mandatory.update(row['evidence_id'] for report_id in pinned_reports
                         for row in self.visual.reports[report_id]['findings'])
        recheck_ids = tuple(self.visual.views) if allow_recheck else ()
        extras = [deepcopy(msg) for msg in messages
                  if msg.get('role') == 'system' or isinstance(msg.get('content'), str)]
        prior_rechecks = [key for key in self.visual.reports
                          if key in self.visual.recheck_reports and key not in pinned_reports]
        old_reports = [key for key in self.visual.reports
                       if key in self.visual.recheck_reports and key in pinned_reports and key not in current_reports]
        omitted_units, omitted_reports = set(), set()
        archive_order = [None, *prior_rechecks, *(row['id'] for row in units if row['id'] not in mandatory)]
        if reserve <= 1536:
            archive_order.extend(old_reports)
            keep_measure = readback if readback in measurements else latest_measure
            archive_order.extend(e for e in measurements if e in mandatory
                                 and e not in current_measurements and e != keep_measure)
        for remove_id in archive_order:
            if remove_id in self.visual.reports:
                omitted_reports.add(remove_id)
            elif remove_id is not None:
                omitted_units.add(remove_id)
            available = all_ids - omitted_units
            reports = self.visual.selected_reports(available, omitted_reports)
            projected = deepcopy(expanded)
            projected['views'] = []
            projected['observations'] = [
                {key: value for key, value in row.items() if key != 'views'}
                for row in projected['observations'] if not usable_evidence_observation(row) or row['evidence_id'] in available]
            # Reports can use multiple images. Expose an evidence ID for visual
            # claims only when its report is actually present in this request.
            scopes = {row['id']: set(row['object_group']) for row in units
                      if row['id'] in available and row['raw']['tool'] == 'measure'}
            for report in reports:
                for finding in report['findings']:
                    scope = scopes.setdefault(finding['evidence_id'], set())
                    if finding['status'] == 'observed':
                        scope.update(finding['object_ids'])
            visible_ids = scopes.keys()
            projected['supporting_evidence_ids'] = sorted(visible_ids)
            projected['supporting_evidence_objects'] = {key: sorted(ids) for key, ids in scopes.items()}
            # The image catalog already supplies the hash for every finding's image ID.
            projected['visual_reports'] = [{**report, 'findings': [
                {key: value for key, value in finding.items() if key != 'image_sha256'}
                for finding in report['findings']]} for report in reports]
            projected['image_catalog'] = self.visual.catalog()
            if projected.get('policy_annealing'):
                projected['policy_annealing'].pop('guidance', None)
                projected['policy_annealing'].pop('correction_policy', None)
            projected['archived_evidence_ids'] = sorted(all_ids - visible_ids)
            projected['archived_visual_report_ids'] = sorted(self.visual.reports.keys() -
                {report['report_id'] for report in reports})
            archived_measures = measurements.keys() - available
            readback_ids = (projected['archived_visual_report_ids'] + sorted(archived_measures)) if allow_readback else ()
            if allow_readback and projected['archived_visual_report_ids']:
                projected['archived_visual_report_index'] = [[report_id, sorted({row['image_id'] for row
                    in self.visual.reports[report_id]['findings']})] for report_id in projected['archived_visual_report_ids']]
            if archived_measures:
                projected['archived_measurement_index'] = [[e, measurements[e]] for e in measurements if e in archived_measures]
            prepared = prepare_qwen_payload(extras, schema, 'semantic_consultant', reserve, projected,
                  wire_schema, text_visual_reports=True, recheck_ids=recheck_ids, readback_ids=readback_ids)
            if reports and prepared.preflight['maximum'] >= OUTPUT_LIMIT:
                compact = prepare_qwen_payload(extras, schema, 'semantic_consultant', reserve, projected,
                    wire_schema, text_visual_reports=True, recheck_ids=recheck_ids,
                    readback_ids=readback_ids, visual_report_rows=True)
                if compact.preflight['maximum'] < prepared.preflight['maximum']:
                    prepared = compact
                    prepared.packing['visual_report_encoding'] = 'observation_rows'
            prepared.packing.update(mode='three_level_text_reports' if self.visual.coordinated else 'two_level_text_reports',
                                    visible_native_evidence_ids=sorted(visible_ids),
                                    visual_report_ids=[report['report_id'] for report in reports],
                                    delivered_visual_reports=reports,
                                    archived_evidence_ids=projected['archived_evidence_ids'],
                                    archived_visual_report_ids=projected['archived_visual_report_ids'],
                                    native_state_unchanged=True)
            if prepared.preflight['maximum'] < OUTPUT_LIMIT:
                return prepared
        for fallback in (3072, 2048, 1536):
            if reserve > fallback:
                return self._reported_payload(messages, schema, fallback, wire_schema,
                                              allow_recheck=allow_recheck, allow_readback=allow_readback)
        return prepared

    def _qwen(self, tag, role, messages, schema, reserve=OUTPUT_RESERVE, wire_schema=None, quote_registry=None):
        original_messages = deepcopy(messages)
        reserve = min(reserve, OUTPUT_RESERVE if role not in ('parser', 'parser_repair') else 1600)
        if role == 'semantic_consultant' and ('tentative-final' in tag or
                (self.visual is None and re.search(r'-action-[1-9][0-9]*$', tag))):
            reserve = min(reserve, 3072)
        try:
            if self.visual is not None and role == 'semantic_consultant' and self.context_state is not None:
                prepared = self._reported_payload(messages, schema, reserve, wire_schema,
                    allow_recheck='-action-' in tag and not self.visual.current_recheck_reports,
                    allow_readback='-action-' in tag and self.visual.current_readback is None)
            else:
                prepared = prepare_qwen_payload(messages, schema, role, reserve,
                    self.context_state if role == 'semantic_consultant' else None, wire_schema)
                if role == 'semantic_consultant' and self.context_state is not None and prepared.preflight['maximum'] >= OUTPUT_LIMIT:
                    prepared = self._selected_payload(tag, original_messages, schema, reserve, wire_schema)
            payload, preflight, packing = prepared.payload, prepared.preflight, prepared.packing
            messages = payload['messages']
            if self.visual is not None:
                if preflight['images'] and role != WORKER:
                    raise EvidenceSelectionError(f'Only the visual worker may receive pixels, got {role}')
                if role == WORKER and len(preflight['images']) != 1:
                    raise EvidenceSelectionError('Each visual worker must receive exactly one original image')
        except Exception as exc:
            rejection = ('rejected_evidence_selection' if isinstance(exc, EvidenceSelectionError) else
                         'rejected_after_selector_provider_capacity' if isinstance(exc, ProviderCapacityViolation) else 'rejected_context_packaging')
            record = {'index': len(self.ledger) + 1, 'tag': tag, 'role': role,
                      'http_sent': False, 'status': rejection,
                      'error_type': type(exc).__name__, 'error': str(exc), 'output_reserved': reserve}
            self.ledger.append(record)
            if role in ('semantic_consultant', 'image_selector', 'evidence_selector', 'parser_repair', COORDINATOR, WORKER):
                self.r_used += 1
            folder = self.root / 'calls' / f"{record['index']:03d}-{tag}-{role}"
            save(folder / 'receipt.json', redact(record))
            save(folder / 'original-request-redacted.json', redact(original_messages))
            save(self.root / 'model-ledger.json', self.ledger)
            if isinstance(exc, (EvidenceSelectionError, ContextCapacityExceeded, ProviderCapacityViolation)): raise
            raise RuntimeError('Context packaging failed before HTTP: ' + type(exc).__name__) from exc
        if self.preview_only:
            raise PreparedCall(payload, preflight, packing)
        if self.bootstrap_only:
            if preflight['maximum'] >= OUTPUT_LIMIT:
                raise RuntimeError(f'Offline bootstrap payload does not fit strict local {OUTPUT_LIMIT}-token bound')
            from eviground.query.query_syntax import syntax_settings
            from importlib.metadata import version
            save(self.root / 'bootstrap-check.json', {
                'ready': True, 'first_actual_api_boundary_reached': True,
                'role_at_boundary': role, 'http_attempts': 0, 'network_calls': 0,
                'model_key_read': False, 'syntax': syntax_settings(),
                'python_executable': sys.executable, 'sys_prefix': sys.prefix,
                'base_prefix': sys.base_prefix,
                'dependencies': {name: version(name) for name in ('spacy', 'en_core_web_sm', 'requests', 'numpy', 'Pillow', 'jsonschema')},
                'original_scene_and_category_loaded': True,
                'worker_sample_fields': ['scene', 'csv_row', 'query'],
                'annotation_target_supplied': False, 'real_provider_schema_validated': True,
                'strict_local_bound_passed': True, 'preflight': preflight,
                'provider_count_contract_verified': False, 'local_capacity_ready': True})
            check = json.loads((self.root / 'bootstrap-check.json').read_text())
            check['actual_inputs'] = self.bootstrap_inputs
            save(self.root / 'bootstrap-check.json', check)
            raise BootstrapReady('Offline bootstrap completed before any HTTP attempt')
        record = {'index': len(self.ledger) + 1, 'tag': tag, 'role': role, 'requested_model': MODEL, 'http_sent': False, 'preflight': preflight, 'output_reserved': preflight['output_reserved'], 'status': 'preflight'}
        self.ledger.append(record)
        folder = self.root / 'calls' / f"{record['index']:03d}-{tag}-{role}"
        folder.mkdir(parents=True, exist_ok=False)
        save(folder / 'request-redacted.json', redact(payload))
        save(folder / 'original-request-redacted.json', redact(original_messages))
        save(folder / 'context-packing.json', packing)
        save(folder / 'original-response-schema.json', schema)
        record['input_images'] = []
        for message in messages:
            if not isinstance(message.get('content'), list):
                continue
            for part in message['content']:
                if part.get('type') == 'image_url':
                    (header, encoded) = part['image_url']['url'].split(',', 1)
                    mime = header.split(';', 1)[0].removeprefix('data:')
                    suffix = {'image/png': '.png', 'image/jpeg': '.jpg', 'image/webp': '.webp'}.get(mime, '.bin')
                    name = f"input-image-{len(record['input_images']):02d}{suffix}"
                    raw = base64.b64decode(encoded, validate=True)
                    (folder / name).write_bytes(raw)
                    record['input_images'].append({'path': name, 'mime': mime, 'sha256': hashlib.sha256(raw).hexdigest()})
        record['original_input_images'] = []
        original_hashes = set()
        for message in original_messages:
            if not isinstance(message.get('content'), list): continue
            for part in message['content']:
                if part.get('type') != 'image_url': continue
                header, encoded = part['image_url']['url'].split(',', 1)
                mime = header.split(';', 1)[0].removeprefix('data:')
                raw = base64.b64decode(encoded, validate=True)
                sha = hashlib.sha256(raw).hexdigest()
                if sha in original_hashes: continue
                original_hashes.add(sha)
                matching = next((row for row in record['input_images'] if row['sha256'] == sha), None)
                if matching:
                    name = matching['path']
                else:
                    suffix = {'image/png': '.png', 'image/jpeg': '.jpg', 'image/webp': '.webp'}.get(mime, '.bin')
                    name = f"original-input-image-{len(record['original_input_images']):02d}{suffix}"
                    (folder / name).write_bytes(raw)
                with Image.open(io.BytesIO(raw)) as image: size = list(image.size)
                record['original_input_images'].append({'path': name, 'mime': mime, 'sha256': sha, 'size': size})
        save(folder / 'receipt.json', record)
        charged_role = role in ('semantic_consultant', 'image_selector', 'evidence_selector', 'parser_repair', COORDINATOR, WORKER)
        if charged_role:
            if not self.record_only and self.r_used >= 6:
                record.update(status='rejected_budget', error='R_budget_exhausted')
                save(folder / 'receipt.json', record)
                save(self.root / 'model-ledger.json', self.ledger)
                raise RuntimeError('R budget exhausted before Qwen request')
            self.r_used += 1
        if preflight['enforced'] and preflight['maximum'] >= OUTPUT_LIMIT:
            record.update(status='rejected_preflight', error='capacity_bound_exceeds_limit')
            save(folder / 'receipt.json', record)
            save(self.root / 'model-ledger.json', self.ledger)
            raise ContextCapacityExceeded(f"Model preflight {preflight['maximum']} >= {OUTPUT_LIMIT}; no request sent")
        if MODEL == GLM_MODEL and not preflight['images']:
            try:
                count = glm_prompt_count(self.session, payload)
            except (requests.RequestException, ValueError) as exc:
                record.update(status='rejected_tokenizer', error_type=type(exc).__name__,
                              error='GLM tokenizer request failed before generation')
                save(folder / 'receipt.json', record)
                save(self.root / 'model-ledger.json', self.ledger)
                raise RuntimeError(record['error']) from exc
            record['server_prompt_count'] = count
            if count + preflight['output_reserved'] >= OUTPUT_LIMIT:
                record.update(status='rejected_preflight', error='provider_tokenization_exceeds_limit')
                save(folder / 'receipt.json', record)
                save(self.root / 'model-ledger.json', self.ledger)
                raise ContextCapacityExceeded(f'GLM tokenizer count {count} plus output does not fit {OUTPUT_LIMIT}')
        started = time.monotonic()
        record['http_sent'] = True
        record['status'] = 'sent'
        save(folder / 'receipt.json', record)
        self.progress("request_sent", call=record['index'], role=role,
                      context_bound=preflight['maximum'], context_limit=OUTPUT_LIMIT)
        try:
            headers = {'Authorization': 'Bearer ' + self.key} if self.key else {}
            response = self.session.post(BASE_URL + '/chat/completions', json=payload, headers=headers, timeout=(45, 180), allow_redirects=False)
            (folder / 'provider-response.json').write_text(response.text.replace(self.key, '[REDACTED]') if self.key else response.text)
            record['http_status'] = response.status_code
            try:
                record['usage'] = response.json().get('usage')
            except (ValueError, AttributeError):
                record['usage'] = None
            save(folder / 'receipt.json', record)
            value = read_response(response, MODEL, preflight, preflight['output_reserved'], enforce_32k=preflight['enforced'])
            if 'server_prompt_count' in record and value['usage']['prompt_tokens'] != record['server_prompt_count']:
                raise ProviderCapacityViolation('GLM completion prompt usage differs from pre-send tokenization')
            content = value['choices'][0]['message'].get('content') or ''
            answer = json.loads(content, object_pairs_hook=unique_json_object)
            if role == WORKER and isinstance(answer, list):
                if len(answer) == 1 and isinstance(answer[0], dict) and set(answer[0]) == {'image_id', 'findings'}:
                    answer = answer[0]
                    record['response_normalization'] = 'unwrapped_single_assigned_image_report'
                else:
                    # The request has exactly one image, so its ID comes from the assignment.
                    answer = {'image_id': schema['properties']['image_id']['const'], 'findings': answer}
                    record['response_normalization'] = 'wrapped_worker_findings_with_assigned_image_id'
                content = packed(answer)
            if role == 'semantic_consultant' and self.context_state is not None:
                validate_visible_references(answer, packing['visible_native_evidence_ids'])
            if self.visual is not None and role == 'semantic_consultant':
                self.visual.check_image_citations(answer, packing['delivered_visual_reports'])
            action = answer.get('action') if isinstance(answer, dict) else None
            if isinstance(action, dict) and 'evidence_request' in action:
                if (self.visual is None or role != 'semantic_consultant' or '-action-' not in tag
                        or not self.visual.views or self.visual.current_recheck_reports):
                    raise ValueError('Image recheck is available once per planning turn with registered images; choose a native action now')
                validate(answer, obj({'state_version': schema['properties']['state_version'],
                                      'action': recheck_action_schema(self.visual.views)}))
            elif isinstance(action, dict) and 'read_evidence_page' in action:
                if not packing['readback_ids']:
                    raise ValueError('Archived evidence readback is unavailable in this decision')
                validate(answer, obj({'state_version': schema['properties']['state_version'],
                                      'action': readback_action_schema(packing['readback_ids'])}))
            else:
                validate(answer, schema)
            record.update(status='completed', usage=value.get('usage'), finish_reason=value['choices'][0].get('finish_reason'), request_id=response.headers.get('x-request-id'))
            (folder / 'response.json').write_text(content)
            save(folder / 'receipt.json', record)
            save(self.root / 'model-ledger.json', self.ledger)
            self.progress("response_saved", call=record['index'], role=role, http_status=response.status_code,
                          context_bound=preflight['maximum'], context_limit=OUTPUT_LIMIT,
                          prompt_tokens=(value.get('usage') or {}).get('prompt_tokens'),
                          completion_tokens=(value.get('usage') or {}).get('completion_tokens'))
            return {'_content': content, 'usage': value.get('usage'), '_finish_reason': value['choices'][0].get('finish_reason'), '_request': {'model': MODEL, 'response_model': value.get('model'), 'http_status': response.status_code, 'request_id': record.get('request_id'), 'latency_s': round(time.monotonic() - started, 4), 'preflight': preflight}}
        except Exception as exc:
            record.update(status='failed', error_type=type(exc).__name__, error=(str(exc).replace(self.key, '[REDACTED]') if self.key else str(exc))[:1000], latency_s=round(time.monotonic() - started, 4))
            save(folder / 'receipt.json', record)
            save(self.root / 'model-ledger.json', self.ledger)
            self.progress("request_failed", call=record['index'], role=role, error_type=type(exc).__name__)
            raise

    def parser_chat(self, messages, **kwargs):
        role = "parser_repair" if messages[0].get("content", "").startswith("Recover only omitted") else "parser"
        return self._qwen("query-parse", role, messages, kwargs["response_schema"]["schema"], kwargs.get("max_tokens", 1600))

    def choose(self, messages, **kwargs):
        self.t_used += 1
        state = extract_state(messages)
        if not state:
            raise ValueError("Missing original native controller state")
        self.context_state = deepcopy(state)
        if self.visual is not None:
            self.visual.current_recheck_reports.clear()
            self.visual.current_readback = None
        version = digest(state)
        folder = self.root / "controller-contracts" / f"t{self.t_used:02d}"
        native_schema = kwargs["response_schema"]["schema"]
        save(folder / "native-action-schema.json", native_schema)
        save(folder / "native-request-redacted.json", redact(messages))
        save(folder / "full-state.json", state)
        save(folder / "full-native-monitor-history.json", self.monitor.history)
        for index, row in enumerate(state.get("observations", [])):
            if (row.get("tool") in ("monitor_check", "finish_check") or row.get("no_evidence_acquired")) and index not in self.repair_observations_charged:
                self.r_used += 1
                self.repair_observations_charged.add(index)
        # Keep the original native state; only the final model payload is packed.
        visible = deepcopy(messages)
        visible[0]["content"] += (
            "\nDirect single-model contract: You interpret evidence, select exactly ONE native "
            "executable action and explicitly propose its complete Monitor. No action alternatives "
            "or second-model selection. Return {state_version, action}, with all native fields "
            "inside action. B/R/T are recorded with no limits; null remaining means unbounded. "
            "No deadline forces completion. Current native state is authoritative. Complete every "
            "binding/exclusion/completion/rollback yourself. The program adds no semantic fields. "
            "All original images attached to THIS request are delivered; metadata alone is not pixels. "
            "Monitor node_updates ordering is nonsemantic: an existing explicit update matching your "
            "current_node_id is serialized first, every original value retained; missing IDs reject. "
            "Use raw same-frame rows, units and rows_left_to_right as numeric facts. Negative right "
            "coordinates are less than positive coordinates. An alternate frame requires your explicit "
            "frame proposal and explanation; do not silently reverse numeric order to support a target. "
            "Cite only observations covering the objects and facts claimed. Unseen is unknown, "
            "classifier hypotheses are priors, not proven categories. Do not invent exclusions. "
            "To replan, explicitly use the native complete replace_plan form. Legal null finish "
            "states missing evidence; no forced target. "
            "This record-only protocol is exploratory, not strict B10/R6/T17 comparison.")
        if MODEL == GLM_MODEL:
            visible[0]["content"] += (
                "\nFor inspect, set elevation_deg to null for the default view, or choose a value "
                "from 10 to 75 degrees inclusive. Set azimuth_deg to null or a value from 0 up to "
                "but excluding 360 degrees.")
        native_feedback = [row for row in state.get("observations", []) if row.get("no_evidence_acquired")]
        visible.append({"role": "user", "content": packed({"state_version": version,
            "candidate_contract": candidate_contract(state, self.monitor.plan, len(self.monitor.history))})})
        stalled = self.observe_progress(state)
        early_repeat = self.pending_tentative_progress == self.last_progress_identity
        if stalled or early_repeat:
            return self.tentative_final(messages, state,
                "three_consecutive_stagnant_transitions" if stalled else "exact_repeated_native_rejection")
        original_visible = deepcopy(visible)
        # Original action schema validates locally; provider wire keeps the native
        # shape but avoids enumerating every graph branch during generation.
        direct_schema = {"type": "object", "properties": {
            "state_version": {"const": version}, "action": native_schema},
            "required": ["state_version", "action"], "additionalProperties": False}
        native_defs = direct_schema["properties"]["action"].get("$defs")
        if native_defs:
            direct_schema = deepcopy(direct_schema)
            direct_schema["$defs"] = direct_schema["properties"]["action"].pop("$defs")
        wire = deepcopy(direct_schema)
        evidence_ids = set(state.get("supporting_evidence_ids", [])) | {"initial"}
        monitor_schema = (self.monitor.delta_schema(evidence_ids) if self.monitor.plan is not None
                          else self.monitor.schema(compact=True, evidence_ids=evidence_ids))
        for branch in wire["properties"]["action"]["anyOf"]:
            branch["properties"]["monitor"] = deepcopy(monitor_schema)
        wire.pop("$defs", None)
        rejected, attempt = {}, 0
        repair_patterns = Counter()
        decision = {"T": self.t_used, "state_version": version, "R_before": self.r_used,
                    "mode": "single_qwen_direct_action", "image_policy": "worker_only" if self.visual else "all_actual_images"}
        self.decisions.append(decision)
        try:
            while True:
                call_index = len(self.ledger) + 1
                try:
                    response = self._qwen(f"t{self.t_used:02d}-action-{attempt}", "semantic_consultant",
                                          visible, wire, reserve=OUTPUT_RESERVE, wire_schema=wire)
                    if self.ledger and self.ledger[-1].get('role') == 'semantic_consultant':
                        call_index = self.ledger[-1]['index']
                    answer = json.loads(response["_content"], object_pairs_hook=unique_json_object)
                    if self.visual is not None and 'read_evidence_page' in answer['action']:
                        self.visual.current_readback = answer['action']['read_evidence_page']
                        repair_patterns.clear()
                        save(folder / f'evidence-readback-{attempt}.json', {
                            'source_id': self.visual.current_readback, 'source_call_index': call_index})
                        visible = deepcopy(original_visible)
                        attempt += 1
                        continue
                    if self.visual is not None and 'evidence_request' in answer['action']:
                        request = answer['action']['evidence_request']
                        self.visual.recheck(state, request)
                        repair_patterns.clear()
                        save(folder / f'image-recheck-{attempt}.json', {'request': request, 'source_call_index': call_index})
                        visible = deepcopy(original_visible) + [{"role": "user", "content": packed({
                            "image_recheck_completed": request,
                            "instruction": "Use the returned findings to choose a native action now. "
                                           "Use inspect for a different camera or measure for geometric evidence."})}]
                        original_visible = deepcopy(visible)
                        attempt += 1
                        continue
                    action = answer["action"]
                    before_order = deepcopy(action["monitor"])
                    if "node_updates" in action["monitor"]:
                        action["monitor"] = canonical_monitor_updates(action["monitor"])
                    elif self.visual is not None and self.monitor.plan is None:
                        action["monitor"] = canonical_initial_monitor(action["monitor"])
                    validate(answer, direct_schema)
                    validate(action, native_schema)
                    save(folder / f"action-{attempt}-serialization.json", {
                        "model_original_monitor": before_order, "serialized_monitor": action["monitor"],
                        "semantic_fields_added": [], "explicit_action": action})
                    save(folder / "selected-model-action.json", action)
                    if native_feedback:
                        core = deepcopy(action)
                        core.pop("reason", None)
                        core["monitor"].pop("revision_reason", None)
                        identity = digest({"action": core, "accepted_native_plan": self.monitor.plan,
                                           "real_evidence": [row for row in state.get("observations", []) if not row.get("no_evidence_acquired")],
                                           "native_feedback": native_feedback[-1]})
                        if identity in self.rejected_native_actions:
                            save(folder / "no-progress-stop.json", {"repeat_of_call": self.rejected_native_actions[identity],
                                "current_call": call_index, "next_decision": "model_tentative_final",
                                "reason": "Exact executable action repeated after native rejection with unchanged accepted plan, real evidence and feedback. Next decision evaluates tentative terminal, not a resource cap."})
                            self.pending_tentative_progress = self.last_progress_identity
                        self.rejected_native_actions[identity] = call_index
                    decision.update(status="model_action_returned", source_call_index=call_index)
                    response["_content"] = json.dumps(action, ensure_ascii=False)
                    return response
                except (ValidationError, ValueError) as exc:
                    if not self.ledger or not self.ledger[-1]["http_sent"]:
                        raise
                    self.r_used += 1
                    last = self.ledger[-1]
                    raw_file = self.root / "calls" / f"{last['index']:03d}-{last['tag']}-{last['role']}" / "provider-response.json"
                    raw = json.loads(raw_file.read_text())["choices"][0]["message"]["content"]
                    identity = hashlib.sha256(raw.encode()).hexdigest()
                    feedback = {"state_version": version, "no_action_or_monitor_accepted": True,
                                "native_validation_errors": complete_validation_feedback(exc),
                                "previous_raw_answer": raw,
                                "instruction": "Correct your response against the supplied contract. No new native tool executed. AnyOf branches are alternatives. No program repair."}
                    save(folder / f"repair-{attempt}.json", feedback)
                    if self.visual is not None:
                        from qwen_step2_context import compact_feedback
                        diagnostics = complete_validation_feedback(exc, concise=True)
                        pattern = digest(sorted({packed((row['path'], row.get('schema_path'),
                            row.get('validator', row['error']))) for row in diagnostics}))
                        repair_patterns[pattern] += 1
                        if identity in rejected or repair_patterns[pattern] >= 3:
                            save(folder / "structural-stagnation.json", {
                                "source_call": last['index'], "pattern_count": repair_patterns[pattern],
                                "diagnostics": diagnostics, "next_decision": "model_tentative_final"})
                            decision.update(status="repeated_structural_rejection", source_call_index=last['index'])
                            return self.tentative_final(messages, state, "repeated_structural_rejection")
                        feedback["native_validation_errors"] = diagnostics
                        feedback = compact_feedback(feedback)
                        feedback = {"repair": [[row["path"], row["error"]]
                                               for row in feedback["native_validation_errors"][:6]]}
                    if identity in rejected:
                        raise RuntimeError("Repeated identical invalid direct action after exact model feedback") from exc
                    rejected[identity] = call_index
                    visible = deepcopy(original_visible)
                    visible.append({"role": "user", "content": packed(feedback)})
                    attempt += 1
        finally:
            decision["R_after"] = self.r_used
            save(self.root / "decision-ledger.json", self.decisions)


def validate_tentative_answer(answer, schema, targets):
    validate(answer, schema)
    if answer["predicted_id"] is not None and answer["predicted_id"] not in targets:
        raise ValueError("Tentative target outside allowed and active intersection")
    if answer["predicted_id"] is not None and not any(answer["predicted_id"] in row["object_ids"] for row in answer["evidence"]):
        raise ValueError("Tentative prediction lacks evidence object coverage")
    if answer["predicted_id"] is None and not answer["unresolved"]:
        raise ValueError("Tentative null must explain missing evidence")
    if answer["predicted_id"] is not None and len(targets) > 1 and not answer["main_rivals"]:
        raise ValueError("Tentative prediction must compare a main rival")
    for row in answer["main_rivals"]:
        if row["object_id"] == answer["predicted_id"]:
            raise TentativeRivalOverlap("Rival list contains predicted target")


def build_frozen_manifest(*, include_single_sample_inputs=True):
    sources = sorted([*PROJECT.glob("src/**/*.py"), Path(__file__),
                      PROJECT / 'scripts/qwen_step2_context.py', PROJECT / 'scripts/check_qwen_step2_context.py',
                      PROJECT / 'scripts/qwen_evidence_selector.py', PROJECT / 'scripts/qwen_visual_reports.py',
                      PROJECT / 'tests/unit/test_qwen_visual_reports.py',
                      PROJECT / 'tests/unit/test_qwen_evidence_selector.py',
                      PROJECT / 'tests/fixtures/qwen_step2_exact_quotes.json',
                      PROJECT / 'tests/fixtures/qwen_step2_worker_findings_array.json',
                      PROJECT / 'tests/unit/test_qwen_step2_context.py', PROJECT / 'docs/qwen-step2-context.md',
                      PROJECT / 'docs/qwen-step2-adaptive.md', PROJECT / 'tests/fixtures/qwen_step2_initial_monitor_wrapper.json',
                      PROJECT / "tests/unit/test_qwen_direct_fresh.py",
                      PROJECT / "tests/fixtures/qwen_direct_q00171_progress.json",
                      PROJECT / "docs/qwen-direct-v3-loop-fix.md",
                      PROJECT / "docs/qwen-direct-protocol.md",
                      PROJECT / "docs/qwen-direct-source-provenance.json", PROJECT / "pyproject.toml"])
    files = [INPUTS / "nr3d/scene0565_00/scene.npz", INPUTS / "nr3d/scene0565_00/scene_instances.json",
             INPUTS / "nr3d/query.json", INPUTS / "categories/scene0565_00/objects_top5.json"]
    glm = MODEL == GLM_MODEL
    return {"candidate": "glm-three-level-visual-reports" if glm else "qwen-step2-three-level-visual-reports",
            "branch": "20261121_step3_glm" if glm else "20261120_step2",
            "baseline_commit": ("e4705403853c9163698cfc6c60ccaccb21968e35" if glm else
                                "8a44a23897687d8fb35a766b71bf67c008767940"),
            "model": MODEL, "endpoint": BASE_URL, "scheduler_sha256": sha256(Path(__file__)),
            "source_closure": {str(p.relative_to(PROJECT)): sha256(p) for p in sources},
            "inputs": {str(p.relative_to(PROJECT)): sha256(p) for p in files} if include_single_sample_inputs else {},
            "python": sys.version, "packages": {d.metadata["Name"]: d.version for d in distributions() if d.metadata.get("Name")},
            "resource_budget_policy": {"mode": "record-only", "B_limit": None, "R_limit": None, "T_limit": None,
                                       "annealing_reference_turns_not_a_cap": 17},
            "qwen_context_policy": {"policy": "strict-local-64k" if glm else "strict-local-32k",
                f"local_{OUTPUT_LIMIT}_refusal_enforced": True, 'strict_operator': '<',
                "provider_context_limit_verified": glm, 'private_count_contract_verified': False,
                "server_tokenizes_text_before_generation": glm, "reference_bound": OUTPUT_LIMIT,
                'requested_output_limits': {'parser': 1600, 'parser_repair': 1600, 'action': 4096,
                    'action_capacity_fallbacks': [3072, 2048, 1536], 'structural_repair': 3072, 'tentative': 3072,
                    'visual_coordinator_plan': 1024, 'visual_coordinator_review': 3072, 'visual_worker': 1024},
                'output_overrun_cushion': OUTPUT_OVERRUN_ALLOWANCE,
                'overflow_policy': 'Reversible text packing, smaller output reservations, then page out older reports and measurement groups. Retain current sources and index archived IDs for one explicit readback per decision. Reject before HTTP if these do not fit.',
                'visual_worker_original_image_bytes': True,
                'framing': '1024 base +128/message +16/part; engineering allowance only; no provider absolute proof'},
            "model_visibility": "Three levels use the same model. Main and visual coordinator receive text only. Each visual worker reads one original image. The coordinator groups original findings by ID and flags conflict or gaps. The main receives reviewed textual reports and raw measurements. Complete original evidence stays local.",
            "decision_protocol": f"Native local Schema/Monitor and verbatim source quotes remain enforced. Main can ask for image rechecks independently of inspect/measure/finish. Visual calls are counted in R; B/R/T are record-only. Every request has a strict local {OUTPUT_LIMIT}-token preflight and responses have provider usage audits.",
            "credential_policy": ("Loopback vLLM service via SSH" if glm else
                                  "Git-external Token Plan config or run-scoped key override; key contents excluded from frozen metadata."),
            "scope": "Diagnostic new single run; not strict B10/R6/T17 comparison, formal score or paper update.",
            "fresh_start": True, "old_runtime_source_imports": [], "inherited_dag_answers_predictions": [],
            "server_model_requests": 0, "gt_read_before_prediction": False}


def run(root, batch_request=None, bootstrap_check=False):
    global API_KEY
    if root.exists():
        raise FileExistsError(f"Run directory already exists: {root}")
    root.mkdir(parents=True)
    request = None
    if batch_request is not None:
        request_path = Path(batch_request).resolve()
        request = json.loads(request_path.read_text())
        if set(request) != {"sample", "scene_dir", "category_path", "shared_manifest", "input_hashes"} or set(request["sample"]) != {"scene", "csv_row", "query"}:
            raise ValueError("Unexpected worker input fields; target annotations are forbidden")
        shared_path = (request_path.parent / request["shared_manifest"]).resolve()
        shared = json.loads(shared_path.read_text())
        if shared['protocol']['model'] == GLM_MODEL:
            configure_glm(shared['protocol']['endpoint'])
        manifest = {**shared["protocol"], "source_closure": shared["source_closure"],
                    "shared_manifest": str(shared_path),
                    "shared_manifest_sha256": sha256(shared_path),
                    "sample": request["sample"], "inputs": request["input_hashes"]}
        for name, expected in request["input_hashes"].items():
            if sha256(Path(name)) != expected:
                raise ValueError(f"Worker input changed: {name}")
    else:
        manifest = build_frozen_manifest()
    API_KEY = '' if bootstrap_check or MODEL == GLM_MODEL else private_qwen_config()
    calls = DirectCalls(root, visual_reports=True, visual_levels=manifest.get('visual_levels', 3))
    calls.bootstrap_only = bootstrap_check
    calls.console_progress = request is not None
    save(root / "frozen-manifest.json", manifest)
    for name, expected in (() if request else manifest["source_closure"].items()):
        source = PROJECT / name
        destination = root / "frozen-source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        if sha256(destination) != expected:
            raise ValueError(f"Source changed while freezing: {name}")
    shutil_inputs = root / "frozen-inputs"
    if not request:
        shutil_inputs.mkdir()
    for path in (() if request else (
        INPUTS / "nr3d/scene0565_00/scene.npz",
        INPUTS / "nr3d/scene0565_00/scene_instances.json",
        INPUTS / "categories/scene0565_00/objects_top5.json",
        INPUTS / "nr3d/query.json",
    )):
        destination = shutil_inputs / path.name
        destination.write_bytes(path.read_bytes())
        if sha256(destination) != sha256(path):
            raise ValueError(f"Frozen input copy mismatch: {path.name}")

    for name, value in {
        "EVIGROUND_API_PROVIDER": "openai",
        "EVIGROUND_BASE_URL": BASE_URL,
        "EVIGROUND_MODEL": MODEL,
        "EVIGROUND_API_KEY": API_KEY,
        "EVIGROUND_PROMPT_PROFILE": "compact",
        "EVIGROUND_DAG_MONITOR_ENABLED": "1",
        "EVIGROUND_DAG_WIRE_FORMAT": "delta",
        "EVIGROUND_POLICY_ANNEALING_ENABLED": "1",
        "EVIGROUND_DAG_REPAIR_LIMIT": "6",
        "EVIGROUND_DECISION_MAX_TOKENS": str(OUTPUT_RESERVE),
        "EVIGROUND_SUCCESSFUL_TOOL_BUDGET": "1",
        "EVIGROUND_CONTEXT_LIMIT": str(OUTPUT_LIMIT),
        "EVIGROUND_DIALOGUE_HISTORY": "0",
    }.items():
        os.environ[name] = value

    import eviground.query.parser as query_parser
    import eviground.grounding.react as react
    import eviground.grounding.dag_monitor as dag_module
    NativeMonitor = dag_module.DagMonitor

    class RegisteredMonitor(NativeMonitor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            calls.monitor = self

    dag_module.DagMonitor = RegisteredMonitor
    query_parser.chat = calls.parser_chat
    react.chat = calls.choose

    from eviground.categories.hypotheses import load_category_objects
    from eviground.categories.retrieval import match_retrieval_categories
    from eviground.grounding.bindings import generic_target_pool
    from eviground.grounding.pipeline import structural_objects
    from eviground.query.parser import parse_query
    from eviground.rendering.renderer import load_objects
    import numpy as np

    scene = Path(request["scene_dir"]) if request else INPUTS / "nr3d/scene0565_00"
    category_path = Path(request["category_path"]) if request else INPUTS / "categories/scene0565_00/objects_top5.json"
    query_record = request["sample"] if request else json.loads((INPUTS / "nr3d/query.json").read_text())
    query = query_record["query"]
    category_objects = load_category_objects(category_path)
    geometry_objects = category_objects + structural_objects(
        scene, category_objects,
    )
    with np.load(scene / "scene.npz", allow_pickle=False) as cloud:
        xyz, rgb, ids = cloud["xyz"], cloud["rgb"], cloud["ids"]

    ledger = []
    trace = {"decision_backend": "single_qwen_direct_fresh",
             "steps": [], "internal_usage": {"compute": 0, "render": 0, "visual_check": 0}}
    calls.trace = trace
    if bootstrap_check:
        calls.bootstrap_inputs = {'sample': query_record, 'scene_dir': str(scene),
            'category_path': str(category_path), 'point_count': len(xyz),
            'category_object_count': len(category_objects), 'geometry_object_count': len(geometry_objects),
            'original_query_loaded': True,
            'GT_fields_present': any(k in query_record for k in ('target_id', 'object_id'))}
    start = datetime.now(timezone.utc).isoformat()
    interrupted = False
    try:
        parsed = parse_query(
            query, calls=ledger,
            scene_categories=[hypothesis["category"] for obj in category_objects
                              for hypothesis in obj.get("category_hypotheses", [])],
        )
        candidates = match_retrieval_categories(parsed["Target"], category_objects)
        if generic_target_pool(parsed["Target"]):
            candidates = list(category_objects)
        if not candidates:
            raise RuntimeError("Fresh parser produced no candidate pool")
        prediction = react.ground(
            query, parsed, geometry_objects, candidates,
            xyz, rgb, ids, root, ledger, trace, None if calls.record_only else 10,
        )
        status = "completed" if prediction.get("predicted_id") is not None else "abstained"
    except BootstrapReady:
        status = 'bootstrap_ready'
        prediction = {'predicted_id': None, 'offline_bootstrap_only': True}
    except TentativeFinal as exc:
        status = "tentative" if exc.answer["predicted_id"] is not None else "tentative_abstained"
        prediction = {"predicted_id": exc.answer["predicted_id"], "decision_status": "tentative",
                      "accepted": False, "strict_monitor_accepted": False, "terminal_mode": "tentative",
                      "model_final_evaluation": exc.answer, "unchanged_native_snapshot": exc.monitor_snapshot}
        trace["termination_cause"] = "tentative_stagnation_final"
        trace["dag_monitor_final_plan"] = exc.monitor_snapshot["plan"]
        trace["dag_monitor_active_target_ids"] = exc.monitor_snapshot["active_target_ids"]
    except KeyboardInterrupt:
        interrupted = True
        status = "interrupted"
        prediction = {"predicted_id": None, "failure_type": "KeyboardInterrupt",
                      "failure": "User interruption; started-only requests may have unknown remote execution. Do not automatically retry."}
    except EvidenceSelectionError as exc:
        status = 'evidence_selection_failed'
        trace['termination_cause'] = status
        prediction = {'predicted_id': None, 'terminal_mode': status, 'failure_type': type(exc).__name__,
                      'failure': str(exc), 'automatic_retry': False}
    except (ContextCapacityExceeded, ProviderCapacityViolation) as exc:
        status = 'capacity_rejected' if isinstance(exc, ContextCapacityExceeded) else 'provider_capacity_violation'
        trace['termination_cause'] = status
        prediction = {'predicted_id': None, 'terminal_mode': status,
                      'failure_type': type(exc).__name__, 'failure': str(exc), 'automatic_retry': False}
    except Exception as exc:
        status = "failed"
        prediction = {"predicted_id": None, "failure_type": type(exc).__name__,
                      "failure": str(exc)[:1500]}
    save(root / "parser-calls.json", ledger)
    save(root / "trace.json", trace)
    save(root / "model-ledger.json", calls.ledger)
    result = {
        "status": status,
        "dataset": "Nr3D",
        "sample": {"scene": query_record["scene"], "csv_row": query_record["csv_row"]},
        "started_utc": start,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "fresh_start": True,
        "model": MODEL,
        "qwen_context_policy": manifest["qwen_context_policy"],
        "predicted_id": prediction.get("predicted_id"),
        "decision_status": prediction.get("decision_status"),
        "budgets": {
            "B_limit": None if calls.record_only else 10,
            "B_used": sum(1 for step in trace["steps"] if step.get("observation")
                          and not step["observation"].get("no_evidence_acquired")
                          and step["observation"].get("tool") not in ("monitor_check", "finish_check")),
            "R_limit": None if calls.record_only else 6,
            "R_used": calls.r_used,
            "T_limit": None if calls.record_only else 17,
            "T_used": calls.t_used,
        },
        "parser_calls": len(ledger),
        "model_call_attempts": len(calls.ledger),
        "flow_complete": trace.get("termination_cause") in (
            "supported_finish", "abstained", "tentative_stagnation_final",
        ),
        "terminal_mode": prediction.get("terminal_mode", "strict_monitor" if status in ("completed", "abstained") else "failed"),
        "strict_monitor_accepted": status == "completed",
        "accepted": status == "completed",
        "legal_terminal_formed": status in ("completed", "abstained", "tentative", "tentative_abstained"),
        "evidence_sufficient": "not_yet_audited",
        "prediction_correct": "not_yet_scored",
        "formal_result": False,
        "resource_budget_policy": manifest["resource_budget_policy"],
        "source_closure_unchanged": all(sha256(PROJECT / name) == expected
                                       for name, expected in manifest["source_closure"].items()),
        "prediction": prediction,
    }
    save(root / "result.json", result)
    calls.progress("sample_terminal", status=status, predicted_id=result["predicted_id"], B=result["budgets"]["B_used"])
    print(packed({"event": "sample_terminal", "status": status, "predicted_id": result["predicted_id"],
                  "budgets": result["budgets"]}) if request else json.dumps(result, ensure_ascii=False), flush=True)
    if interrupted:
        raise SystemExit(130)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--worker-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--bootstrap-check", action='store_true', help='Real worker startup, no credentials or network; stop at first API boundary')
    args = parser.parse_args()
    if args.bootstrap_check:
        import socket
        def deny_network(*_args, **_kwargs):
            raise RuntimeError('Offline bootstrap prohibits all network requests')
        socket.socket.connect = deny_network
        socket.create_connection = deny_network
        requests.Session.request = deny_network
    run(args.run_dir, args.worker_input, args.bootstrap_check)
