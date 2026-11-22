"""Isolated single-Qwen native ReAct episode; no Laya runtime or remote GPU."""
from __future__ import annotations
import argparse
import base64
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import distributions
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import requests
from PIL import Image
from jsonschema import Draft202012Validator, ValidationError, validate

PROJECT = Path(__file__).resolve().parents[1]
INPUTS = PROJECT / "data/qwen_direct"
MODEL = "Qwen/Qwen3-VL-32B-Instruct"
BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_KEY_ENV = "SILICONFLOW_API_KEY"
OUTPUT_LIMIT = 262144
OUTPUT_RESERVE = 8192
CONTEXT_POLICY = "strict-local-256k"
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

def qwen_preflight(messages, reserve):
    images = []
    text_bytes = 0
    parts = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts += 1
            text_bytes += len(content.encode("utf-8"))
            continue
        parts += len(content)
        for part in content:
            if part["type"] == "text":
                text_bytes += len(part["text"].encode("utf-8"))
                continue
            if part["type"] != "image_url":
                raise ValueError("Only text and static image attachments are supported")
            url = part["image_url"]["url"]
            if not url.startswith("data:image/") or ";base64," not in url:
                raise ValueError("Only frozen local image attachments are allowed")
            raw = base64.b64decode(url.split(",", 1)[1], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                width, height = image.size
                if getattr(image, "n_frames", 1) != 1 or max(width, height) / min(width, height) > 200:
                    raise ValueError("Only static images with aspect ratio <= 200 are supported")
            images.append({
                "sha256": hashlib.sha256(raw).hexdigest(),
                "width": width,
                "height": height,
                "visual_tokens_upper_bound": image_token_bound(width, height),
            })
    image_bound = sum(row["visual_tokens_upper_bound"] for row in images)
    framing = 1024 + 128 * len(messages) + 16 * parts
    maximum = text_bytes + framing + image_bound + reserve
    return {
        "method": "Conservative local UTF-8 and image-grid estimate",
        "text_bytes": text_bytes,
        "framing_allowance": framing,
        "image_tokens_upper_bound": image_bound,
        "images": images,
        "output_reserved": reserve,
        "maximum": maximum,
        "limit": OUTPUT_LIMIT,
        "enforced": True,
        "context_policy": CONTEXT_POLICY,
    }

def image_token_bound(width, height):
    # Round original dimensions up conservatively without assuming server downsampling.
    return max(256, math.ceil(width / 28) * math.ceil(height / 28)) + 16


class ContextCapacityExceeded(RuntimeError):
    """The request was rejected before HTTP."""


class ProviderCapacityViolation(RuntimeError):
    """Provider usage violated the local request or context bound."""


class SiliconFlowRequestError(RuntimeError):
    """The endpoint returned a definite non-200 response."""

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
                name = f"shared_{len(names)}"
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

def complete_validation_feedback(error):
    if not isinstance(error, ValidationError):
        return [{"path": [], "error": str(error)}]
    if error.context:
        return [row for child in error.context for row in complete_validation_feedback(child)]
    return [{"path": list(error.absolute_path), "schema_path": list(error.absolute_schema_path),
             "validator": error.validator, "error": error.message}]

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

def read_response(response, model, preflight, reserve):
    if response.status_code != 200:
        raise SiliconFlowRequestError(f"SiliconFlow HTTP {response.status_code}; stopped without retry")
    value = response.json()
    if value.get("model") != model:
        raise RuntimeError(f"Unexpected Qwen response model: {value.get('model')}")
    choice = value["choices"][0]
    usage = value.get("usage")
    if not isinstance(usage, dict) or any(
            type(usage.get(name)) is not int or usage[name] < 0
            for name in ("prompt_tokens", "completion_tokens")):
        raise ProviderCapacityViolation("Provider did not report valid prompt/completion token usage")
    prompt_tokens, completion_tokens = usage["prompt_tokens"], usage["completion_tokens"]
    if prompt_tokens + reserve > OUTPUT_LIMIT or completion_tokens > reserve:
        raise ProviderCapacityViolation("Provider usage exceeded the reserved context window or output budget")
    if prompt_tokens > preflight["maximum"] - reserve:
        raise ProviderCapacityViolation("Provider usage exceeded the conservative local input estimate")
    if choice.get("finish_reason") == "length":
        raise ValueError("Output reached max_tokens. Return compact JSON within the same schema.")
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


class DirectCalls:
    def __init__(self, root):
        self.root = root
        self.key = API_KEY
        self.record_only = True
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
            "You are the SAME grounding Qwen, performing ONE final tentative evaluation after no progress. "
            "This is an explicitly user-authorized TENTATIVE exit, not a strict Monitor finish. "
            "Do not emit or complete Monitor nodes, exclusions or rollback. The accepted plan and all "
            "other active candidates remain unchanged. Read the full original description, raw "
            "evidence and actual attached images. Choose the best-supported target ONLY from "
            "legal_current_prediction_ids, compare your strongest remaining rival(s), cite original "
            "evidence covering each claim, and state unresolved assumptions/uncertainty. If no "
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
                feedback["previous_raw_answer"] = json.loads(raw.read_text())["choices"][0]["message"]["content"]
                visible = deepcopy(original_visible) + [{"role": "user", "content": packed(feedback)}]
    def _qwen(self, tag, role, messages, schema, reserve=OUTPUT_RESERVE, wire_schema=None, quote_registry=None):
        messages = deepcopy(messages)
        wire_schema = provider_schema(factor_schema(wire_schema if wire_schema is not None else schema))
        Draft202012Validator.check_schema(wire_schema)
        messages[0]['content'] += ('\nReturn only a JSON object matching this JSON Schema. '
                                   'Do not use Markdown or extra text.\n' + packed(wire_schema))
        if 'action' in schema.get('properties', {}) and 'state_version' in schema['properties']:
            version = schema['properties']['state_version']['const']
            messages.append({'role': 'user', 'content': (
                f'Use exactly two top-level keys: "state_version" set to "{version}", '
                'and "action" containing the complete action object.')})
        preflight = qwen_preflight(messages, reserve)
        if self.bootstrap_only:
            if preflight['maximum'] > OUTPUT_LIMIT:
                raise ContextCapacityExceeded('Offline parser request exceeds the local context bound')
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
                'annotation_target_supplied': False, 'real_provider_schema_validated': False,
                'preflight': preflight})
            check = json.loads((self.root / 'bootstrap-check.json').read_text())
            check['actual_inputs'] = self.bootstrap_inputs
            save(self.root / 'bootstrap-check.json', check)
            raise BootstrapReady('Offline bootstrap completed before any HTTP attempt')
        record = {'index': len(self.ledger) + 1, 'tag': tag, 'role': role, 'requested_model': MODEL, 'http_sent': False, 'preflight': preflight, 'output_reserved': reserve, 'status': 'preflight'}
        self.ledger.append(record)
        folder = self.root / 'calls' / f"{record['index']:03d}-{tag}-{role}"
        folder.mkdir(parents=True, exist_ok=False)
        payload = {'model': MODEL, 'messages': messages, 'max_tokens': reserve, 'temperature': 0}
        save(folder / 'request-redacted.json', redact(payload))
        save(folder / 'original-response-schema.json', schema)
        record['input_images'] = []
        if preflight['maximum'] > OUTPUT_LIMIT:
            record.update(status='rejected_preflight', error='capacity_bound_exceeds_limit')
            save(folder / 'receipt.json', record)
            save(self.root / 'model-ledger.json', self.ledger)
            raise ContextCapacityExceeded(f"Local context estimate {preflight['maximum']} > {OUTPUT_LIMIT}; no request sent")
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
        save(folder / 'receipt.json', record)
        charged_role = role in ('semantic_consultant', 'image_selector', 'parser_repair')
        if charged_role:
            if not self.record_only and self.r_used >= 6:
                record.update(status='rejected_budget', error='R_budget_exhausted')
                save(folder / 'receipt.json', record)
                save(self.root / 'model-ledger.json', self.ledger)
                raise RuntimeError('R budget exhausted before Qwen request')
            self.r_used += 1
        started = time.monotonic()
        record['http_sent'] = True
        record['status'] = 'sent'
        save(folder / 'receipt.json', record)
        self.progress("request_sent", call=record['index'], role=role,
                      context_bound=preflight['maximum'], context_limit=OUTPUT_LIMIT)
        try:
            response = self.session.post(BASE_URL + '/chat/completions', json=payload, headers={'Authorization': 'Bearer ' + self.key}, timeout=(45, 180), allow_redirects=False)
            (folder / 'provider-response.json').write_text(redact(response.text))
            record['http_status'] = response.status_code
            try:
                record['usage'] = response.json().get('usage')
            except (ValueError, AttributeError):
                record['usage'] = None
            save(folder / 'receipt.json', record)
            value = read_response(response, MODEL, preflight, reserve)
            content = (value['choices'][0]['message'].get('content') or '').strip()
            fence = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n\s*```', content, re.IGNORECASE | re.DOTALL)
            if fence:
                content = fence[1].strip()
            answer = json.loads(content, object_pairs_hook=unique_json_object)
            version = schema.get("properties", {}).get("state_version", {}).get("const")
            if (version is not None and isinstance(answer, dict) and set(answer) == {"action"}
                    and isinstance(answer["action"], dict) and answer["action"].get("state_version") == version):
                answer["state_version"] = answer["action"].pop("state_version")
                content = packed(answer)
                record['response_normalization'] = 'lifted_matching_action_state_version'
            validate(answer, schema)
            record.update(status='completed', usage=value.get('usage'), finish_reason=value['choices'][0].get('finish_reason'), request_id=response.headers.get('x-request-id'))
            (folder / 'response.json').write_text(content)
            save(folder / 'receipt.json', record)
            save(self.root / 'model-ledger.json', self.ledger)
            self.progress("response_saved", call=record['index'], role=role, http_status=response.status_code,
                          prompt_tokens=(value.get('usage') or {}).get('prompt_tokens'),
                          completion_tokens=(value.get('usage') or {}).get('completion_tokens'))
            return {'_content': content, 'usage': value.get('usage'), '_finish_reason': value['choices'][0].get('finish_reason'), '_request': {'model': MODEL, 'response_model': value.get('model'), 'http_status': response.status_code, 'request_id': record.get('request_id'), 'latency_s': round(time.monotonic() - started, 4), 'preflight': preflight}}
        except Exception as exc:
            record.update(status='failed', error_type=type(exc).__name__, error=redact(str(exc))[:1000], latency_s=round(time.monotonic() - started, 4))
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
        # Keep every original text row and every actual image; no Laya input cap.
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
        decision = {"T": self.t_used, "state_version": version, "R_before": self.r_used,
                    "mode": "single_qwen_direct_action", "image_policy": "all_actual_images"}
        self.decisions.append(decision)
        try:
            while True:
                call_index = len(self.ledger) + 1
                try:
                    response = self._qwen(f"t{self.t_used:02d}-action-{attempt}", "semantic_consultant",
                                          visible, wire, reserve=OUTPUT_RESERVE, wire_schema=wire)
                    answer = json.loads(response["_content"], object_pairs_hook=unique_json_object)
                    action = answer["action"]
                    before_order = deepcopy(action["monitor"])
                    if "node_updates" in action["monitor"]:
                        action["monitor"] = canonical_monitor_updates(action["monitor"])
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
                                "instruction": "Correct your explicit action/Monitor against the original native contract. Same state, evidence and images. No new tool executed. AnyOf branches are alternatives. No program repair."}
                    save(folder / f"repair-{attempt}.json", feedback)
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
            raise ValueError("Rival list contains predicted target")


def build_frozen_manifest(*, include_single_sample_inputs=True):
    sources = sorted([*PROJECT.glob("src/**/*.py"), Path(__file__),
                      PROJECT / "tests/unit/test_qwen_direct_fresh.py",
                      PROJECT / "tests/fixtures/qwen_direct_q00171_progress.json",
                      PROJECT / "docs/qwen-direct-v3-loop-fix.md",
                      PROJECT / "docs/qwen-direct-protocol.md",
                      PROJECT / "docs/qwen-direct-source-provenance.json", PROJECT / "pyproject.toml"])
    files = [INPUTS / "nr3d/scene0565_00/scene.npz", INPUTS / "nr3d/scene0565_00/scene_instances.json",
             INPUTS / "nr3d/query.json", INPUTS / "categories/scene0565_00/objects_top5.json"]
    return {"candidate": "single-qwen3vl-siliconflow-256k", "branch": "20261121_step1_qwen3vl",
            "baseline_commit": "8a44a23897687d8fb35a766b71bf67c008767940",
            "model": MODEL, "endpoint": BASE_URL, "scheduler_sha256": sha256(Path(__file__)),
            "source_closure": {str(p.relative_to(PROJECT)): sha256(p) for p in sources},
            "inputs": {str(p.relative_to(PROJECT)): sha256(p) for p in files} if include_single_sample_inputs else {},
            "python": sys.version, "packages": {d.metadata["Name"]: d.version for d in distributions() if d.metadata.get("Name")},
            "resource_budget_policy": {"mode": "record-only", "B_limit": None, "R_limit": None, "T_limit": None,
                                       "annealing_reference_turns_not_a_cap": 17},
            "qwen_context_policy": {"policy": CONTEXT_POLICY, "local_262144_refusal_enforced": True,
                                    "reference_bound": OUTPUT_LIMIT, "output_reserve": OUTPUT_RESERVE,
                                    "input_count_policy": "Conservative text/image estimate before sending, provider usage checked after response",
                                    "overflow_policy": "Record capacity rejection and retain source evidence without truncation"},
            "model_visibility": "Full original native text/history and ALL actual attached original images per consultation; no Laya cap, selector or extra reviewer. Each image copied with SHA in receipt. Current native Monitor plan/revision and latest rejection/rollback in one candidate block; full original observation/action history remains delivered.",
            "decision_protocol": "Same Qwen direct native action/Monitor with exact compact delta/replace_plan schema visible and strict path unchanged. Historical acquisition coverage (sorted groups/context/15-degree bins/new visible IDs/raw measurement facts and frames) plus accepted Monitor states detects three consecutive no-progress transitions; engineering proxy only. Then, or on exact repeated native rejection, same Qwen once selects a tentative target from allowed∩active with at most one structural repair. No call cap. Native Monitor untouched, tentative accepted=false.",
            "credential_policy": "SiliconFlow per-process Bearer key, excluded from saved artifacts. JSON Schema in the prompt, validated locally.",
            "scope": "Diagnostic new single run; not strict B10/R6/T17 comparison, formal score or paper update.",
            "fresh_start": True, "old_runtime_source_imports": [], "inherited_dag_answers_predictions": [],
            "server_model_requests": 0, "gt_read_before_prediction": False}


def run(root, batch_request=None, bootstrap_check=False):
    global API_KEY
    API_KEY = '' if bootstrap_check else os.environ.pop(SILICONFLOW_KEY_ENV, '').strip()
    if not bootstrap_check and not API_KEY:
        raise ValueError(f'Missing {SILICONFLOW_KEY_ENV}')
    if root.exists():
        raise FileExistsError(f"Run directory already exists: {root}")
    root.mkdir(parents=True)
    calls = DirectCalls(root)
    calls.bootstrap_only = bootstrap_check
    request = None
    if batch_request is not None:
        request = json.loads(Path(batch_request).read_text())
        if set(request) != {"sample", "scene_dir", "category_path", "shared_manifest", "input_hashes"} or set(request["sample"]) != {"scene", "csv_row", "query"}:
            raise ValueError("Unexpected worker input fields; target annotations are forbidden")
        shared = json.loads(Path(request["shared_manifest"]).read_text())
        protocol = shared["protocol"]
        if (protocol.get("model") != MODEL or protocol.get("endpoint") != BASE_URL
                or protocol.get("qwen_context_policy", {}).get("reference_bound") != OUTPUT_LIMIT):
            raise ValueError("Frozen worker model, endpoint or context limit does not match this branch")
        manifest = {**shared["protocol"], "source_closure": shared["source_closure"],
                    "shared_manifest": str(Path(request["shared_manifest"]).resolve()),
                    "shared_manifest_sha256": sha256(Path(request["shared_manifest"])),
                    "sample": request["sample"], "inputs": request["input_hashes"]}
        for name, expected in request["input_hashes"].items():
            if sha256(Path(name)) != expected:
                raise ValueError(f"Worker input changed: {name}")
        calls.console_progress = True
    else:
        manifest = build_frozen_manifest()
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
    except ContextCapacityExceeded as exc:
        status = "capacity_rejected"
        prediction = {"predicted_id": None, "failure_type": type(exc).__name__, "failure": str(exc)}
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
