"""Run frozen ablation conditions with API keys supplied only through stdin."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STUDY = Path(os.getenv("EVIGROUND_ABLATION_STUDY", ROOT / "experiments/ablation_20261104"))
MODEL = os.getenv("EVIGROUND_ABLATION_MODEL", "gpt-6.1-sol")
BASE_URL = os.getenv("EVIGROUND_ABLATION_BASE_URL", os.getenv("EVIGROUND_BASE_URL", "https://api.openai.com/v1"))
SCENES = Path(os.getenv("EVIGROUND_ABLATION_SCENES", ROOT / "local_outputs"))
EXPORT = Path(os.getenv("EVIGROUND_ABLATION_PREDICTIONS", ROOT / "local_outputs/classifier_predictions"))
DAG_MONITOR = os.getenv("EVIGROUND_ABLATION_DAG_MONITOR_ENABLED", "0").strip()
if DAG_MONITOR not in ("0", "1"):
    raise ValueError("EVIGROUND_ABLATION_DAG_MONITOR_ENABLED must be 0 or 1")
VARIANTS = {
    "E0": {"name": "Full EviGround", "inspect": 1, "measure": 1, "top_k": 5, "guard": 1},
    "E1": {"name": "Without supplementary inspection", "inspect": 0, "measure": 1, "top_k": 5, "guard": 0},
    "E2": {"name": "Without conditional geometry", "inspect": 1, "measure": 0, "top_k": 5, "guard": 1},
    "E3": {"name": "Top-1 category only", "inspect": 1, "measure": 1, "top_k": 1, "guard": 0},
    "E4": {"name": "Without alternative-category inspection check", "inspect": 1, "measure": 1, "top_k": 5, "guard": 0},
    "E5": {"name": "Without dynamic evidence acquisition", "inspect": 0, "measure": 0, "top_k": 5, "guard": 0},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, document, frozen=False):
    text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if frozen and path.exists() and path.read_text() != text:
        raise ValueError(f"Frozen input changed: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def environment(key, variant, workers):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("EVIGROUND_", "SEEGROUND_"))}
    env.update({"PYTHONPATH": str(STUDY / "source"), "PYTHON_DOTENV_DISABLED": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "EVIGROUND_API_PROVIDER": "openai",
                "EVIGROUND_BASE_URL": BASE_URL,
                "EVIGROUND_MODEL": MODEL, "EVIGROUND_API_KEY": key,
                "EVIGROUND_ALLOW_HTTP": "1" if BASE_URL.startswith("http://") else "0",
                "EVIGROUND_ENABLE_THINKING": "1",
                "EVIGROUND_THINKING_TOKEN_FLOOR": "8192",
                "EVIGROUND_OPENAI_TOKEN_PARAMETER": "max_completion_tokens",
                "EVIGROUND_OPENAI_TEMPERATURE": "0", "EVIGROUND_CONTEXT_LIMIT": "131072",
                "EVIGROUND_SYNTAX_BACKEND": "spacy", "EVIGROUND_API_CONCURRENCY": str(workers),
                "EVIGROUND_INSPECT_ENABLED": str(variant["inspect"]),
                "EVIGROUND_MEASURE_ENABLED": str(variant["measure"]),
                "EVIGROUND_CATEGORY_INSPECTION_REQUIRED": str(variant["guard"])})
    env["EVIGROUND_DAG_MONITOR_ENABLED"] = DAG_MONITOR
    env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
    return env


def freeze_inputs(workers):
    source_hashes = {}
    for path in sorted((ROOT / "src/eviground").rglob("*.py")):
        if path.name.startswith("._"):
            continue
        relative = path.relative_to(ROOT / "src")
        target = STUDY / "source" / relative
        raw = path.read_bytes()
        if target.exists() and target.read_bytes() != raw:
            raise ValueError(f"Inference code changed after snapshot: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(raw)
        source_hashes[str(relative)] = hashlib.sha256(raw).hexdigest()
    save(STUDY / "source_identity.json", source_hashes, frozen=True)
    source = STUDY / "nr3d_subset_1500.json"
    subset = json.loads(source.read_text())
    if len(subset["rows"]) != 1500 or len({r["scene"] for r in subset["rows"]}) != 130:
        raise ValueError("Unexpected frozen subset")
    manifests = []
    for slot in range(2):
        path = STUDY / f"nr3d_shard_{slot + 1}.json"
        save(path, {**subset, "rows": subset["rows"][slot::2]}, frozen=True)
        manifests.append(path)
    # Technical pilots use two fixed descriptions outside the evaluation subset.
    selected = {r["csv_row"] for r in subset["rows"]}
    full = json.loads((ROOT / "data/queries/nr3d_val.json").read_text())
    pilot = [r for r in full["rows"] if r["csv_row"] not in selected][:2]
    for slot in range(2):
        save(STUDY / f"pilot_shard_{slot + 1}.json", {**full, "rows": [pilot[slot]]}, frozen=True)
    protocol = {
        "dataset": "nr3d", "model": MODEL,
        "base_url": BASE_URL, "queries": 1500,
        "conditions": VARIANTS, "nonterminal_evidence_budget": 5,
        "dag_monitor_enabled": DAG_MONITOR == "1",
        "controller": "dynamic_dag_monitor" if DAG_MONITOR == "1" else "react",
        "initial_evidence": "same overview, top-down view and basic AABB geometry in all conditions",
        "subset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sampling": subset["sampling"], "sharding": "alternating fixed csv_row order; same key slot across conditions",
        "workers_per_key": workers, "total_workers": 2 * workers,
        "thinking_enabled": True, "completion_token_floor": 8192, "temperature": 0,
        "primary_comparisons": "E0 minus each E1-E5 on exactly the same 1500 descriptions",
        "auxiliary_comparisons": ["E4-E1", "E4-E3"],
        "evaluation_policy": "one pass; nulls and errors included; no accuracy-driven retries or sample changes",
        "transient_retry_policy": "at most one extra API attempt per trajectory",
        "pilot_csv_rows": [r["csv_row"] for r in pilot],
        "pilot_policy": "technical verification only; excluded from 1500-query evaluation",
        "reference_policy": "Qwen full-set result is separately labeled and never substituted for GPT E0",
    }
    protocol_path = STUDY / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        protocol["workers_per_key"] = previous["workers_per_key"]
        protocol["total_workers"] = previous["total_workers"]
    save(protocol_path, protocol, frozen=True)
    return manifests


def preflight(keys, workers):
    """Exercise the actual client, image input and strict JSON contract for both slots."""
    script = '''
import json
from pathlib import Path
from PIL import Image
from eviground.api.client import chat, image_part, _redact
from eviground.api.response_schemas import obj, TEXT, schema_kwargs
import sys
p=Path(sys.argv[1]); Image.new("RGB", (48,48), (230,20,20)).save(p)
try:
    data=chat([{"role":"user","content":[{"type":"text","text":"Name the image's dominant color. Return JSON with color."}, image_part(p)]}], max_tokens=512, json_mode=True, **schema_kwargs("vision_preflight", obj({"color":TEXT})))
    print(json.dumps({"ok":True,"requested_model":data["_request"]["model"],"response_model":data.get("model"),"content":data["_content"],"finish_reason":data.get("_finish_reason"),"usage":data.get("usage")}))
except Exception as exc:
    print(json.dumps({"ok":False,"error":_redact(str(exc))}));sys.exit(1)
'''
    def check(slot):
        process = subprocess.run([sys.executable, "-c", script, str(STUDY / f"preflight_{slot+1}.png")],
                                 env=environment(keys[slot], VARIANTS["E0"], workers),
                                 capture_output=True, text=True, cwd=ROOT)
        try:
            result = json.loads(process.stdout)
        except ValueError:
            result = {"ok": False, "error": "Preflight process failed before producing diagnostics"}
        save(STUDY / f"preflight_key_{slot+1}.json", result)
        print(f"key slot {slot+1} preflight: {result}", flush=True)
        if (process.returncode or not result.get("ok") or result.get("finish_reason") != "stop"
                or "red" not in result.get("content", "").lower()
                or result.get("response_model") != MODEL):
            raise ValueError(f"Key slot {slot+1} failed model/vision/schema preflight")
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(check, range(2)))


def run_pair(variant_id, keys, manifests, workers, phase, prepare=False):
    variant = VARIANTS[variant_id]
    processes, handles, outputs = [], [], []
    try:
        for slot, manifest in enumerate(manifests):
            output = STUDY / phase / variant_id / f"key_slot_{slot+1}"
            output.mkdir(parents=True, exist_ok=True)
            handle = (output / ("prepare.log" if prepare else "batch.log")).open("a")
            handles.append(handle); outputs.append(output)
            command = [sys.executable, "-u", "-m", "eviground.cli.ground",
                       "--manifest", str(manifest), "--predictions", str(EXPORT),
                       "--scene-root", str(SCENES), "--output", str(output),
                       "--category-top-k", str(variant["top_k"]), "--review-limit", "5",
                       "--workers", str(workers), "--api-concurrency", str(workers)]
            if prepare:
                command.append("--prepare-only")
            processes.append(subprocess.Popen(command, env=environment(keys[slot], variant, workers),
                                              cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                              start_new_session=True))
        while any(p.poll() is None for p in processes):
            progress = []
            for output in outputs:
                result_file = output / "results.json"
                progress.append(json.loads(result_file.read_text())["summary"] if result_file.exists() else {})
            save(STUDY / "status.json", {"updated_at": now(), "phase": phase,
                                         "condition": variant_id, "prepare_only": prepare,
                                         "pid": os.getpid(), "progress": progress})
            print(f"{now()} {phase}/{variant_id} prepare={prepare} "
                  f"processed={[p.get('processed', 0) for p in progress]}", flush=True)
            time.sleep(15)
        if any(p.returncode for p in processes):
            raise ValueError(f"{phase}/{variant_id} process failed; inspect batch logs")
        return [json.loads((output / "results.json").read_text()) for output in outputs]
    finally:
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for handle in handles:
            handle.close()


def aggregate(variant_id, documents):
    from eviground.evaluation.reporting import report
    rows = sorted([r for d in documents for r in d["results"]], key=lambda r: r["csv_row"])
    expected = json.loads((STUDY / "nr3d_subset_1500.json").read_text())["rows"]
    if [r["csv_row"] for r in rows] != [r["csv_row"] for r in expected]:
        raise ValueError("Aggregated results do not match the frozen selection")
    output = STUDY / "runs" / variant_id
    summary = report(output, rows, 1500)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers-per-key", type=int, default=3)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Use the two successful preflights already performed with the same keys")
    parser.add_argument("--skip-technical-pilot", action="store_true",
                        help="Use all twelve completed technical pilots already verified")
    parser.add_argument("--start-condition", choices=tuple(VARIANTS), default="E0")
    parser.add_argument("--end-condition", choices=tuple(VARIANTS), default="E5")
    parser.add_argument("--detach", action="store_true",
                        help="Continue in a separate process with persistent runner.log")
    args = parser.parse_args()
    if not 1 <= args.workers_per_key <= 4:
        parser.error("workers-per-key must be 1-4")
    STUDY.mkdir(parents=True, exist_ok=True)
    manifests = freeze_inputs(args.workers_per_key)
    print("Waiting for a JSON array of one or two API keys on stdin (not persisted).", flush=True)
    keys = json.loads(sys.stdin.readline())
    if (not isinstance(keys, list) or len(keys) not in (1, 2)
            or any(not isinstance(k, str) or not k for k in keys)):
        raise ValueError("Supply one key shared by both shards, or two API keys")
    if len(keys) == 1:
        keys = [keys[0], keys[0]]
    if args.detach:
        command = [sys.executable, "-u", *[arg for arg in sys.argv if arg != "--detach"]]
        with (STUDY / "runner.log").open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        process.stdin.write((json.dumps(keys) + "\n").encode())
        process.stdin.close()
        save(STUDY / "runner_process.json", {"pid": process.pid, "started_at": now()})
        print(f"Detached experiment runner started: pid={process.pid}", flush=True)
        return
    if args.skip_preflight:
        for slot in range(2):
            result = json.loads((STUDY / f"preflight_key_{slot+1}.json").read_text())
            if not result.get("ok") or result.get("response_model") != MODEL:
                raise ValueError("A successful model preflight is required for both slots")
    else:
        preflight(keys, args.workers_per_key)
    if args.preflight_only:
        return
    condition_ids = list(VARIANTS)
    first, last = condition_ids.index(args.start_condition), condition_ids.index(args.end_condition)
    if first > last:
        raise ValueError("start-condition must not follow end-condition")
    selected_conditions = condition_ids[first:last + 1]
    # Validate all 130 scenes before any grounding request.
    if args.start_condition == "E0":
        run_pair("E0", keys, manifests, args.workers_per_key, "runs", prepare=True)
    pilots = [STUDY / f"pilot_shard_{i+1}.json" for i in range(2)]
    for variant_id in selected_conditions:
        if args.skip_technical_pilot:
            documents = [json.loads((STUDY / "technical_pilot" / variant_id /
                                     f"key_slot_{i+1}/results.json").read_text()) for i in range(2)]
        else:
            documents = run_pair(variant_id, keys, pilots, args.workers_per_key, "technical_pilot")
        if any(not d["summary"]["complete"] or d["summary"]["errors"] for d in documents):
            raise ValueError(f"Technical pilot failed for {variant_id}; final evaluation not started")
    summaries = {}
    for variant_id in selected_conditions:
        documents = run_pair(variant_id, keys, manifests, args.workers_per_key, "runs")
        summaries[variant_id] = aggregate(variant_id, documents)
        save(STUDY / "condition_summaries.json", summaries)
    if selected_conditions == condition_ids:
        baseline = json.loads((STUDY / "runs/E0/results.json").read_text())["results"]
        comparisons = {}
        for variant_id in VARIANTS:
            rows = json.loads((STUDY / "runs" / variant_id / "results.json").read_text())["results"]
            comparisons[variant_id] = {
                "accuracy": summaries[variant_id]["accuracy"],
                "e0_minus_condition_pp": 100 * (summaries["E0"]["accuracy"] - summaries[variant_id]["accuracy"]),
                "e0_only_correct": sum(a["correct"] and not b["correct"] for a, b in zip(baseline, rows)),
                "condition_only_correct": sum(b["correct"] and not a["correct"] for a, b in zip(baseline, rows)),
            }
        save(STUDY / "paired_comparisons.json", comparisons)
        phase = "complete"
    else:
        comparisons = {}
        phase = "partial_complete"
    save(STUDY / "status.json", {"updated_at": now(), "phase": phase,
                                 "conditions": selected_conditions, "summaries": summaries})
    print(json.dumps(comparisons or summaries, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        save(STUDY / "status.json", {"updated_at": now(), "phase": "technical_failure",
                                     "error_type": type(exc).__name__, "message": str(exc)})
        raise
