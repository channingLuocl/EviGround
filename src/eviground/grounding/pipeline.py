"""Predicted-category retrieval followed by bounded visual ReAct inference."""

import json
import os
from pathlib import Path
from time import perf_counter
import uuid
import numpy as np
from eviground.api.client import _redact, api_config_summary, api_provider
from eviground.categories.hypotheses import (
    category_names,
    load_category_objects,
    sha256,
)
from eviground.evaluation.run_storage import ensure_run_writable, write_json
from eviground.categories.retrieval import match_retrieval_categories
from eviground.grounding.bindings import generic_target_pool
from eviground.grounding.react import ground
from eviground.grounding.settings import settings
from eviground.grounding.recovery import (
    sample_recovery,
    input_identity,
    file_identity,
    save_checkpoint,
    actual_attempts,
)
from eviground.query.parser import parse_query
from eviground.rendering.renderer import load_objects


def structural_objects(scene_dir, objects, excluded_ids=()):
    """Supplement geometry only; annotated names never reach inference."""
    known = {o["bbox_id"] for o in objects} | set(excluded_ids)
    return [
        {"bbox_id": o["bbox_id"], "bbox_3d": o["bbox_3d"], "target": "object"}
        for o in load_objects(scene_dir / "scene_instances.json")
        if o["bbox_id"] not in known
    ]


def run_query(scene_dir, query, out, top_k=5, review_limit=3):
    started = perf_counter()
    scene_dir, out = (Path(scene_dir), Path(out))
    ensure_run_writable(out)
    if top_k not in (1, 5) or review_limit < 1:
        raise ValueError("Expected top_k=1|5 and a positive tool budget")
    out.mkdir(parents=True, exist_ok=True)
    old_run = json.loads((out / 'run.json').read_text()) if (out / 'run.json').exists() else {}
    previous = old_run.get('previous_execution_costs', [])
    if old_run:
        previous = [*previous, old_run.get('execution_cost', {
            'online_latency_complete': False, 'online_latency_observed_s': old_run.get('pipeline_latency_s', 0)})]
    calls = []
    execution_id = uuid.uuid4().hex
    token = sample_recovery.set({"used": False, "attempts": [], 'execution_id': execution_id})
    record = {
        "query": query,
        "calls": calls,
        "category_top_k": top_k,
        "gt_categories_supplied_to_model": False,
        "query_target_id_supplied_to_model": False,
        "model": os.getenv("EVIGROUND_MODEL"),
        "category_cache_root": os.getenv("EVIGROUND_CATEGORY_ROOT"),
        "stage_timings_s": {},
        "execution_id": execution_id,
        "execution_complete": False,
        "previous_execution_costs": previous,
    }
    stage, stage_started = "load_categories", perf_counter()

    def begin_stage(name):
        nonlocal stage, stage_started
        now = perf_counter()
        record["stage_timings_s"][stage] = now - stage_started
        stage, stage_started = name, now

    def save_progress(complete=False):
        from eviground.api.usage_log import read_attempts
        from eviground.evaluation.costs import sample_cost

        elapsed = perf_counter() - started
        record['stage_timings_s'][stage] = perf_counter() - stage_started
        record['execution_complete'] = complete
        record['online_latency_observed_s'] = elapsed + sum(
            c.get('online_latency_observed_s') or 0 for c in previous)
        record['online_latency_s'] = record['online_latency_observed_s']
        record['http_attempt_records'] = read_attempts(out / 'http_attempts.jsonl')
        record['execution_cost'] = sample_cost({**record, 'previous_execution_costs': [],
            'online_latency_s': elapsed, 'online_latency_observed_s': elapsed})
        record['cost'] = sample_cost(record)
        record['online_latency_s'] = record['cost']['online_latency_s']
        write_json(out / 'run.json', _redact(record, limit=None))

    sample_recovery.get()['save_progress'] = save_progress
    try:
        save_progress()
        record['api_config'] = api_config_summary()
        category_root = os.getenv("EVIGROUND_CATEGORY_ROOT")
        if not category_root:
            raise ValueError("Prepared classifier category root is required")
        objects_path = Path(category_root) / scene_dir.name / f"objects_top{top_k}.json"
        objects = load_category_objects(objects_path)
        table = json.loads(objects_path.read_text())
        record["category_input"] = {key: table[key] for key in (
            "version", "gate", "source_predictions_sha256", "cloud_sha256", "category_source",
            "frontend_protocol_sha256") if key in table}
        record["category_input"]["table_sha256"] = sha256(objects_path)
        supplemental = structural_objects(scene_dir, objects, table.get("suppressed_instance_ids", ()))
        geometry_objects = objects + supplemental
        begin_stage("load_scene")
        if api_provider() != "openai":
            raise ValueError("EviGround requires the OpenAI-compatible multimodal API")
        with np.load(scene_dir / "scene.npz", allow_pickle=False) as cloud:
            xyz, rgb, ids = (cloud["xyz"], cloud["rgb"], cloud["ids"])
        config = {
            "api": api_config_summary(),
            "category_top_k": top_k,
            "review_limit": review_limit,
            "policy": settings(),
            "model": os.getenv("EVIGROUND_MODEL"),
            "base_url": os.getenv("EVIGROUND_BASE_URL"),
            "syntax_backend": os.getenv("EVIGROUND_SYNTAX_BACKEND", "spacy"),
            "scene_sha256": file_identity(
                (
                    scene_dir / "scene.npz",
                    objects_path,
                    scene_dir / "scene_instances.json",
                )
            ),
        }
        if config['scene_sha256']['scene.npz'] != table['cloud_sha256']:
            raise ValueError('Category cache belongs to a different scene point cloud')
        if table.get('category_source') in ('seeground_mask3d_scannet200', 'seeground_pointnext_clip_reproduction'):
            for name, field in (('scene_instances.json', 'instances_sha256'), ('masks.npz', 'masks_sha256'),
                                ('objects.json', 'candidate_geometry_sha256')):
                if field in table and sha256(scene_dir / name) != table[field]:
                    raise ValueError(f'Native category geometry changed: {name}')
        identity = input_identity(query, geometry_objects, config)
        pending = out / "pending_inference.json"
        if (
            pending.exists()
            and json.loads(pending.read_text()).get("checkpoint_identity") != identity
        ):
            raise ValueError(
                "Pending inference has different code, inputs or configuration"
            )
        budget = sample_recovery.get()
        budget.update(
            api_checkpoint=str(out / "api_recovery.json"),
            checkpoint_identity=identity,
            completed_requests={},
            resuming=False,
        )
        if (out / "api_recovery.json").exists():
            saved = json.loads((out / "api_recovery.json").read_text())
            if saved.get("checkpoint_identity") != identity or saved["used"]:
                raise ValueError(
                    "Saved API recovery changed inputs or exhausted its retry budget"
                )
            budget.update(saved, resuming=True, manual_resume_pending=True)
        save_checkpoint(pending, {"checkpoint_identity": identity})
        begin_stage("parse_query")
        budget['call_stage'] = 'parse_query'
        save_progress()
        parsed = parse_query(
            query,
            calls=calls,
            scene_categories=[name for o in objects for name in category_names(o)],
        )
        begin_stage("retrieve_objects")
        candidates = match_retrieval_categories(parsed["Target"], objects)
        if generic_target_pool(parsed["Target"]):
            candidates = list(objects)
        references = match_retrieval_categories(parsed["Anchor"], geometry_objects)
        parsed["candidate_generation"] = {
            "status": "bound" if candidates else "unknown",
            "source": "predicted_category_hypotheses",
            "category_pool_ids": [o["bbox_id"] for o in candidates],
        }
        trace = {
            "decision_backend": "openai",
            "planner_input": "text_and_scene_images",
            "steps": [],
            "internal_usage": {"compute": 0, "render": 0, "visual_check": 0},
        }
        record["active_trace"] = trace
        begin_stage("react_grounding")
        if candidates:
            prediction = ground(
                query,
                parsed,
                geometry_objects,
                candidates,
                xyz,
                rgb,
                ids,
                out,
                calls,
                trace,
                review_limit,
            )
        else:
            reason = "no retained category hypothesis matches the parsed target"
            trace.update(
                stop_reason="category_abstention",
                final_evidence_state={"status": "missing", "missing": [reason]},
            )
            prediction = {
                "query": query,
                "predicted_id": None,
                "bbox_3d": None,
                "implementation_version": settings()["version"],
                "prediction_source": "abstained",
                "decision_status": "missing",
                "active_trace": trace,
            }
        table = json.loads(objects_path.read_text())
        prediction["category_input"] = {
            key: table[key]
            for key in ("version", "gate", "source_predictions_sha256", "cloud_sha256")
        }
        prediction["category_input"]["table_sha256"] = sha256(objects_path)
        for key in ("category_source", "frontend_protocol_sha256"):
            if key in table:
                prediction["category_input"][key] = table[key]
        prediction["category_target_ids"] = [o["bbox_id"] for o in candidates]
        prediction["category_anchor_ids"] = [o["bbox_id"] for o in references]
        prediction["category_bindings"] = {
            role: [
                {
                    "bbox_id": o["bbox_id"],
                    "matched_hypotheses": [
                        h
                        for h in o.get("category_hypotheses", [])
                        if match_retrieval_categories(
                            parsed[field], [{**o, "category_hypotheses": [h]}]
                        )
                    ],
                }
                for o in pool
            ]
            for role, field, pool in (
                ("targets", "Target", candidates),
                ("references", "Anchor", references),
            )
        }
        prediction.update(
            category_top_k=top_k,
            gt_categories_supplied_to_model=False,
            query_target_id_supplied_to_model=False,
            pipeline_latency_s=round(perf_counter() - started, 4),
            api_calls=len(calls),
            api_attempts=sum((actual_attempts(c) for c in calls)),
            call_usage=[c.get("usage") for c in calls],
            recovery_attempts=budget["attempts"],
        )
        record["parsed_query"] = parsed
        begin_stage("save_results")
        write_json(out / "prediction.json", prediction)
        for name in ("pending_inference.json", "api_recovery.json"):
            (out / name).unlink(missing_ok=True)
        return prediction
    except BaseException as exc:
        diagnostic = getattr(exc, "diagnostic", None)
        if diagnostic:
            if (
                calls
                and "response" not in calls[-1]
                and "api_diagnostic" not in calls[-1]
            ):
                failure_call = calls[-1]
            else:
                failure_call = {"stage": stage}
                calls.append(failure_call)
            stage = diagnostic.get("failure_stage") or failure_call["stage"]
            failure_call.update(error=str(exc), api_diagnostic=diagnostic)
            record["error_diagnostic"] = diagnostic
        record.update(error=str(exc), error_stage=stage, error_type=type(exc).__name__)
        (out / "prediction.json").unlink(missing_ok=True)
        raise
    finally:
        record["stage_timings_s"][stage] = perf_counter() - stage_started
        record.update(
            pipeline_latency_s=round(perf_counter() - started, 4),
            recovery_attempts=sample_recovery.get()["attempts"],
        )
        try:
            save_progress(complete=True)
            # Include result/failure persistence, before the metrics-only rewrite.
            save_progress(complete=True)
        finally:
            sample_recovery.reset(token)
