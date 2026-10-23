"""Single-pass evaluation. Answers and annotated boxes stay in the coordinator."""

import json
from eviground.evaluation.run_storage import ensure_run_writable, write_json


def evaluate(row, prediction, log, top_k):
    data = json.loads(prediction.read_text())
    if (
        data["query"] != row["query"]
        or data.get("category_top_k") != top_k
        or data.get("gt_categories_supplied_to_model") is not False
        or (data.get("query_target_id_supplied_to_model") is not False)
    ):
        raise ValueError(f"Prediction provenance mismatch: {prediction}")
    trace = data.get("active_trace", {})
    return {
        **row,
        "category_top_k": top_k,
        "predicted_id": data.get("predicted_id"),
        "correct": data.get("predicted_id") == row["target_id"],
        "decision_status": data.get("decision_status"),
        "api_calls": data.get("api_calls", 0),
        "api_attempts": data.get("api_attempts", 0),
        "call_usage": data.get("call_usage", []),
        "recovery_attempts": data.get("recovery_attempts", []),
        "pipeline_latency_s": data.get("pipeline_latency_s"),
        "implementation_version": data.get("implementation_version"),
        "prediction_source": data.get("prediction_source"),
        "stop_reason": trace.get("stop_reason"),
        "prediction_path": str(prediction),
        "log": str(log),
    }


def summarize(results, total=None):
    total = len(results) if total is None else total
    correct = sum((r.get("correct") is True for r in results))
    answered = sum((r.get("predicted_id") is not None for r in results))
    return {
        "selected_total": total,
        "processed": len(results),
        "pending": total - len(results),
        "complete": len(results) == total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "processed_accuracy": correct / len(results) if results else None,
        "answered": answered,
        "null_count": len(results) - answered,
        "errors": sum((bool(r.get("error")) for r in results)),
        "abstained": sum(
            (r.get("predicted_id") is None and (not r.get("error")) for r in results)
        ),
        "actual_api_attempts": sum((r.get("api_attempts", 0) for r in results)),
        "acc25": (
            sum((r.get("iou_3d", 0) >= 0.25 for r in results)) / total
            if total
            else None
        ),
        "acc50": (
            sum((r.get("iou_3d", 0) >= 0.5 for r in results)) / total if total else None
        ),
        "evaluation_policy": "single_pass",
    }


def report(output, results, total):
    ensure_run_writable(output)
    if len({r["csv_row"] for r in results}) != len(results) or len(results) > total:
        raise ValueError("Results must be unique and fit the fixed query selection")
    summary = summarize(results, total)
    groups = {
        key: {
            str(value): summarize([r for r in results if r.get(key) == value])
            for value in sorted({r.get(key) for r in results}, key=str)
        }
        for key in ("easy", "view_dep")
    }
    write_json(
        output / "results.json",
        {
            "summary": summary,
            "groups": groups,
            "results": sorted(results, key=lambda r: r["csv_row"]),
        },
    )
    return summary
