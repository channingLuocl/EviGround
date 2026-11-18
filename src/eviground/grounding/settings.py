"""Resource bounds for the current EviGround method."""

import os


def action_no_repeat_enabled():
    raw = os.getenv("EVIGROUND_ACTION_NO_REPEAT", "0").strip()
    if raw not in ("0", "1"):
        raise ValueError("EVIGROUND_ACTION_NO_REPEAT must be 0 or 1")
    return raw == "1"


def policy_annealing_enabled():
    raw = os.getenv("EVIGROUND_POLICY_ANNEALING_ENABLED", "0").strip()
    if raw not in ("0", "1"):
        raise ValueError("EVIGROUND_POLICY_ANNEALING_ENABLED must be 0 or 1")
    return raw == "1"


def dialogue_history_enabled():
    raw = os.getenv("EVIGROUND_DIALOGUE_HISTORY", "0").strip()
    if raw not in ("0", "1"):
        raise ValueError("EVIGROUND_DIALOGUE_HISTORY must be 0 or 1")
    return raw == "1"


def dag_wire_format():
    raw = os.getenv("EVIGROUND_DAG_WIRE_FORMAT", "snapshot").strip()
    if raw not in ("snapshot", "delta"):
        raise ValueError("EVIGROUND_DAG_WIRE_FORMAT must be snapshot or delta")
    return raw


def successful_tool_budget_enabled():
    raw = os.getenv("EVIGROUND_SUCCESSFUL_TOOL_BUDGET", "0").strip()
    if raw not in ("0", "1"):
        raise ValueError("EVIGROUND_SUCCESSFUL_TOOL_BUDGET must be 0 or 1")
    return raw == "1" or dag_monitor_enabled()


def decision_max_tokens():
    value = int(os.getenv("EVIGROUND_DECISION_MAX_TOKENS", "4000" if dag_monitor_enabled() else "1600"))
    if value < 1:
        raise ValueError("EVIGROUND_DECISION_MAX_TOKENS must be positive")
    return value


def dag_monitor_enabled():
    raw = os.getenv("EVIGROUND_DAG_MONITOR_ENABLED", "0").strip()
    if raw not in ("0", "1"):
        raise ValueError("EVIGROUND_DAG_MONITOR_ENABLED must be 0 or 1")
    return raw == "1"


def dag_repair_limit():
    raw = os.getenv("EVIGROUND_DAG_REPAIR_LIMIT", "6").strip()
    try:
        value = int(raw)
    except ValueError:
        raise ValueError("EVIGROUND_DAG_REPAIR_LIMIT must be a nonnegative integer") from None
    if value < 0:
        raise ValueError("EVIGROUND_DAG_REPAIR_LIMIT must be a nonnegative integer")
    return value


def evidence_policy():
    """Explicit tool availability for controlled ablations; defaults are unchanged."""
    enabled = {}
    for name, variable in (
        ("inspect", "EVIGROUND_INSPECT_ENABLED"),
        ("measure", "EVIGROUND_MEASURE_ENABLED"),
        ("category_inspection_required", "EVIGROUND_CATEGORY_INSPECTION_REQUIRED"),
    ):
        raw = os.getenv(variable, "1").strip()
        if raw not in ("0", "1"):
            raise ValueError(f"{variable} must be 0 or 1")
        enabled[name] = raw == "1"
    enabled["category_inspection_required"] &= enabled["inspect"]
    return enabled


def settings():
    policy = {
        "version": "eviground-0.1.0",
        "extra_api_attempts_limit": 1,
        "context_text_upper_bound": int(os.getenv("EVIGROUND_CONTEXT_LIMIT", "131072")),
        "evidence_tools": evidence_policy(),
        "dag_monitor_enabled": dag_monitor_enabled(),
        "policy_annealing_enabled": policy_annealing_enabled(),
        "dialogue_history_enabled": dialogue_history_enabled(),
        "dag_wire_format": dag_wire_format(),
        "successful_tool_budget": successful_tool_budget_enabled(),
        "decision_max_tokens": decision_max_tokens(),
        "action_no_repeat": action_no_repeat_enabled(),
    }
    if policy["successful_tool_budget"]:
        policy["dag_monitor_repair_limit"] = dag_repair_limit()
    if policy["policy_annealing_enabled"]:
        from eviground.grounding.annealing import SCHEDULE

        policy["policy_annealing_schedule"] = SCHEDULE
    return policy


def evidence_inventory(query, parsed):
    """Retain parser coverage gaps and roles, without an execution schedule."""
    from eviground.query.validation import size_conditions

    rows = []
    for condition in parsed.get("condition_records", []):
        kind = condition.get("condition_type", "")
        dimension = (
            "size"
            if kind == "size"
            else (
                "distance"
                if kind in ("relation:near", "relation:nearest", "relation:farthest")
                else (
                    "position"
                    if kind.startswith("relation:")
                    else (
                        "viewpoint"
                        if "observer" in kind or "viewpoint" in kind
                        else (
                            "appearance"
                            if "attribute" in kind or "category" in kind
                            else "unresolved"
                        )
                    )
                )
            )
        )
        rows.append(
            {
                "source_quote": condition.get("source_quote"),
                "dimension": dimension,
                "target_role": condition.get("target_role"),
                "reference_role": condition.get("reference_role"),
                "condition": kind,
                "binding_status": condition.get("binding_status"),
                "dependencies": (
                    ["bind reference"] if condition.get("reference_role") else []
                ),
            }
        )
    for category in parsed.get("Target", []):
        for spec in size_conditions(parsed, category):
            rows.append(
                {
                    "source_quote": spec.get("source_phrase", spec["original_quote"]),
                    "dimension": "size",
                    "target_role": {"category": category},
                    "condition": spec["word"],
                    "metric": spec["metric"],
                    "scope_kind": spec["scope_kind"],
                    "scope_phrase": spec.get("scope_phrase"),
                    "member_count": spec.get("required_member_count"),
                    "attachment": spec.get("syntax_attachment"),
                    **(
                        {"scope_hypotheses": spec["scope_hypotheses"]}
                        if spec.get("scope_hypotheses")
                        else {}
                    ),
                    "dependencies": ["bind comparison members"]
                    + spec.get("scope_relation_quotes", []),
                }
            )
    for phrase in parsed.get("Unhandled", []):
        if phrase.casefold() in query.casefold():
            registered = [row for row in rows if row["condition"] == "unhandled"
                          and row["source_quote"] == phrase]
            if registered:
                for row in registered:
                    if "obtain quoted predicate evidence" not in row["dependencies"]:
                        row["dependencies"].append("obtain quoted predicate evidence")
                continue
            rows.append(
                {
                    "source_quote": phrase,
                    "dimension": "unresolved",
                    "condition": "unhandled",
                    "dependencies": ["obtain quoted predicate evidence"],
                }
            )
    return {
        "items": rows,
        "role_coreferences": parsed.get("target_coreferences", []),
        "policy": "requirements and dependencies only; planner chooses order",
    }
