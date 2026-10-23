"""Resource bounds for the current EviGround method."""

import os


def settings():
    return {
        "version": "eviground-0.1.0",
        "extra_api_attempts_limit": 1,
        "context_text_upper_bound": int(os.getenv("EVIGROUND_CONTEXT_LIMIT", "131072")),
    }


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
