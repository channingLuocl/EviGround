"""Shared round-progress guidance for Monitor and ordinary ReAct policies."""

SCHEDULE = {"version": "round_progress_v1", "focus_start": 1 / 3, "verify_start": 2 / 3,
            "mechanism": "stage-conditioned policy prompts; hard stop at resource deadline",
            "sampling_temperature_changed": False}

GUIDANCE = {
    "explore": "Compare plausible interpretations and target/reference candidates. Acquire evidence "
               "for conditions that can distinguish them. Do not stop on an unsupported first guess.",
    "focus": "Focus on the remaining plausible rivals and unresolved query conditions. Request only "
             "evidence that can change their comparison; avoid unrelated expansion and repeated views.",
    "verify": "Prioritize final verification of the best-supported target against its strongest rivals. "
              "Request further evidence only for a critical unresolved condition that could change "
              "the choice. Otherwise finish when supported, or abstain if essential evidence is missing.",
    "deadline": "No further evidence actions are allowed. Verify the retained evidence and finish. "
                "Return null if it cannot support a valid target; do not fabricate evidence or exclusions.",
}


def policy_state(step, decision_limit, remaining_tools, last_decision=False):
    progress = step / max(decision_limit - 1, 1)
    stage = ("deadline" if remaining_tools <= 0 or last_decision else
             "explore" if progress < SCHEDULE["focus_start"] else
             "focus" if progress < SCHEDULE["verify_start"] else "verify")
    return {"progress": progress, "stage": stage, "guidance": GUIDANCE[stage],
            "correction_policy": "Critical counterevidence always permits rebinding and reconsidering "
                                 "previously excluded candidates. Never lower the evidence standard. "
                                 "Finish early when supported; no minimum number of tool calls is required."}
