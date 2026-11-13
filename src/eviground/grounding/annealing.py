"""Continuous round-progress guidance for Monitor and ordinary ReAct policies."""

SCHEDULE = {"version": "continuous_linear_v2",
            "weights": {"exploration": "1-progress", "verification": "progress"},
            "mechanism": "continuous-weight policy prompts; hard stop at resource deadline",
            "sampling_temperature_changed": False}

GUIDANCE = (
    "Exploration priority={exploration_weight:.6f}; final-verification priority={verification_weight:.6f}. "
    "Use these as relative action priorities, not action probabilities or confidence scores. "
    "Exploration compares plausible target/reference interpretations and acquires evidence that "
    "distinguishes them. Final verification checks the best-supported target against its strongest "
    "rivals and unresolved query conditions. As verification priority increases, concentrate further "
    "acquisition on critical gaps that could change the choice; avoid unrelated expansion and "
    "repeated views. Choose the next action using these priorities and the evidence already available."
)
DEADLINE_GUIDANCE = (
    "No further evidence actions are allowed. Verify the retained evidence and finish. "
    "Return null if it cannot support a valid target; do not fabricate evidence or exclusions."
)


def policy_state(step, decision_limit, remaining_tools, last_decision=False):
    progress = step / max(decision_limit - 1, 1)
    exploration_weight, verification_weight = 1 - progress, progress
    deadline = remaining_tools <= 0 or last_decision
    guidance = (DEADLINE_GUIDANCE if deadline else
                GUIDANCE.format(exploration_weight=exploration_weight,
                                verification_weight=verification_weight))
    return {"progress": progress, "stage": "deadline" if deadline else "continuous",
            "exploration_weight": exploration_weight, "verification_weight": verification_weight,
            "guidance": guidance,
            "correction_policy": "Critical counterevidence always permits rebinding and reconsidering "
                                 "previously excluded candidates. Never lower the evidence standard. "
                                 "Finish early when supported; no minimum number of tool calls is required."}
