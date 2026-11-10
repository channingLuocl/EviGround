"""Model-revised task DAG with evidence-backed state and reversible exclusions."""

from copy import deepcopy

from eviground.api.response_schemas import TEXT, REASON, array, enum, obj
from eviground.grounding.settings import evidence_inventory


INSTRUCTIONS = """
You also act as the planning monitor. Return a complete, compact task DAG in
monitor on EVERY turn, not just a next-action explanation. Plan the whole route
from the original utterance to verification, and reconsider earlier assumptions
after each observation. Keep stable node IDs when their meaning is unchanged.
Copy unchanged completed nodes verbatim instead of paraphrasing their conclusions.

Nodes are questions to resolve, not physical objects. Use 2-12 nodes, including
one final verification node that depends transitively on every other node.
Record the exact supporting source_quotes, condition_ids, dependencies, allowed
actions, bound object IDs, status, evidence IDs and conclusion for each node.
Cover every monitor_requirements item. Keep all original clauses, negation,
comparison membership and observer assumptions. Nested references need binding
before dependent relations; middle/ordinal/size comparisons need the relevant
comparison group and an appropriate arrangement, frame or metric first. Do not
force a reference chain when the utterance instead needs group comparison.

The current_node_id is the stage being worked on. For a tool call it must be
pending and all dependencies completed; the action must be allowed by that node.
A completed node needs an explicit conclusion grounded in existing evidence IDs.
bound_object_ids records the node's current object bindings, not every object it
may need to examine. Tool focus_ids may also include unbound alternatives and
reference/comparison objects needed to resolve the current task. Examining an
object does not automatically bind it or complete the task.
Completed reasoning-only nodes may have allowed_actions=[]; pending nodes need
an available action. This also permits initial-evidence-only tool ablations.
initial denotes the supplied initial views/basic geometry, not future evidence.
Tool errors and monitor feedback are not supporting observations. Viewing an
object does not itself prove a predicate; assess actual visibility and content.
Measurements are conditional on the chosen reference/group/frame, not proof of
their interpretation. Unknown or unobserved does not mean false.

Use excluded_target_ids ONLY for targets contradicted by this node's evidence;
state the exclusion basis in its conclusion. The cited observation must include
the excluded object; initial includes all supplied views/basic object geometry.
Excluded rivals need not remain in bound_object_ids, which may hold the winner.
Do not exclude an unexamined rival
merely to obtain uniqueness. Exclusion affects target eligibility, never removal
of a physical member from a comparison group. The original target pool is fixed.

If a prior binding or conclusion fails, put its old node ID in reopen_node_ids,
make that stage pending and move the current stage there. Rebuild the affected
downstream plan. Its old descendants are automatically reset to pending, with
their conclusions/evidence/exclusions cleared, even if you mark them completed.
New nodes introduced during a rollback also start pending; renaming a stage
does not preserve an old completion.
Unchanged completed branches and ALL raw observations remain available. A
changed completed conclusion/binding also triggers this reset. Reactivated
targets must be reconsidered. Never add a backwards dependency: revise the next
version of the DAG instead. Explain the revision in revision_reason.
reopen_node_ids applies only to this revision; normally return an empty array.

Finish with a target ONLY at the completed final verification node, after all
nodes are completed, every requirement is covered and the program-derived active
target pool contains exactly that ID. Explain why other targets are contradicted,
and leave unresolved empty. Uniqueness alone is not sufficient. Existing category
inspection requirements still apply. If the decision budget ends without a valid
unique supported target, finish with predicted_id=null and state what is missing;
do not force exclusions. A null finish is abstention, not successful completion.
Invalid plans receive correction feedback and use a separate bounded repair
allowance, not an evidence-tool call. Correct the reported node/requirement IDs
before repeating a rejected plan. remaining_tool_calls counts successful tools;
remaining_plan_repairs bounds further correction attempts. Once no tools remain,
return action=finish, including during a plan repair. Keep node conclusions and
revision_reason concise; prefer small task graphs.
"""


def monitor_schema(known, allowed, available, requirements):
    node = obj({
        "id": REASON,
        "goal": REASON,
        "kind": enum("binding", "attribute", "relation", "comparison", "verification"),
        "source_quotes": array(REASON),
        "condition_ids": array(enum(*requirements)),
        "depends_on": array(REASON),
        "allowed_actions": array(enum(*sorted(available))),
        "status": enum("pending", "completed"),
        "bound_object_ids": array({"type": "integer", "enum": sorted(known)}),
        "evidence_ids": array(REASON),
        "conclusion": TEXT,
        "excluded_target_ids": array({"type": "integer", "enum": sorted(allowed)}),
    })
    return obj({"nodes": array(node), "current_node_id": REASON,
                "reopen_node_ids": array(REASON), "revision_reason": REASON})


class DagMonitor:
    """Validate snapshots; invalidate descendants rather than erasing observations."""

    def __init__(self, query, parsed, known, allowed, available):
        self.query, self.known = query, set(known)
        self.allowed, self.available = set(allowed), set(available)
        self.requirements = [{"id": "q0", "source_quote": query,
                              "condition": "verify complete original description"}]
        for row in evidence_inventory(query, parsed)["items"]:
            self.requirements.append({**row, "id": f"q{len(self.requirements)}"})
        self.plan, self.history = None, []

    def schema(self):
        return monitor_schema(self.known, self.allowed, self.available,
                              [r["id"] for r in self.requirements])

    def context(self, observations):
        return {"monitor_requirements": self.requirements,
                "task_dag": self.plan,
                "active_target_ids": self.active_ids(),
                "supporting_evidence_ids": sorted(self.evidence_ids(observations))}

    def active_ids(self, plan=None):
        excluded = {i for n in (plan or self.plan or {}).get("nodes", [])
                    if n["status"] == "completed" for i in n["excluded_target_ids"]}
        return sorted(self.allowed - excluded)

    @staticmethod
    def evidence_ids(observations):
        return {"initial"} | {o["evidence_id"] for o in observations
                              if o.get("evidence_id") and not o.get("no_evidence_acquired")
                              and o.get("tool") not in ("finish_check", "monitor_check")
                              and not o.get("error")}

    def evidence_objects(self, observations):
        """Object scope actually supplied by each usable observation."""
        scopes = {"initial": self.known}
        valid = self.evidence_ids(observations)
        for observation in observations:
            eid = observation.get("evidence_id")
            if eid in valid:
                scopes[eid] = set(observation.get("focus_ids", [])) | {
                    row["id"] for row in observation.get("objects", [])
                }
        return scopes

    @staticmethod
    def descendants(nodes, roots):
        affected = set(roots)
        while True:
            expanded = affected | {n["id"] for n in nodes
                                   if affected.intersection(n["depends_on"])}
            if expanded == affected:
                return affected
            affected = expanded

    def update(self, proposal, action, observations):
        """Accept only an executable plan/action pair, atomically."""
        from jsonschema import ValidationError, validate

        try:
            validate(proposal, self.schema())
        except ValidationError as exc:
            raise ValueError(f"Invalid monitor DAG: {exc.message}") from exc
        plan = deepcopy(proposal)
        nodes = plan["nodes"]
        if not 2 <= len(nodes) <= 12:
            raise ValueError("Monitor needs 2-12 task nodes including final verification")
        by_id = {n["id"]: n for n in nodes}
        if len(by_id) != len(nodes):
            raise ValueError("Monitor node IDs must be unique")
        requirements = {r["id"] for r in self.requirements}
        evidence = self.evidence_ids(observations)
        evidence_objects = self.evidence_objects(observations)
        for n in nodes:
            if (not n["source_quotes"] or any(q.casefold() not in self.query.casefold()
                                              for q in n["source_quotes"])):
                raise ValueError("Every task needs exact quotes from the description")
            for field in ("depends_on", "condition_ids", "allowed_actions", "bound_object_ids",
                          "evidence_ids", "excluded_target_ids"):
                if len(set(n[field])) != len(n[field]):
                    raise ValueError(f"Duplicate monitor node entries: {field}")
            if not n["allowed_actions"] and n["status"] != "completed":
                raise ValueError("Pending tasks need at least one available action")
            if any(i not in by_id for i in n["depends_on"]):
                raise ValueError("Monitor dependency references an absent node")
            if not set(n["evidence_ids"]) <= evidence:
                raise ValueError(f"Task {n['id']} cites absent, future or failed evidence; "
                                 f"use existing IDs {sorted(evidence)}")
            if n["status"] == "completed" and (not n["evidence_ids"] or not n["conclusion"].strip()):
                raise ValueError("Completed tasks need existing evidence and a conclusion")
            if n["status"] != "completed" and n["excluded_target_ids"]:
                raise ValueError("Pending tasks cannot exclude targets")
            observed = set().union(*(evidence_objects[eid] for eid in n["evidence_ids"]))
            unsupported = set(n["excluded_target_ids"]) - observed
            if unsupported:
                raise ValueError(f"Task {n['id']} excludes {sorted(unsupported)} outside its cited evidence; "
                                 "inspect/measure those objects or cite supporting initial evidence")
        # Kahn traversal checks acyclicity without depending on model array order.
        done = set()
        while len(done) < len(nodes):
            ready = {n["id"] for n in nodes if n["id"] not in done
                     and set(n["depends_on"]) <= done}
            if not ready:
                raise ValueError("Monitor dependencies contain a cycle")
            done.update(ready)
        finals = [n for n in nodes if n["kind"] == "verification"]
        if len(finals) != 1 or finals[0]["allowed_actions"] != ["finish"]:
            raise ValueError("Monitor needs one final verification task allowing only finish")
        final = finals[0]
        if any("finish" in n["allowed_actions"] for n in nodes if n is not final):
            raise ValueError("Only final verification may finish")
        ancestors = {final["id"]}
        while True:
            expanded = ancestors | {d for i in ancestors for d in by_id[i]["depends_on"]}
            if expanded == ancestors:
                break
            ancestors = expanded
        if ancestors != set(by_id):
            raise ValueError("Final verification must depend on every planned task")
        covered = {c for n in nodes for c in n["condition_ids"]}
        if covered != requirements:
            raise ValueError("Monitor DAG must cover all language requirements; "
                             f"assign missing IDs {sorted(requirements - covered)} to the tasks that resolve them")

        old = {n["id"]: n for n in (self.plan or {}).get("nodes", [])}
        reopened = set(plan["reopen_node_ids"])
        if not reopened <= set(old):
            raise ValueError("Can only reopen nodes from the previous DAG")
        semantic = ("goal", "kind", "source_quotes", "condition_ids", "depends_on",
                    "bound_object_ids", "conclusion", "excluded_target_ids")
        def changed(before, after):
            return any((set(before[k]) != set(after[k]) if isinstance(before[k], list)
                        else before[k] != after[k]) for k in semantic)

        roots = reopened | {i for i, n in old.items() if i not in by_id or
                            (n["status"] == "completed" and
                             (by_id[i]["status"] != "completed" or
                              changed(n, by_id[i])))}
        invalidated = self.descendants(list(old.values()), roots) if roots else set()
        if roots:
            invalidated.update(set(by_id) - set(old))
        # Also clear newly attached descendants of a reopened or replaced stage.
        invalidated = self.descendants(nodes, invalidated) if invalidated else set()
        for i in invalidated & set(by_id):
            by_id[i].update(status="pending", evidence_ids=[], conclusion="",
                            excluded_target_ids=[])
        for n in nodes:
            if n["status"] == "completed" and any(by_id[d]["status"] != "completed"
                                                   for d in n["depends_on"]):
                raise ValueError("A completed task has an incomplete prerequisite")
        current = by_id.get(plan["current_node_id"])
        if current is None:
            raise ValueError("Current monitor stage is absent")
        if action["action"] not in current["allowed_actions"]:
            raise ValueError("Action is not allowed by the current monitor stage")
        if action["action"] != "finish":
            if current["status"] != "pending" or any(by_id[d]["status"] != "completed"
                                                      for d in current["depends_on"]):
                raise ValueError("Current stage is not ready; resolve its prerequisites first")
        elif action.get("predicted_id") is not None:
            if (current is not final or any(n["status"] != "completed" for n in nodes)
                    or self.active_ids(plan) != [action["predicted_id"]]
                    or action["predicted_id"] not in current["bound_object_ids"]
                    or action.get("unresolved")):
                raise ValueError("Finish needs completed final verification, one active target and no unresolved conditions")
        before, after = set(self.active_ids()), set(self.active_ids(plan))
        event = {"revision": len(self.history) + 1, "current_node_id": current["id"],
                 "invalidated_node_ids": sorted(invalidated),
                 "reactivated_target_ids": sorted(after - before),
                 "active_target_ids": sorted(after), "plan": deepcopy(plan)}
        self.plan = plan
        self.history.append(event)
        return event
