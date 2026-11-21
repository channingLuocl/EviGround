"""Model-revised task DAG with evidence-backed state and reversible exclusions."""

from copy import deepcopy
from itertools import permutations
import re

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
supporting_evidence_objects lists the objects covered by each usable evidence ID.
For exclusions based on initial geometry, explicitly cite initial; a later local
view or measurement does not cover the entire initial scene. Exclusions from a
pending node or outside cited evidence are discarded, keeping those targets active.
Use the returned task_dag and discarded_exclusions instead of repeating those
exclusions. An otherwise valid tool action may still proceed to gather evidence.
If multiple targets remain after a correction, resolve those rivals explicitly;
reopen the appropriate completed stage if further inspection is needed.

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
A valid rollback is retained even when its proposed action is rejected. Use the
returned task_dag and ready_node_ids to recheck the reset stages against retained
observations; do not repeat a stale finish or keep reopening the same stage.

Finish with a target ONLY at the completed final verification node, after all
nodes are completed, every requirement is covered and the program-derived active
target pool contains exactly that ID. Explain why other targets are contradicted,
and leave unresolved empty. Uniqueness alone is not sufficient. Existing category
inspection requirements still apply. When finish_check requires category inspection,
the controller reopens a nonterminal task and its descendants. Follow the returned
ready_node_ids and inspect required_inspection_ids with plausible primary matches;
keep the final verification node finish-only. A requested focus ID does not count
as visibly inspected unless it appears in the rendered marked_ids.
If the decision budget ends without a valid
unique supported target, finish with predicted_id=null and state what is missing;
do not force exclusions. A null finish is abstention, not successful completion.
Invalid plans receive correction feedback and use a separate bounded repair
allowance, not an evidence-tool call. Correct the reported node/requirement IDs
before repeating a rejected plan. remaining_tool_calls counts successful tools;
remaining_plan_repairs bounds further correction attempts. Once no tools remain,
return action=finish, including during a plan repair. Keep node conclusions and
revision_reason concise; prefer small task graphs.
"""

COMPACT_INSTRUCTIONS = """
Return monitor as a complete task DAG on every action. nodes contains 1-11
evidence tasks with IDs n0 through n10. verification is a separate REQUIRED
final task with ID n11; never put it in nodes. Together they form 2-12 nodes.
requirement_nodes maps EVERY language requirement
q0, q1, ... to the node IDs that resolve it. Fill every required key, including
q0; do not omit the detailed conditions. depends_on and
current_node_id contain NODE IDs, never q IDs or prerequisite descriptions.
prerequisite_descriptions describes needed reasoning, not node references.
q0 is the complete-description verification requirement: assign it to n11.
Use evidence tasks to resolve the individual clauses and bindings before that
final check; do not make full-description verification a prerequisite binding.

Plan evidence tasks (binding/attribute/relation/comparison) followed by exactly
one verification task. Only verification allows ["finish"]. It depends
transitively on every other node. Evidence tasks allow inspect and/or measure;
completed reasoning-only tasks may allow []. The current node for a tool action
is pending, allows that tool, and has completed dependencies. For example, an
evidence node n0 can have depends_on=[] and verification n11 depends_on=["n0"].
Choose the actual tasks and their order from the description and observations.

source_quotes are exact substrings of description. bound_object_ids records
actual bindings, not every alternative inspected. Completed nodes need existing
supporting_evidence_ids and an observed conclusion; initial means supplied views
and basic geometry. Never cite future tools, feedback, or errors as evidence.
Pending nodes have empty evidence_ids, conclusion, and excluded_target_ids.
ID and action lists are sets: never repeat an entry.
Keep unresolved conditions pending. Do not copy the query as if it were observed.

Exclude a target only in a completed node whose cited evidence covers that ID
and contradicts it; explain the contradiction. Unknown is not false. Do not
exclude rivals to manufacture uniqueness. Check actual references, comparison
members, viewpoint assumptions, negation, and every original clause.

Use task_dag and ready_node_ids after feedback. Copy unchanged completed nodes
verbatim. If evidence overturns a binding/conclusion, reopen its old node ID in
reopen_node_ids and revise descendants: they reset to pending and excluded
targets reactivate. Do not repeat a rejected graph unchanged. A required category
inspection reopens an evidence node; keep final verification finish-only.

For a non-null finish, current_node_id is the completed verification node, all
nodes are completed, all requirements covered, unresolved=[], and evidence-backed
exclusions leave exactly predicted_id active. Explain why its strongest rivals
fail. With missing essential evidence, finish with predicted_id=null instead.
Do not force exclusions at the deadline. Keep the graph and conclusions concise.

Wire-shape example ONLY, not a plan or evidence for the current scene. This shows
one unresolved task; choose the actual number, goals and dependencies from the
query. Do not fill all 11 slots. Unused nodes do not appear. In particular, a
first task has no dependencies; never make tasks mutually depend on each other.
{"nodes":[{"id":"n0","goal":"resolve the described conditions","kind":"binding",
"source_quotes":["<exact description substring>"],"depends_on":[],
"allowed_actions":["inspect","measure"],"status":"pending","bound_object_ids":[],
"evidence_ids":[],"conclusion":"","excluded_target_ids":[]}],
"current_node_id":"n0","reopen_node_ids":[],"revision_reason":"start evidence acquisition",
"verification":{"id":"n11","goal":"verify the supported unique target","kind":"verification",
"source_quotes":["<exact description substring>"],"depends_on":["n0"],
"allowed_actions":["finish"],"status":"pending","bound_object_ids":[],
"evidence_ids":[],"conclusion":"","excluded_target_ids":[]},"requirement_nodes":{"q0":["n11"]}}
Replace quoted placeholders with actual source text and assign ALL actual q IDs,
not just q0. Later turns must update this pending shape using returned evidence:
complete a task only with its observed conclusion and existing evidence IDs;
move to another ready task or completed verification instead of repeating it.
"""

DELTA_INSTRUCTIONS = """
For this turn ONLY, the following serialization replaces the complete-DAG output
shape above; all planning, evidence and completion semantics remain unchanged.
The current task_dag is the complete authoritative plan. On this turn, return
monitor as node_updates, current_node_id, reopen_node_ids and revision_reason.
Follow the field order enforced by the decoder. The entire proposal is still
validated before any tool or finish executes.
Besides the required checkpoint described below, node_updates contains only
nodes whose state changes; omit other unchanged nodes.
For each changed node give its id, status, bound_object_ids, evidence_ids,
conclusion and excluded_target_ids. Keep its goal, dependencies, source quotes
and requirement assignments: the controller retains those exact definitions.
Read the acquired observations and update supported tasks before choosing the
next action. A completed comparison and final verification may be recorded in
the same response if all clauses are supported. A non-null finish still needs
every task completed and current_node_id=n11; pending tasks cannot finish.
For a non-null finish, node_updates must explicitly start with the completed
n11 verification checkpoint, followed by any other changed completed tasks.
Supply its actual binding, existing evidence IDs and observed conclusion even
if its state is unchanged. When finishing with a delta, also include a completed
update for EVERY task that is still pending in task_dag; omitted tasks stay pending.
For a tool action, node_updates starts with its current
evidence node as pending, even if unchanged. Only later updates may complete
other evidence tasks; verification remains pending while a tool is requested.
Never request a tool from a completed node. A tool must not use n11.
To add/remove tasks, change dependencies, goals or requirement assignments,
return monitor={"replace_plan": <complete revised compact DAG>} instead.
This replaces the whole proposal, not just the changed nodes. For rollback,
use reopen_node_ids as before; downstream states and exclusions reset under
the same rules. Never mark a task completed just because its tool was called.
Existing evidence, coverage, unique-target and final-verification checks apply.
"""

def compact_delta_instructions():
    """Retain planning rules without also asking for the obsolete snapshot shape."""
    rules = COMPACT_INSTRUCTIONS.split("\nWire-shape example ONLY,", 1)[0]
    rules = rules.replace("Return monitor as a complete task DAG on every action. ",
                          "The supplied task_dag is the complete accepted plan. ", 1)
    wire = DELTA_INSTRUCTIONS.replace(
        "For this turn ONLY, the following serialization replaces the complete-DAG output\n"
        "shape above; all planning, evidence and completion semantics remain unchanged.",
        "Use the following delta serialization for this turn; all planning, evidence\n"
        "and completion semantics remain unchanged.", 1)
    return rules + wire


NODE_STATE_FIELDS = ("id", "status", "bound_object_ids", "evidence_ids",
                     "conclusion", "excluded_target_ids")


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


def exact_source_quotes(query, requirements):
    """The native compact-DAG quote domain, with literal punctuation preserved."""
    return sorted({row['source_quote'] for row in requirements
                   if row.get('source_quote') and row['source_quote'].casefold() in query.casefold()})


def usable_evidence_observation(row):
    return bool(row.get('evidence_id')) and not row.get('no_evidence_acquired') and \
        row.get('tool') not in ('finish_check', 'monitor_check') and not row.get('error')


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

    def schema(self, compact=False, evidence_ids=None):
        schema = monitor_schema(self.known, self.allowed, self.available,
                                [r["id"] for r in self.requirements])
        if compact:
            node_ids = enum(*(f"n{i}" for i in range(12)))
            task_ids = enum(*(f"n{i}" for i in range(11)))
            nodes = schema["properties"]["nodes"]
            nodes.update(minItems=1, maxItems=11)
            node = nodes["items"]
            del node["properties"]["condition_ids"]
            node["required"].remove("condition_ids")
            node["properties"]["id"] = task_ids
            node["properties"]["depends_on"] = {**array(task_ids), "maxItems": 11}
            node["properties"]["bound_object_ids"]["maxItems"] = len(self.known)
            node["properties"]["excluded_target_ids"]["maxItems"] = len(self.allowed)
            quotes = exact_source_quotes(self.query, self.requirements)
            node["properties"]["source_quotes"] = {
                **array(enum(*quotes)), "minItems": 1, "maxItems": len(quotes)}
            evidence_task, verification = deepcopy(node), deepcopy(node)
            evidence_task["properties"]["kind"] = enum("binding", "attribute", "relation", "comparison")
            tools = sorted(self.available - {"finish"})
            evidence_task["properties"]["allowed_actions"] = {
                "type": "array", "enum": [[]] + [
                    list(group) for size in range(1, len(tools) + 1)
                    for group in permutations(tools, size)
                ]}
            verification["properties"]["kind"] = enum("verification")
            verification["properties"]["id"] = enum("n11")
            verification["properties"]["depends_on"]["minItems"] = 1
            verification["properties"]["allowed_actions"] = {**array(enum("finish")), "minItems": 1, "maxItems": 1}
            def status_branches(task):
                pending, completed = deepcopy(task), deepcopy(task)
                pending["properties"]["status"] = enum("pending")
                if "enum" in pending["properties"]["allowed_actions"] and tools:
                    pending["properties"]["allowed_actions"]["enum"].remove([])
                for name in ("evidence_ids", "excluded_target_ids"):
                    pending["properties"][name]["maxItems"] = 0
                pending["properties"]["conclusion"] = enum("")
                completed["properties"]["status"] = enum("completed")
                completed["properties"]["conclusion"] = REASON
                completed["properties"]["evidence_ids"]["minItems"] = 1
                if evidence_ids is not None:
                    completed["properties"]["evidence_ids"].update(
                        items=enum(*sorted(evidence_ids)), maxItems=len(evidence_ids))
                return {"anyOf": [pending, completed]}
            nodes["items"] = status_branches(evidence_task)
            schema["properties"]["verification"] = status_branches(verification)
            schema["required"].append("verification")
            schema["properties"]["current_node_id"] = node_ids
            schema["properties"]["reopen_node_ids"] = {**array(node_ids), "maxItems": 12}
            schema["properties"]["requirement_nodes"] = obj({
                r["id"]: {**array(node_ids), "minItems": 1, "maxItems": 12}
                for r in self.requirements
            })
            schema["required"].append("requirement_nodes")
        return schema

    def delta_schema(self, evidence_ids=None):
        """Short state edits plus an explicit full-plan replacement escape hatch."""
        if self.plan is None:
            raise ValueError("A delta needs an accepted task DAG")
        full = self.schema(compact=True, evidence_ids=evidence_ids)
        node_ids = [node["id"] for node in self.plan["nodes"]]
        branches = deepcopy(full["properties"]["nodes"]["items"]["anyOf"])
        for branch in branches:
            branch["properties"] = {key: branch["properties"][key] for key in NODE_STATE_FIELDS}
            branch["properties"]["id"] = enum(*node_ids)
            branch["required"] = list(NODE_STATE_FIELDS)
        delta = obj({
            "node_updates": {**array({"anyOf": branches}), "maxItems": len(node_ids)},
            "current_node_id": enum(*node_ids),
            "reopen_node_ids": {**array(enum(*node_ids)), "maxItems": len(node_ids)},
            "revision_reason": REASON,
        })
        return {"anyOf": [delta, obj({"replace_plan": full})]}

    def decode(self, proposal):
        """Expand compact requirement assignments into the existing DAG contract."""
        if "node_updates" in proposal or "replace_plan" in proposal:
            from jsonschema import validate

            validate(proposal, self.delta_schema())
            if "replace_plan" in proposal:
                return self.decode(proposal["replace_plan"])
            updates = proposal["node_updates"]
            if len({node["id"] for node in updates}) != len(updates):
                raise ValueError("Duplicate node IDs in monitor delta")
            plan = deepcopy(self.plan)
            by_id = {node["id"]: node for node in plan["nodes"]}
            for node in updates:
                by_id[node["id"]].update(deepcopy(node))
            for key in ("current_node_id", "reopen_node_ids", "revision_reason"):
                plan[key] = deepcopy(proposal[key])
            return plan
        if "requirement_nodes" not in proposal:
            return proposal
        from jsonschema import validate

        validate(proposal, self.schema(compact=True))
        plan = deepcopy(proposal)
        assignments = plan.pop("requirement_nodes")
        plan["nodes"].append(plan.pop("verification"))
        known_nodes = {node["id"] for node in plan["nodes"]}
        if any(node_id not in known_nodes for group in assignments.values() for node_id in group):
            raise ValueError("Requirement assignment references an absent DAG node")
        for node in plan["nodes"]:
            node["condition_ids"] = [key for key, group in assignments.items() if node["id"] in group]
        return plan

    def encode(self, plan):
        """Keep prior plans in the same wire format requested on the next turn."""
        if plan is None or any(key in plan for key in ("requirement_nodes", "node_updates", "replace_plan")):
            return deepcopy(plan)
        wire = deepcopy(plan)
        verification = [n for n in wire["nodes"] if n["kind"] == "verification"]
        if len(verification) != 1:
            raise ValueError("Internal DAG needs exactly one verification node")
        wire["requirement_nodes"] = {
            r["id"]: [n["id"] for n in wire["nodes"] if r["id"] in n["condition_ids"]]
            for r in self.requirements
        }
        for node in wire["nodes"]:
            del node["condition_ids"]
        wire["verification"] = verification[0]
        wire["nodes"] = [n for n in wire["nodes"] if n["kind"] != "verification"]
        return wire

    def context(self, observations):
        return {"monitor_requirements": self.requirements,
                "task_dag": self.plan,
                "ready_node_ids": self.ready_ids(),
                "active_target_ids": self.active_ids(),
                "supporting_evidence_objects": {eid: sorted(ids) for eid, ids
                                                in self.evidence_objects(observations).items()},
                "discarded_exclusions": self.history[-1].get("discarded_exclusions", []) if self.history else [],
                "supporting_evidence_ids": sorted(self.evidence_ids(observations))}

    def ready_ids(self, plan=None):
        nodes = (plan or self.plan or {}).get("nodes", [])
        completed = {n["id"] for n in nodes if n["status"] == "completed"}
        return [n["id"] for n in nodes if n["status"] == "pending"
                and set(n["depends_on"]) <= completed]

    def active_ids(self, plan=None):
        excluded = {i for n in (plan or self.plan or {}).get("nodes", [])
                    if n["status"] == "completed" for i in n["excluded_target_ids"]}
        return sorted(self.allowed - excluded)

    @staticmethod
    def evidence_ids(observations):
        return {"initial"} | {o["evidence_id"] for o in observations if usable_evidence_observation(o)}

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

    def require_inspection(self, target_id, observations):
        """Route an unmet category guard back to an existing evidence stage."""
        plan = deepcopy(self.plan)
        tasks = [n for n in plan["nodes"] if n["kind"] != "verification"]
        relevant = [n for n in tasks if target_id in n["bound_object_ids"]]
        stage = min(relevant or tasks,
                    key=lambda n: len(self.descendants(plan["nodes"], {n["id"]})))
        stage.update(status="pending", evidence_ids=[], conclusion="", excluded_target_ids=[])
        stage["allowed_actions"] = sorted(set(stage["allowed_actions"]) | {"inspect"})
        affected = self.descendants(plan["nodes"], {stage["id"]})
        for n in plan["nodes"]:
            if n["id"] in affected and not n["allowed_actions"]:
                n["allowed_actions"] = ["inspect"]
        plan.update(current_node_id=stage["id"], reopen_node_ids=[],
                    revision_reason=f"Category inspection of target {target_id} required at stage {stage['id']}")
        event = self.update(plan, {"action": "inspect"}, observations)
        event.update(transition="category_inspection_required", required_inspection_ids=[target_id])
        return event

    def update(self, proposal, action, observations):
        """Reject malformed graphs; retain safe rollbacks and conservative corrections."""
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
        discarded_exclusions = []
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
            observed = set().union(*(evidence_objects[eid] for eid in n["evidence_ids"]))
            unsupported = set(n["excluded_target_ids"])
            if n["status"] == "completed":
                unsupported -= observed
            if unsupported:
                n["excluded_target_ids"] = [i for i in n["excluded_target_ids"] if i not in unsupported]
                discarded_exclusions.append({"node_id": n["id"], "target_ids": sorted(unsupported),
                                             "reason": "outside_cited_evidence" if n["status"] == "completed"
                                             else "node_not_completed"})
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
            raise ValueError("Monitor needs one final verification task allowing only finish; "
                             f"use a nonterminal evidence stage for inspect/measure. "
                             f"Previously ready nodes: {self.ready_ids()}")
        final = finals[0]
        nonfinal_finish = [n["id"] for n in nodes if n is not final and "finish" in n["allowed_actions"]]
        if nonfinal_finish:
            raise ValueError(f"Only final verification may finish: remove finish from {nonfinal_finish}; "
                             f"use final node {final['id']}")
        ancestors = {final["id"]}
        while True:
            expanded = ancestors | {d for i in ancestors for d in by_id[i]["depends_on"]}
            if expanded == ancestors:
                break
            ancestors = expanded
        if ancestors != set(by_id):
            missing = sorted(set(by_id) - ancestors)
            raise ValueError("Final verification must depend on every planned task; "
                             f"missing node IDs {missing}. Add dependency paths from final verification to these tasks")
        covered = {c for n in nodes for c in n["condition_ids"]}
        if covered != requirements:
            raise ValueError("Monitor DAG must cover all language requirements; "
                             f"assign missing IDs {sorted(requirements - covered)} to the tasks that resolve them")

        # The explicit, uniquely verified decision already supplies an omitted final binding.
        normalized_binding = (action["action"] == "finish" and action.get("predicted_id") is not None
                              and plan["current_node_id"] == final["id"]
                              and all(n["status"] == "completed" for n in nodes)
                              and self.active_ids(plan) == [action["predicted_id"]]
                              and not action.get("unresolved") and not final["bound_object_ids"])
        if normalized_binding:
            final["bound_object_ids"] = [action["predicted_id"]]

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
        unresolved = {" ".join(text.split()).casefold() for text in action.get("unresolved", [])}
        for n in nodes:
            goal = re.escape(" ".join(n["goal"].split()).casefold())
            label = re.escape(n["id"].casefold())
            if n["status"] == "completed" and any(re.fullmatch(
                    goal + r"(?:\s*\(\s*" + label + r"\s*\))?", text) for text in unresolved):
                raise ValueError(f"Completed task {n['id']} is still listed in unresolved; "
                                 "keep it pending or resolve the conflicting assessment")
        current = by_id.get(plan["current_node_id"])
        if current is None:
            raise ValueError("Current monitor stage is absent")
        before, after = set(self.active_ids()), set(self.active_ids(plan))
        event = {"revision": len(self.history) + 1, "current_node_id": current["id"],
                 "invalidated_node_ids": sorted(invalidated),
                 "reactivated_target_ids": sorted(after - before),
                 "active_target_ids": sorted(after), "plan": deepcopy(plan)}
        if normalized_binding:
            event["normalized_final_binding"] = True
        if discarded_exclusions:
            event["discarded_exclusions"] = discarded_exclusions

        def reject_action(message):
            if invalidated or discarded_exclusions:
                event["action_error"] = message
                self.plan = plan
                self.history.append(event)
                prefix = (f"Rollback accepted; reset nodes {sorted(invalidated)}." if invalidated else
                          "Unsupported exclusions discarded; those targets remain active.")
                message = (f"{prefix} "
                           f"Ready node IDs: {self.ready_ids()}. Action rejected: {message}")
            raise ValueError(message)

        if action["action"] not in current["allowed_actions"]:
            reject_action(f"Action {action['action']} is not allowed by current stage {current['id']}; "
                          f"allowed actions: {current['allowed_actions']}")
        if action["action"] != "finish":
            if current["status"] != "pending" or any(by_id[d]["status"] != "completed"
                                                      for d in current["depends_on"]):
                reject_action(f"Current stage {current['id']} is not ready; "
                              f"resolve ready stages {self.ready_ids(plan)} first")
        elif action.get("predicted_id") is not None:
            incomplete = [n["id"] for n in nodes if n["status"] != "completed"]
            if current is not final:
                reject_action(f"Finish must use final verification node {final['id']}")
            if incomplete:
                reject_action(f"Finish needs completed tasks: recheck {incomplete} using existing evidence "
                              "or acquire missing evidence before finishing")
            if self.active_ids(plan) != [action["predicted_id"]]:
                reject_action(f"Finish needs one active target matching predicted_id; "
                              f"active targets: {self.active_ids(plan)}")
            if action["predicted_id"] not in current["bound_object_ids"]:
                reject_action(f"Final binding {current['bound_object_ids']} conflicts with "
                              f"predicted_id {action['predicted_id']}")
            if action.get("unresolved"):
                reject_action("Finish cannot select a target while conditions remain unresolved")
        elif not any(text.strip() for text in action.get("unresolved", [])):
            reject_action("Abstention must state missing evidence in unresolved; "
                          f"pending task IDs: {[n['id'] for n in nodes if n['status'] != 'completed']}")
        self.plan = plan
        self.history.append(event)
        return event
