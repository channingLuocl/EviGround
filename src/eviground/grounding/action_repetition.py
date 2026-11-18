"""Positive JSON Schema constraints excluding previously acquired observations."""

from copy import deepcopy
from decimal import Decimal
import json
import re


def exclude_acquired_actions(schema, acquired, known_ids, available, max_focus=8,
                             invalid_focus_groups=()):
    """Intersect the native schema with acquired and known-invalid exclusions.

    The local Guidance backend supports these positive intersections. Neither
    reasoning nor DAG fields participate in observation identity.
    """
    if "inspect" in available:
        schema = deepcopy(schema)
        for branch in schema.get("anyOf", [schema]):
            properties = branch["properties"]
            for name in ("azimuth_deg", "elevation_deg"):
                spec = properties[name]
                for variant in spec.get("anyOf", [spec]):
                    if variant.get("type") == "number":
                        variant["type"] = "integer"
    known = sorted(known_ids)

    def fields(**properties):
        # Partial constraints must not close the original action/Monitor object.
        return {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": True}

    def different_group(previous, maximum):
        # A distinct set either has a different length or a first new member.
        # Runtime validation still rejects repeated IDs within an array.
        size = len(previous)
        branches = []
        if size:
            branches.append({"type": "array", "maxItems": size - 1})
        if size < maximum:
            branches.append({"type": "array", "minItems": size + 1})
        outside = sorted(set(known) - set(previous))
        if outside:
            for position in range(size):
                branches.append({
                    "type": "array", "minItems": size, "maxItems": size,
                    "prefixItems": [{"enum": previous}] * position + [{"enum": outside}],
                    "items": {"enum": known},
                })
        return branches

    exclusions = []
    for group in invalid_focus_groups:
        # A same-length array drawn from this smaller set must contain duplicates.
        # Do not remove any distinct-ID action, or repair the model's proposal.
        if (not group or any(type(i) is not int or i not in known for i in group)
                or len(set(group)) == len(group)):
            raise ValueError("Invalid focus exclusion must contain repeated known IDs")
        exclusions.append({"anyOf": [fields(focus_ids=changed) for changed in
                                    different_group(group, max_focus)]})
    for previous in acquired:
        branches = []
        other_actions = [name for name in available if name != previous["action"]]
        if other_actions:
            branches.append(fields(action={"enum": other_actions}))
        branches.extend(fields(focus_ids=group) for group in
                        different_group(previous["focus_ids"], max_focus))
        if previous["action"] == "inspect":
            branches.append(fields(context={"enum": [
                name for name in ("scene", "isolated") if name != previous["context"]]}))
            for name, default in (("azimuth_deg", 315), ("elevation_deg", 30)):
                value = previous[name]
                angles = [{"type": "number", "exclusiveMaximum": value - 0.0000005},
                          {"type": "number", "minimum": value + 0.0000005}]
                if value != default:
                    angles.append({"type": "null"})
                branches.append(fields(**{name: {"anyOf": angles}}))
        elif previous["frame"] is None:
            branches.append(fields(frame={"type": "object"}))
        else:
            branches.append(fields(frame={"type": "null"}))
            for name, group in previous["frame"].items():
                branches.extend(fields(frame=fields(**{name: changed})) for changed in
                                different_group(group, len(known)))
        exclusions.append({"anyOf": branches})
    if not exclusions:
        return schema
    return {**schema, "allOf": [*schema.get("allOf", []), *exclusions],
            "x-eviground-acquired-actions": acquired,
            **({"x-eviground-invalid-focus-groups": list(invalid_focus_groups)}
               if invalid_focus_groups else {})}


def no_repeat_grammar(schema):
    """Factor the small tool identity out of the full native Monitor grammar.

    Intersecting many whole action schemas expands the Monitor combinatorially.
    Guidance's regular-expression complement can instead mask only tool identity;
    every remaining field retains its original native JSON Schema subgrammar.
    The complete response still passes the original schema and semantic checks.
    """
    previous = schema["x-eviground-acquired-actions"]
    invalid_groups = schema.get("x-eviground-invalid-focus-groups", [])
    if len(schema.get("allOf", [])) != len(previous) + len(invalid_groups):
        raise ValueError("Cannot factor an action schema with other root intersections")
    identity = ("action", "focus_ids", "context", "azimuth_deg", "elevation_deg", "frame")
    whitespace = r"[ \t\r\n]*"
    rules, choices, values = [], [], {}

    def literal(value):
        return re.escape(json.dumps(value, separators=(",", ":")))

    def group(ids, minimum, maximum):
        member = "(?:" + "|".join(literal(i) for i in ids) + ")"
        if maximum == 0:
            return r"\[" + whitespace + r"\]"
        body = member + "(?:" + whitespace + "," + whitespace + member + "){" + str(max(0, minimum - 1)) + "," + str(maximum - 1) + "}"
        return r"\[" + whitespace + ("(?:" + body + ")?" if minimum == 0 else body) + whitespace + r"\]"

    def fields(properties):
        return (whitespace + "," + whitespace).join(
            literal(name) + whitespace + ":" + whitespace + value
            for name, value in properties.items())

    def frame(value, known=None):
        if value is None:
            return "null"
        return r"\{" + whitespace + fields({name: group(ids, len(ids), len(ids))
            if known is None else group(known, 0, len(known))
            for name, ids in ((key, value[key]) for key in ("origin_ids", "facing_ids"))}) + whitespace + r"\}"

    def number(value, default):
        # Exclude the entire one-millionth-degree camera bin, including long
        # decimal spellings such as 30.000000000000004 around the default 30.
        center = int(Decimal(str(value)) * 10000000)
        alternatives = []
        for scaled in range(max(0, center - 5), center + 5):
            integer, fraction = divmod(scaled, 10000000)
            digits = str(fraction).zfill(7)
            alternatives.append(str(integer) + r"\." + digits + "[0-9]*")
            short = digits.rstrip('0')
            alternatives.append(str(integer) + (r"\." + short if short else r"(?:\.0+)?"))
        exact = "(?:" + "|".join(alternatives) + ")"
        return "(?:" + exact + "|null)" if value == default else exact

    def native(spec):
        encoded = json.dumps({**spec, "$defs": schema.get("$defs", {})}, separators=(",", ":"))
        if encoded not in values:
            name = "native_" + str(len(values))
            values[encoded] = name
            rules.append(name + ": %json " + encoded)
        return values[encoded]

    if "anyOf" not in schema:
        raise ValueError("Action exclusion grammar requires action-specific schema branches")
    for index, branch in enumerate(schema["anyOf"]):
        properties = branch["properties"]
        action = properties["action"]["enum"][0]
        name = "action_" + str(index)
        choices.append(name)
        reason_first = next(iter(properties)) == "reason"
        beginning = '\"{\" '
        if reason_first:
            beginning += (' '.join([json.dumps(json.dumps("reason")), json.dumps(":"),
                                    native(properties["reason"]), json.dumps(",")]) + ' ')
        if action == "finish":
            if next(iter(properties)) != "action" and not reason_first:
                rules.append(name + ": " + native(branch))
                continue
            # Match the action in one lexeme, like tool identities. Splitting it
            # across JSON/literal rules lets Guidance's longer tool lexemes prune
            # this branch at the shared '"action":"' prefix.
            terminal = "FINISH_" + str(index)
            rules.append(terminal + ": /" + fields(dict(action=literal(action))) + "/")
            suffix = []
            for key, spec in properties.items():
                if key != "action" and not (reason_first and key == "reason"):
                    suffix.extend([json.dumps(','), json.dumps(json.dumps(key)), json.dumps(':'), native(spec)])
            rules.append(name + ': ' + beginning + terminal + ' ' + ' '.join(suffix) + ' "}"')
            continue
        known = properties["focus_ids"]["items"]["enum"]
        prefix = dict(action=literal(action), focus_ids=group(known, 1, properties["focus_ids"]["maxItems"]))
        if action == "inspect":
            prefix.update(context='(?:"scene"|"isolated")',
                azimuth_deg=r"(?:null|(?:0|[1-9][0-9]?|[12][0-9]{2}|3[0-5][0-9])(?:\.0+)?)",
                elevation_deg=r"(?:null|(?:[1-6][0-9]|7[0-5])(?:\.0+)?)", frame="null")
        else:
            prefix.update(context="null", azimuth_deg="null", elevation_deg="null",
                          frame="(?:null|" + frame({"origin_ids": [], "facing_ids": []}, known) + ")")
        blocked = [fields({**prefix, "focus_ids": group(sorted(set(ids)), len(ids), len(ids))})
                   for ids in invalid_groups]
        for old in previous:
            if old["action"] != action:
                continue
            repeated = dict(action=literal(action), focus_ids=group(old["focus_ids"], len(old["focus_ids"]), len(old["focus_ids"])))
            if action == "inspect":
                repeated.update(context=literal(old["context"]), azimuth_deg=number(old["azimuth_deg"], 315),
                                elevation_deg=number(old["elevation_deg"], 30), frame="null")
            else:
                repeated.update(context="null", azimuth_deg="null", elevation_deg="null", frame=frame(old["frame"]))
            blocked.append(fields(repeated))
        terminal = "IDENTITY_" + str(index)
        rule = terminal + ": /" + fields(prefix) + "/"
        if blocked:
            rule += " & ~/(?:" + "|".join(blocked) + ")/"
        rules.append(rule)
        suffix = []
        for key, spec in properties.items():
            if key not in identity and not (reason_first and key == "reason"):
                suffix.extend([json.dumps(','), json.dumps(json.dumps(key)), json.dumps(':'), native(spec)])
        rules.append(name + ': ' + beginning + terminal + ' ' + ' '.join(suffix) + ' "}"')
    return "start: " + " | ".join(choices) + "\n" + "\n".join(rules) + '\n%ignore /[ \\t\\r\\n]+/\n'
