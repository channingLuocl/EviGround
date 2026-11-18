"""The fallback inventory must not duplicate an already registered condition."""

from copy import deepcopy
import unittest

from eviground.grounding.dag_monitor import DagMonitor
from eviground.grounding.settings import evidence_inventory


class RequirementInventoryTests(unittest.TestCase):
    query = "cabinet touching the wall farthest from entrance door"
    phrase = "touching the wall farthest from entrance door"

    def parsed(self):
        return {
            "Target": [],
            "condition_records": [{
                "condition_type": "unhandled", "source_quote": self.phrase,
                "target_role": None, "reference_role": None,
                "binding_status": "unbound",
            }],
            "Unhandled": [self.phrase],
        }

    def test_registered_unhandled_condition_keeps_one_requirement_and_dependency(self):
        parsed = self.parsed()
        before = deepcopy(parsed)
        rows = evidence_inventory(self.query, parsed)["items"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_quote"], self.phrase)
        self.assertEqual(rows[0]["condition"], "unhandled")
        self.assertEqual(rows[0]["binding_status"], "unbound")
        self.assertEqual(rows[0]["dependencies"], ["obtain quoted predicate evidence"])
        self.assertEqual(parsed, before)

    def test_monitor_registers_original_description_and_condition_once(self):
        monitor = DagMonitor(self.query, self.parsed(), {1, 2}, {1, 2},
                             ["inspect", "measure", "finish"])
        self.assertEqual([row["id"] for row in monitor.requirements], ["q0", "q1"])
        self.assertEqual(monitor.requirements[0]["source_quote"], self.query)

    def test_unhandled_fallback_remains_when_condition_records_are_absent(self):
        parsed = self.parsed()
        parsed["condition_records"] = []
        rows = evidence_inventory(self.query, parsed)["items"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["dependencies"], ["obtain quoted predicate evidence"])

    def test_same_quote_different_predicate_is_preserved(self):
        parsed = self.parsed()
        parsed["condition_records"][0]["condition_type"] = "relation:near"
        rows = evidence_inventory(self.query, parsed)["items"]
        self.assertEqual([row["condition"] for row in rows], ["relation:near", "unhandled"])

    def test_distinct_quotes_and_existing_reference_dependency_are_preserved(self):
        parsed = self.parsed()
        parsed["condition_records"][0]["reference_role"] = {"categories": ["wall"]}
        parsed["Unhandled"].append("entrance door")
        rows = evidence_inventory(self.query, parsed)["items"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["reference_role"], {"categories": ["wall"]})
        self.assertEqual(rows[0]["dependencies"], [
            "bind reference", "obtain quoted predicate evidence"])
        self.assertEqual(rows[1]["source_quote"], "entrance door")

    def test_repeated_fallback_does_not_duplicate_the_dependency(self):
        parsed = self.parsed()
        parsed["Unhandled"] *= 2
        rows = evidence_inventory(self.query, parsed)["items"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["dependencies"], ["obtain quoted predicate evidence"])

    def test_registered_roles_remain_distinct(self):
        parsed = self.parsed()
        first = parsed["condition_records"][0]
        first["target_role"] = {"category": "cabinet"}
        second = {**first, "target_role": None,
                  "reference_role": {"categories": ["wall"]}}
        parsed["condition_records"].append(second)
        rows = evidence_inventory(self.query, parsed)["items"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["target_role"], first["target_role"])
        self.assertEqual(rows[1]["reference_role"], second["reference_role"])

    def test_nonquoted_fallback_is_not_registered(self):
        parsed = {"Target": [], "Unhandled": ["invented clause"]}
        self.assertEqual(evidence_inventory(self.query, parsed)["items"], [])


if __name__ == "__main__":
    unittest.main()
