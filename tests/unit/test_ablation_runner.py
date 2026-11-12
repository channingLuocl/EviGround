"""Ablation input locations are explicit and portable across machines."""

import importlib.util
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


RUNNER = Path(__file__).resolve().parents[2] / "scripts/run_nr3d_ablation.py"


def load_runner(environment):
    spec = importlib.util.spec_from_file_location("nr3d_ablation_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, environment, clear=True):
        spec.loader.exec_module(module)
    return module


class AblationRunnerTests(unittest.TestCase):
    def test_key_input_skips_empty_lines_and_preserves_slot_assignment(self):
        runner = load_runner({})
        with redirect_stdout(io.StringIO()):
            keys = runner.read_keys(io.StringIO('\n  \n[" unit-test-key "]\n'))
        self.assertEqual(keys, ["unit-test-key"] * 2)
        self.assertEqual(runner.read_keys(io.StringIO('["first-test-key", "second-test-key"]\n')),
                         ["first-test-key", "second-test-key"])

    def test_key_input_reports_closed_stdin_without_json_traceback(self):
        runner = load_runner({})
        for contents in ("", "\n  \n"):
            with self.subTest(contents=contents), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "No API key received"):
                    runner.read_keys(io.StringIO(contents))

    def test_invalid_key_input_never_exposes_the_submitted_value(self):
        runner = load_runner({})
        private = "private-test-credential"
        for contents in (private, json.dumps(private), json.dumps({"key": private}),
                         "[]", '["  "]', '[123]', '["a","b","c"]'):
            with self.subTest(contents=contents):
                with self.assertRaises(ValueError) as caught:
                    runner.read_keys(io.StringIO(contents + "\n"))
                self.assertNotIn(private, str(caught.exception))
                self.assertIn("JSON array", str(caught.exception))

    def test_defaults_do_not_depend_on_a_private_endpoint_or_home_directory(self):
        runner = load_runner({})
        self.assertEqual(runner.BASE_URL, "https://api.openai.com/v1")
        self.assertEqual(runner.SCENES, runner.ROOT / "local_outputs")
        self.assertEqual(runner.EXPORT, runner.ROOT / "local_outputs/classifier_predictions")

    def test_explicit_paths_and_endpoint_reach_worker_environment(self):
        runner = load_runner({
            "EVIGROUND_ABLATION_STUDY": "/study",
            "EVIGROUND_ABLATION_SCENES": "/data/scenes",
            "EVIGROUND_ABLATION_PREDICTIONS": "/data/predictions",
            "EVIGROUND_ABLATION_MODEL": "configured-model",
            "EVIGROUND_ABLATION_BASE_URL": "https://ablation.example/v1",
            "EVIGROUND_BASE_URL": "https://other.example/v1",
        })
        self.assertEqual(runner.SCENES, Path("/data/scenes"))
        self.assertEqual(runner.EXPORT, Path("/data/predictions"))
        worker = runner.environment("unit-test-key", runner.VARIANTS["A3"], 2)
        self.assertEqual(worker["PYTHONPATH"], "/study/source")
        self.assertEqual(worker["EVIGROUND_MODEL"], "configured-model")
        self.assertEqual(worker["EVIGROUND_BASE_URL"], "https://ablation.example/v1")
        self.assertEqual(worker["EVIGROUND_INSPECT_ENABLED"], "0")
        self.assertEqual(worker["EVIGROUND_MEASURE_ENABLED"], "0")
        self.assertEqual(worker["EVIGROUND_ALLOW_HTTP"], "0")

    def test_shared_endpoint_is_used_when_ablation_override_is_absent(self):
        runner = load_runner({"EVIGROUND_BASE_URL": "https://shared.example/v1"})
        self.assertEqual(runner.BASE_URL, "https://shared.example/v1")

    def test_shared_key_preflight_checks_both_slots_sequentially_and_keeps_failure(self):
        runner = load_runner({})
        failure = Mock(returncode=1, stdout=json.dumps({"ok": False, "error": "HTTP 401", "http_status": 401}))
        success = Mock(returncode=0, stdout=json.dumps({"ok": True, "finish_reason": "stop",
                                                       "content": '{"color":"red"}', "response_model": runner.MODEL}))
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(runner, "STUDY", Path(directory)),
              patch.object(runner, "ThreadPoolExecutor") as pool,
              patch.object(runner.subprocess, "run", side_effect=[failure, success]) as run,
              redirect_stdout(io.StringIO())):
            with self.assertRaisesRegex(ValueError, "Key slots 1 failed"):
                runner.preflight(["unit-test-key"] * 2, 3)
            pool.assert_not_called()
            self.assertEqual(run.call_count, 2)
            first, second = run.call_args_list
            self.assertEqual(first.kwargs["env"], second.kwargs["env"])
            self.assertEqual(json.loads((runner.STUDY / "preflight_key_1.json").read_text())["http_status"], 401)
            self.assertTrue(json.loads((runner.STUDY / "preflight_key_2.json").read_text())["ok"])

    def test_preflight_failure_reports_provider_diagnostics_without_credentials(self):
        from eviground.api import client

        runner = load_runner({})
        process = Mock(returncode=1, stdout='{"ok":false,"error":"test failure"}')
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(runner, "STUDY", Path(directory)),
              patch.object(runner.subprocess, "run", return_value=process) as run,
              redirect_stdout(io.StringIO())):
            with self.assertRaises(ValueError):
                runner.preflight(["unit-test-key"] * 2, 3)
            script = run.call_args.args[0][2]
            secret = "unit-test-private-credential"
            failure = client.APIError("HTTP 401", diagnostic={"failure_kind": "http_4xx", "http_status": 401,
                "request_id": "test-request-id", "response_body": f"Invalid API key: {secret}"})
            output = io.StringIO()
            with (patch.dict(os.environ, {"EVIGROUND_API_KEY": secret}),
                  patch.object(client, "chat", side_effect=failure),
                  patch.object(runner.sys, "argv", ["-c", str(Path(directory) / "test.png")]),
                  redirect_stdout(output)):
                with self.assertRaises(SystemExit) as stopped:
                    exec(script, {})
            self.assertEqual(stopped.exception.code, 1)
            result = json.loads(output.getvalue())
            self.assertEqual(result["http_status"], 401)
            self.assertEqual(result["request_id"], "test-request-id")
            self.assertNotIn(secret, output.getvalue())

    def test_module_matrix_overrides_ambient_switches(self):
        runner = load_runner({"EVIGROUND_ABLATION_DAG_MONITOR_ENABLED": "0"})
        self.assertEqual(list(runner.VARIANTS), ["A0", "A1", "A2", "A3", "A4"])
        expected = [(1, 1, 1, 1, 5, 1), (1, 0, 1, 1, 5, 1),
                    (0, 1, 1, 1, 5, 1), (1, 1, 0, 0, 5, 0), (1, 1, 1, 1, 1, 0)]
        for variant, values in zip(runner.VARIANTS.values(), expected):
            self.assertEqual(tuple(variant[k] for k in ("monitor", "annealing", "inspect", "measure", "top_k", "guard")), values)
            with patch.dict(os.environ, {"EVIGROUND_DAG_MONITOR_ENABLED": "1",
                                         "EVIGROUND_POLICY_ANNEALING_ENABLED": "0"}):
                worker = runner.environment("unit-test-key", variant, 2)
            self.assertEqual(worker["EVIGROUND_DAG_MONITOR_ENABLED"], str(variant["monitor"]))
            self.assertEqual(worker["EVIGROUND_POLICY_ANNEALING_ENABLED"], str(variant["annealing"]))
            self.assertEqual(worker["EVIGROUND_SUCCESSFUL_TOOL_BUDGET"], "1")
            self.assertEqual(worker["EVIGROUND_DECISION_MAX_TOKENS"], "4000")

    def test_explicit_evidence_and_repair_budgets_reach_workers(self):
        runner = load_runner({"EVIGROUND_ABLATION_REVIEW_LIMIT": "12",
                              "EVIGROUND_ABLATION_DAG_REPAIR_LIMIT": "6"})
        worker = runner.environment("unit-test-key", runner.VARIANTS["A0"], 3)
        self.assertEqual(worker["EVIGROUND_DAG_REPAIR_LIMIT"], "6")
        self.assertEqual(runner.REVIEW_LIMIT, 12)
        with tempfile.TemporaryDirectory() as directory:
            runner.STUDY = Path(directory)
            for slot in (1, 2):
                output = runner.STUDY / f"runs/A0/key_slot_{slot}"
                output.mkdir(parents=True)
                (output / "results.json").write_text(json.dumps({"summary": {"complete": True}}))
            process = Mock(returncode=0)
            process.poll.return_value = 0
            with patch.object(runner.subprocess, "Popen", return_value=process) as popen:
                runner.run_pair("A0", ["unit-test-key"] * 2, [Path("a"), Path("b")], 3, "runs")
            for call in popen.call_args_list:
                command = call.args[0]
                self.assertEqual(command[command.index("--review-limit") + 1], "12")
        for environment in ({"EVIGROUND_ABLATION_REVIEW_LIMIT": "0"},
                            {"EVIGROUND_ABLATION_DAG_REPAIR_LIMIT": "-1"}):
            with self.assertRaises(ValueError):
                load_runner(environment)

    def test_budgets_are_frozen_in_the_protocol(self):
        runner = load_runner({"EVIGROUND_ABLATION_DAG_MONITOR_ENABLED": "1",
                              "EVIGROUND_ABLATION_REVIEW_LIMIT": "12",
                              "EVIGROUND_ABLATION_DAG_REPAIR_LIMIT": "6"})
        with tempfile.TemporaryDirectory() as directory:
            runner.ROOT = Path(directory) / "repo"
            runner.STUDY = Path(directory) / "study"
            source = runner.ROOT / "src/eviground/model.py"
            source.parent.mkdir(parents=True)
            source.write_text("MODEL = 'offline fixture'\n")
            full = runner.ROOT / "data/queries/nr3d_val.json"
            full.parent.mkdir(parents=True)
            full.write_text(json.dumps({"rows": [{"csv_row": 1500}, {"csv_row": 1501}]}))
            runner.STUDY.mkdir()
            subset = {"rows": [{"csv_row": i, "scene": f"scene{i % 130:04d}"} for i in range(1500)],
                      "sampling": {"seed": 20261104}}
            (runner.STUDY / "nr3d_subset_1500.json").write_text(json.dumps(subset))
            manifests = runner.freeze_inputs(3)
            protocol = json.loads((runner.STUDY / "protocol.json").read_text())
            self.assertEqual(protocol["successful_tool_budget"], 12)
            self.assertEqual(protocol["repair_allowance"], 6)
            self.assertEqual([len(json.loads(p.read_text())["rows"]) for p in manifests], [750, 750])
            runner.DAG_REPAIR_LIMIT = 7
            with self.assertRaisesRegex(ValueError, "Frozen input changed"):
                runner.freeze_inputs(3)

    def test_pilot_only_never_starts_formal_evaluation(self):
        runner = load_runner({})
        docs = [{"summary": {"complete": True, "errors": 0}, "results": []}]
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(runner, "STUDY", Path(directory)),
              patch.object(runner, "freeze_inputs", return_value=[]),
              patch.object(runner, "preflight"),
              patch.object(runner, "check_pilots"),
              patch.object(runner, "run_pair", return_value=docs) as run_pair,
              patch.object(runner.sys, "argv", [str(RUNNER), "--technical-pilot-only", "--end-condition", "A0"]),
              patch.object(runner.sys, "stdin", io.StringIO('["unit-test-key"]\n'))):
            runner.main()
            self.assertEqual(run_pair.call_count, 1)
            self.assertEqual(run_pair.call_args.args[-1], "technical_pilot")
            self.assertEqual(json.loads((runner.STUDY / "status.json").read_text())["phase"],
                             "technical_pilot_complete")

    def test_dag_pilot_gate_checks_execution_not_accuracy(self):
        runner = load_runner({"EVIGROUND_ABLATION_DAG_MONITOR_ENABLED": "1"})
        with tempfile.TemporaryDirectory() as directory:
            runner.STUDY = Path(directory)
            path = runner.STUDY / "prediction.json"
            prediction = {"predicted_id": None, "decision_reasons": ["essential visual evidence unavailable"],
                          "active_trace": {"dag_monitor_history": [{"revision": 1}],
                                           "policy_annealing_enabled": True, "policy_annealing_history": [{"stage": "explore"}],
                                           "react_budget": {"accounting": "successful_tools", "failed_checks": 1}}}
            docs = [{"summary": {"complete": True, "errors": 0},
                     "results": [{"csv_row": 7, "correct": False, "prediction_path": str(path)}]}]
            path.write_text(json.dumps(prediction))
            runner.check_pilots("A0", docs)
            for variant in ("no_plan", "exhausted_repairs"):
                invalid = json.loads(json.dumps(prediction))
                if variant == "no_plan":
                    invalid["active_trace"]["dag_monitor_history"] = []
                else:
                    invalid["active_trace"]["react_budget"]["failed_checks"] = 7
                path.write_text(json.dumps(invalid))
                with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, "DAG technical pilot failed"):
                    runner.check_pilots("A0", docs)

    def test_dag_pilot_failure_reports_validation_feedback(self):
        runner = load_runner({"EVIGROUND_ABLATION_DAG_MONITOR_ENABLED": "1"})
        with tempfile.TemporaryDirectory() as directory:
            runner.STUDY = Path(directory)
            path = runner.STUDY / "prediction.json"
            prediction = {"predicted_id": None, "decision_reasons": ["DAG validation repair budget exhausted"],
                          "active_trace": {"dag_monitor_history": [{"revision": 1}],
                                           "policy_annealing_enabled": True, "policy_annealing_history": [{"stage": "explore"}],
                                           "react_budget": {"accounting": "successful_tools", "failed_checks": 7, "successful_tool_calls": 0},
                                           "react_observations": [{"tool": "monitor_check", "error": "Final node must allow finish"}]}}
            path.write_text(json.dumps(prediction))
            docs = [{"summary": {"complete": True, "errors": 0},
                     "results": [{"csv_row": 20, "correct": False, "prediction_path": str(path)}]}]
            with self.assertRaisesRegex(ValueError, "row 20: 7 failed checks, 0 evidence tools; last: Final node must allow finish"):
                runner.check_pilots("A0", docs)
            diagnostic = json.loads((runner.STUDY / "technical_pilot/A0/diagnostics.json").read_text())[0]
            self.assertEqual(diagnostic["validation_feedback"],
                             [{"tool": "monitor_check", "message": "Final node must allow finish"}])

    def test_empty_category_pool_is_a_valid_pilot_abstention(self):
        runner = load_runner({})
        with tempfile.TemporaryDirectory() as directory:
            runner.STUDY = Path(directory)
            path = runner.STUDY / "prediction.json"
            path.write_text(json.dumps({"predicted_id": None, "category_target_ids": [],
                                        "decision_reasons": ["no matching category"],
                                        "active_trace": {"stop_reason": "category_abstention"}}))
            docs = [{"summary": {"complete": True, "errors": 0},
                     "results": [{"csv_row": 1, "prediction_path": str(path)}]}]
            for name in ("A0", "A4"):
                runner.check_pilots(name, docs)
                saved = json.loads((runner.STUDY / "technical_pilot" / name / "diagnostics.json").read_text())
                self.assertTrue(saved[0]["category_abstention"])
                self.assertFalse(saved[0]["answered"])
