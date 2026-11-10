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
        worker = runner.environment("unit-test-key", runner.VARIANTS["E1"], 2)
        self.assertEqual(worker["PYTHONPATH"], "/study/source")
        self.assertEqual(worker["EVIGROUND_MODEL"], "configured-model")
        self.assertEqual(worker["EVIGROUND_BASE_URL"], "https://ablation.example/v1")
        self.assertEqual(worker["EVIGROUND_INSPECT_ENABLED"], "0")
        self.assertEqual(worker["EVIGROUND_MEASURE_ENABLED"], "1")
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

    def test_monitor_switch_is_explicitly_forwarded_to_frozen_workers(self):
        runner = load_runner({"EVIGROUND_ABLATION_DAG_MONITOR_ENABLED": "1"})
        worker = runner.environment("unit-test-key", runner.VARIANTS["E0"], 2)
        self.assertEqual(worker["EVIGROUND_DAG_MONITOR_ENABLED"], "1")
        baseline = load_runner({})
        self.assertEqual(baseline.environment("unit-test-key", baseline.VARIANTS["E0"], 2)
                         ["EVIGROUND_DAG_MONITOR_ENABLED"], "0")
        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            load_runner({"EVIGROUND_ABLATION_DAG_MONITOR_ENABLED": "true"})

    def test_explicit_evidence_and_repair_budgets_reach_workers(self):
        runner = load_runner({"EVIGROUND_ABLATION_REVIEW_LIMIT": "12",
                              "EVIGROUND_ABLATION_DAG_REPAIR_LIMIT": "6"})
        worker = runner.environment("unit-test-key", runner.VARIANTS["E0"], 3)
        self.assertEqual(worker["EVIGROUND_DAG_REPAIR_LIMIT"], "6")
        self.assertEqual(runner.REVIEW_LIMIT, 12)
        with tempfile.TemporaryDirectory() as directory:
            runner.STUDY = Path(directory)
            for slot in (1, 2):
                output = runner.STUDY / f"runs/E0/key_slot_{slot}"
                output.mkdir(parents=True)
                (output / "results.json").write_text(json.dumps({"summary": {"complete": True}}))
            process = Mock(returncode=0)
            process.poll.return_value = 0
            with patch.object(runner.subprocess, "Popen", return_value=process) as popen:
                runner.run_pair("E0", ["unit-test-key"] * 2, [Path("a"), Path("b")], 3, "runs")
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
            self.assertEqual(protocol["nonterminal_evidence_budget"], 12)
            self.assertEqual(protocol["dag_monitor_repair_limit"], 6)
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
              patch.object(runner, "run_pair", return_value=docs) as run_pair,
              patch.object(runner.sys, "argv", [str(RUNNER), "--technical-pilot-only", "--end-condition", "E0"]),
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
                                           "dag_monitor_budget": {"failed_checks": 1}}}
            docs = [{"summary": {"complete": True, "errors": 0},
                     "results": [{"csv_row": 7, "correct": False, "prediction_path": str(path)}]}]
            path.write_text(json.dumps(prediction))
            runner.check_pilots("E0", docs)
            for variant in ("no_plan", "exhausted_repairs"):
                invalid = json.loads(json.dumps(prediction))
                if variant == "no_plan":
                    invalid["active_trace"]["dag_monitor_history"] = []
                else:
                    invalid["active_trace"]["dag_monitor_budget"]["failed_checks"] = 7
                path.write_text(json.dumps(invalid))
                with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, "DAG technical pilot failed"):
                    runner.check_pilots("E0", docs)
