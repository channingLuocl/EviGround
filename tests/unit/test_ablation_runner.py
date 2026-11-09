"""Ablation input locations are explicit and portable across machines."""

import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch


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
