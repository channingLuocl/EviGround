"""ScanRefer must validate private API configuration before starting workers."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


LAUNCHER = Path(__file__).resolve().parents[2] / "scripts/run_scanrefer_r18.sh"


class ScanReferLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        local = self.root / "local"
        local.mkdir()
        (self.root / ".env").write_text("SEEGROUND_BASE_URL=https://old.example/v1\n")
        (local / "qwen_api.py").write_text(
            "import os\n"
            "class QwenAPIError(ValueError): pass\n"
            "def _settings():\n"
            "    if not all(os.getenv(k, '').strip() for k in "
            "('SEEGROUND_BASE_URL', 'SEEGROUND_MODEL', 'SEEGROUND_API_KEY')):\n"
            "        raise QwenAPIError('API configuration missing')\n"
        )
        (local / "run_grounding_v9.py").write_text(
            "import json, os, sys\nfrom pathlib import Path\n"
            "Path(sys.argv[sys.argv.index('--output')+1]).write_text(json.dumps({"
            "'base': os.environ['SEEGROUND_BASE_URL'], "
            "'model': os.environ['SEEGROUND_MODEL']}))\n"
        )
        self.output = self.root / "batch_started.json"
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("SEEGROUND_")}
        self.env.update(HOME=str(self.root), SEEGROUND_R18_ROOT=str(self.root),
                        SEEGROUND_V9_PYTHON=sys.executable)

    def launch(self, *args):
        return subprocess.run(
            ["bash", str(LAUNCHER), "manifest", "predictions", "scenes",
             str(self.output), *args], env=self.env, capture_output=True, text=True
        )

    def write_credentials(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("SEEGROUND_BASE_URL=https://new.example/v1\n"
                        "SEEGROUND_API_KEY=unit-test-secret\n")
        path.chmod(0o600)

    def test_missing_credentials_fail_before_batch_even_in_preflight(self):
        result = self.launch("--prepare-only")
        self.assertEqual(result.returncode, 2)
        self.assertIn("API 配置检查失败", result.stderr)
        self.assertFalse(self.output.exists())

    def test_default_private_config_supplies_matching_endpoint_and_key(self):
        self.write_credentials(self.root / ".config/seeground/qwen-token-plan-20261022.env")
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.output.read_text()),
                         {"base": "https://new.example/v1", "model": "qwen3.7-plus"})
        self.assertNotIn("unit-test-secret", result.stdout + result.stderr)

    def test_explicit_config_path_is_respected_and_missing_path_fails(self):
        config = self.root / "private.env"
        self.env["SEEGROUND_ENV_FILE"] = str(config)
        result = self.launch()
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.output.exists())
        self.write_credentials(config)
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.output.exists())


if __name__ == "__main__":
    unittest.main()
