"""The public ScanRefer command must dispatch to the IoU-scored pipeline."""

import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


READY = Path(__file__).resolve().parents[2] / "scripts/scanrefer_ready.sh"


class ScanReferReadyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "data"
        scripts = self.root / "scripts"
        scripts.mkdir()
        self.ready = scripts / "scanrefer_ready.sh"
        source = READY.read_text().replace(
            "DATA=/Volumes/KESU/datasets/ScanRefer", "DATA=" + shlex.quote(str(self.data))
        ).replace(
            "PY=/Users/luocl/Desktop/seeground/reproductions/SeeGround/.venv/bin/python",
            "PY=" + shlex.quote(sys.executable),
        )
        self.ready.write_text(source.replace("/usr/bin/caffeinate -i ", ""))
        paths = ["scanrefer_val_manifest.json", "scanrefer_mask3d_prepared_scenes/protocol.json",
                 "scanrefer_mask3d_r8_verified_export/EXPORT_VERIFIED.json",
                 "proposal_policies/mask_nms080_with_scores_v1.json",
                 "experiment_sources/r23_scores_recovery_v1/src/eviground/__main__.py"]
        for name in paths:
            path = self.data / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        program = "import json,os,sys;print(json.dumps({'args':sys.argv[1:],'source':os.environ['EVIGROUND_SOURCE_ROOT']}))"
        (scripts / "run_eviground_local.sh").write_text(
            "#!/bin/bash\nexec " + shlex.quote(sys.executable) + " -c "
            + shlex.quote(program) + ' "$@"\n'
        )

    def launch(self, action):
        return subprocess.run(["bash", str(self.ready), action],
                              capture_output=True, text=True, timeout=10)

    def test_preflight_and_run_use_new_pipeline_and_separate_output(self):
        for action in ("preflight", "run"):
            with self.subTest(action=action):
                result = self.launch(action)
                self.assertEqual(result.returncode, 0, result.stderr)
                value = json.loads(result.stdout)
                args = value["args"]
                self.assertEqual("--prepare-only" in args, action == "preflight")
                self.assertEqual(args[args.index("--output") + 1],
                                 str(self.data / "runs/scanrefer_r23_top5_thinking_single_pass"))
                self.assertEqual(args[args.index("--proposal-policy") + 1],
                                 str(self.data / "proposal_policies/mask_nms080_with_scores_v1.json"))
                self.assertEqual(value["source"],
                                 str(self.data / "experiment_sources/r23_scores_recovery_v1/src"))
                self.assertEqual(args[args.index("--category-top-k") + 1], "5")

    def test_missing_policy_fails_before_dispatching_a_batch(self):
        (self.data / "proposal_policies/mask_nms080_with_scores_v1.json").unlink()
        result = self.launch("run")
        self.assertEqual(result.returncode, 2)
        self.assertIn("尚未就绪", result.stderr)
        self.assertEqual(result.stdout, "")
