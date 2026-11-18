"""Native runs reject old inputs and report measured wall time separately."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from eviground.evaluation.runtime import gpu_sample, observe_batch


class NativeRuntimeTests(unittest.TestCase):
    def test_gpu_collection_failure_remains_explicit(self):
        with patch('eviground.evaluation.runtime.subprocess.run', side_effect=FileNotFoundError('no GPU')):
            record = gpu_sample()
        self.assertIn('error', record)
        self.assertNotIn('gpus', record)

    def test_batch_throughput_uses_new_results_and_measured_wall_time(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {'EVIGROUND_MONITOR_GPU': '0'}):
            results = [1]
            with observe_batch(root, lambda: len(results)):
                results.extend([2, 3])
            record = json.loads(next((Path(root) / 'batch_executions').glob('*.json')).read_text())
        self.assertEqual(record['processed_increment'], 2)
        self.assertGreater(record['wall_time_s'], 0)
        self.assertAlmostEqual(record['throughput_samples_per_s'], 2 / record['wall_time_s'])
        self.assertTrue(record['complete'])

    def test_launcher_binds_native_source_and_prevents_old_override(self):
        launcher = Path(__file__).resolve().parents[2] / 'scripts/run_qwen2_vl.sh'
        with tempfile.TemporaryDirectory() as root:
            capture = Path(root) / 'capture'
            capture.write_text('#!' + sys.executable + '\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
            capture.chmod(0o700)
            env = {**os.environ, 'PYTHON': str(capture), 'EVIGROUND_NATIVE_INPUT_ROOT': root}
            for dataset, source in (('nr3d', 'seeground_pointnext_clip_reproduction'),
                                    ('scanrefer', 'seeground_mask3d_scannet200')):
                result = subprocess.run(['bash', str(launcher), '2b', dataset, '--prepare-only'],
                    env=env, capture_output=True, text=True, check=True)
                args = json.loads(result.stdout)
                self.assertEqual(args[args.index('--expected-category-source') + 1], source)
                self.assertEqual(args[args.index('--category-cache') + 1], root + '/shared_tables/' + dataset)
                self.assertEqual(args[args.index('--review-limit') + 1], '10')
            result = subprocess.run(['bash', str(launcher), '2b', 'nr3d', '--predictions', 'old'],
                env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('launcher owns', result.stderr)


if __name__ == '__main__':
    unittest.main()
