"""EviGround configuration migration preserves credentials and explicit settings."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from eviground import config
from eviground.cli.entrypoint import configure_r18


class ConfigTests(unittest.TestCase):
    def test_legacy_environment_is_accepted(self):
        with patch.dict(os.environ, {'PYTHON_DOTENV_DISABLED': '1',
                        'SEEGROUND_API_KEY': 'unit-test-key', 'SEEGROUND_MODEL': 'legacy-model'}, clear=True):
            config.load_environment()
            self.assertEqual(os.environ['EVIGROUND_API_KEY'], 'unit-test-key')
            self.assertEqual(os.environ['EVIGROUND_MODEL'], 'legacy-model')

    def test_new_settings_take_precedence_including_explicit_empty_values(self):
        with patch.dict(os.environ, {'PYTHON_DOTENV_DISABLED': '1',
                        'SEEGROUND_MODEL': 'old', 'EVIGROUND_MODEL': 'new',
                        'SEEGROUND_JEV_MODEL': 'old', 'EVIGROUND_JEV_MODEL': ''}, clear=True):
            config.load_environment()
            self.assertEqual(os.environ['EVIGROUND_MODEL'], 'new')
            self.assertEqual(os.environ['EVIGROUND_JEV_MODEL'], '')

    def test_existing_dotenv_file_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.env').write_text('SEEGROUND_API_KEY=unit-test-key\nSEEGROUND_MODEL=old\n')
            with patch.object(config, 'ROOT', root), patch.dict(os.environ, {}, clear=True):
                config.load_environment()
                self.assertEqual(os.environ['EVIGROUND_API_KEY'], 'unit-test-key')
                self.assertEqual(os.environ['EVIGROUND_MODEL'], 'old')

    def test_mixed_dotenv_respects_new_names_and_process_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.env').write_text('SEEGROUND_MODEL=old\nEVIGROUND_MODEL=file\n')
            with patch.object(config, 'ROOT', root), patch.dict(os.environ, {}, clear=True):
                config.load_environment()
                self.assertEqual(os.environ['EVIGROUND_MODEL'], 'file')
            with patch.object(config, 'ROOT', root), \
                    patch.dict(os.environ, {'EVIGROUND_MODEL': 'process'}, clear=True):
                config.load_environment()
                self.assertEqual(os.environ['EVIGROUND_MODEL'], 'process')

    def test_default_policy_uses_migrated_configuration(self):
        with patch.dict(os.environ, {'PYTHON_DOTENV_DISABLED': '1',
                        'SEEGROUND_API_KEY': 'unit-test-key', 'SEEGROUND_MODEL': 'configured-model'}, clear=True):
            configure_r18()
            self.assertEqual(os.environ['EVIGROUND_MODEL'], 'configured-model')
            self.assertEqual(os.environ['EVIGROUND_API_KEY'], 'unit-test-key')
            self.assertEqual(os.environ['EVIGROUND_R8_GROUNDING_REVISION'], 'r18')
            self.assertEqual(os.environ['EVIGROUND_V9'], '1')
            self.assertEqual(os.environ['EVIGROUND_ENABLE_THINKING'], '1')

    def test_repository_root_supports_both_names_with_new_name_preferred(self):
        for overrides, expected in (({'SEEGROUND_ROOT': '/tmp/legacy-root'}, '/tmp/legacy-root'),
                                    ({'SEEGROUND_ROOT': '/tmp/legacy-root',
                                      'EVIGROUND_ROOT': '/tmp/eviground-root'}, '/tmp/eviground-root')):
            with self.subTest(overrides=overrides):
                environment = {k: v for k, v in os.environ.items()
                               if k not in ('SEEGROUND_ROOT', 'EVIGROUND_ROOT')}
                environment.update(overrides)
                result = subprocess.run([sys.executable, '-c',
                    'from eviground.paths import ROOT; print(ROOT)'],
                    env=environment, check=True, capture_output=True, text=True)
                self.assertEqual(result.stdout.strip(), str(Path(expected).resolve()))

    def test_vendored_pointnext_retains_its_required_compatibility_switch(self):
        from eviground.classifiers import pointcloud_classifier_v82 as encoder
        state = {'pc_projection': np.zeros((256, 512))}
        torch = SimpleNamespace(load=Mock(return_value=state))
        model = Mock()
        models = SimpleNamespace(PcdClassifier=Mock(return_value=model))
        with patch.dict(sys.modules, {'torch': torch, 'models.pcd_classifier': models}), \
                patch.object(encoder, 'file_hash', return_value=encoder.POINTNEXT_SHA256), \
                patch.object(sys, 'path', list(sys.path)), patch.dict(os.environ, {}, clear=True):
            encoder.load_pointnext('mock-checkpoint')
            self.assertEqual(os.environ['EVIGROUND_POINTNEXT_INFERENCE_ONLY'], '1')
            self.assertEqual(os.environ['SEEGROUND_POINTNEXT_INFERENCE_ONLY'], '1')
            model.load_state_dict.assert_called_once_with(state, strict=True)


if __name__ == '__main__':
    unittest.main()
