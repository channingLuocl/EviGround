"""Offline branch/resume checks; never invoke inference or the real outputs."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import retry_null_extra as extra


class ExtraNullTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        source, output = root / 'source', root / 'extra'
        source.mkdir()
        output.mkdir()
        (source / 'terminal_stage').mkdir()
        (root / 'versions').mkdir()
        extra.save(root / 'versions/r18-evidence-v1.json', {'max_retry_rounds': 3})
        rows = [{'csv_row': i, 'scene': 'scene0001_00', 'target_id': i, 'query': 'chair'}
                for i in range(3)]
        baseline = [{**r, 'predicted_id': p, 'iou_3d': v}
                    for r, p, v in zip(rows, (0, 99, None), (1., 0., 0.))]
        manifest = root / 'manifest.json'
        extra.save(manifest, {'dataset': 'scanrefer', 'rows': rows})
        extra.save(source / 'results.json', {'results': baseline})
        extra.save(source / 'history.json', {'rounds': [2, 3]})
        extra.save(source / 'protocol.json', {'max_total_retry_rounds': 3})
        extra.save(source / 'terminal_stage/FINAL_EVALUATION.json', {'accounting': {}})
        hashes = {name: extra.digest(source / name) for name in
                  ('results.json', 'history.json', 'protocol.json')}
        hashes['FINAL_EVALUATION.json'] = extra.digest(source / 'terminal_stage/FINAL_EVALUATION.json')
        extra.save(source / 'FROZEN.json', {'rounds_used': 3, 'maximum_rounds': 3,
                                          'artifact_sha256': hashes})
        for name in extra.ENTRIES:
            (root / name).write_text('offline fixture only\n')
        return root, source, output, manifest, baseline

    def context(self, root, source, output):
        return patch.multiple(extra, ROOT=root, SOURCE=source, OUTPUT=output,
                              ROUND=output / 'round004')

    def test_selects_only_null_and_preserves_frozen_nonnull_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root, source, output, manifest, baseline = self.fixture(directory)
            before = {p.name: extra.digest(p) for p in source.glob('*.json')}
            with self.context(root, source, output):
                extra.prepare(manifest)
                protocol, copied = extra.checked(manifest)
            self.assertEqual(protocol['selected_ids'], [2])
            self.assertEqual(protocol['max_additional_rounds'], 1)
            self.assertEqual(copied, baseline)
            self.assertEqual(before, {p.name: extra.digest(p) for p in source.glob('*.json')})
            self.assertEqual(extra.merge(baseline, [{**baseline[2], 'predicted_id': 2}], [2])[:2], baseline[:2])

    def test_interrupted_trajectory_cannot_be_silently_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            root, source, output, manifest, baseline = self.fixture(directory)
            (output / 'round004/q00002').mkdir(parents=True)
            with self.context(root, source, output), patch.object(extra.subprocess, 'run') as launch:
                with self.assertRaisesRegex(ValueError, 'audited recovery'):
                    extra.run_extra({'selected_ids': [2]}, baseline)
                launch.assert_not_called()

    def test_preflight_and_completed_round_never_launch_another_round(self):
        with tempfile.TemporaryDirectory() as directory:
            root, source, output, manifest, baseline = self.fixture(directory)
            with self.context(root, source, output):
                extra.prepare(manifest)
                extra.save(output / 'results.json', {'results': baseline})
                extra.save(output / 'completion.json', {'results_sha256': extra.digest(output / 'results.json')})
                for action in ('preflight', 'run'):
                    with patch.object(extra, 'verify', return_value=({}, manifest)), \
                         patch.object(extra, 'validate_runtime_settings'), \
                         patch.object(extra, 'run_extra') as launch, \
                         patch.object(extra, 'report_extra'), \
                         patch.dict(extra.os.environ, {}, clear=True), \
                         patch('sys.argv', ['retry_null_extra.py', action]), \
                         redirect_stdout(io.StringIO()):
                        extra.main()
                        launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
