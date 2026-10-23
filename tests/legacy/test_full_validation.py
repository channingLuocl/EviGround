"""Full validation selection matches released queries rather than model outcomes."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from seeground.experiments.select_full_validation import select


class FullSelectionTests(unittest.TestCase):
    def fixture(self, root):
        scene = 'scene0001_00'
        csv_path, split, olt, baseline, size = [root / n for n in
            ('nr3d.csv', 'val.txt', 'olt.zip', 'baseline.zip', 'size.json')]
        split.write_text(scene)
        row = {'scan_id': scene, 'mentions_target_class': 'True', 'correct_guess': 'False',
               'stimulus_id': scene + '-chair-2-1-2', 'target_id': '1', 'utterance': 'the left chair',
               'instance_type': 'chair', 'tokens': "['the', 'left', 'chair']"}
        with csv_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader()
            for change in ({}, {'correct_guess': 'True'}, {'mentions_target_class': 'False'},
                           {'scan_id': 'train_scene'}):
                writer.writerow({**row, **change})
        with ZipFile(olt, 'w') as archive:
            for kind in ('gt', 'pred'):
                archive.writestr(f'seeground_object_lookup_table/nr3d/{kind}/{scene}.json',
                                json.dumps([{'bbox_id': 1}, {'bbox_id': 2}]))
        with ZipFile(baseline, 'w') as archive:
            archive.writestr(f'seeground_results/nr3d/val/model/pred/{scene}.json', json.dumps(
                [{'query': row['utterance'], 'gt_id': 1}] * 2))
        size.write_text(json.dumps({'rows': [{'csv_row': 3, 'scene': scene,
            'query': row['utterance'], 'target_id': 1, 'stimulus_id': row['stimulus_id'],
            'stimulus_candidate_ids': [1, 2], 'stage': 'original_300'}]}))
        return csv_path, split, olt, baseline, size

    def test_author_protocol_keeps_incorrect_guesses_and_duplicate_queries(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = select(*self.fixture(Path(td)))
            self.assertEqual([r['csv_row'] for r in manifest['rows']], [2, 3])
            self.assertTrue(manifest['author_queries_match'])
            self.assertEqual(manifest['summary']['overlap_size_300'], 1)
            self.assertEqual(manifest['rows'][1]['stage'], 'original_300')
            self.assertTrue(all(r['easy'] and r['view_dep'] for r in manifest['rows']))

    def test_rejects_missing_author_query_instead_of_silently_changing_denominator(self):
        with tempfile.TemporaryDirectory() as td:
            paths = self.fixture(Path(td))
            with ZipFile(paths[3], 'w') as archive:
                archive.writestr('seeground_results/nr3d/val/model/pred/scene0001_00.json',
                                json.dumps([{'query': 'the left chair', 'gt_id': 1}]))
            with self.assertRaisesRegex(ValueError, 'differs'):
                select(*paths)


if __name__ == '__main__':
    unittest.main()
