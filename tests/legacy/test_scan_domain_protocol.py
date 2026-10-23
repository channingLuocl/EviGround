"""Room quarantine and training membership checks for the revised protocol."""
import unittest
from seeground.experiments.scan_domain_protocol import plan


def manifest(scenes, rows):
    return {'scenes': scenes, 'samples': rows, 'sources': [{'scene': s} for s in scenes],
            'excluded_training_instances': [], 'excluded_structural_instances': []}


class ProtocolTests(unittest.TestCase):
    def fixture(self):
        return {'train': manifest(['scene0001_00'], [{'scene': 'scene0001_00', 'bbox_id': 1, 'label': 0}]),
                'dev': manifest(['scene0002_00'], [{'scene': 'scene0002_00', 'bbox_id': 0, 'label': -1},
                                                  {'scene': 'scene0002_00', 'bbox_id': 1, 'label': 0}]),
                'eval': manifest(['scene0003_00'], [{'scene': 'scene0003_00', 'bbox_id': 4, 'label': -1}])}

    def test_rescans_quarantined_and_new_rooms_reserved(self):
        train, dev, protocol = plan(['scene0001_00', 'scene0002_00'],
            ['scene0003_00', 'scene0003_01', 'scene0004_00', 'scene0004_01'], self.fixture())
        self.assertEqual(protocol['holdout_rooms'], ['scene0004'])
        self.assertEqual(protocol['holdout_scenes'], ['scene0004_00', 'scene0004_01'])
        self.assertEqual(protocol['quarantined_rescans'], ['scene0003_01'])
        self.assertEqual(len(train['samples']), 2)
        self.assertEqual(len(train['excluded_training_instances']), 1)
        self.assertEqual(dev['samples'][0]['label'], -1)
        self.assertEqual(dev['split'], 'dev')

    def test_reject_official_room_overlap(self):
        with self.assertRaises(ValueError):
            plan(['scene0001_00', 'scene0002_00'], ['scene0001_01', 'scene0003_00'], self.fixture())

    def test_reject_no_unused_rooms(self):
        with self.assertRaises(ValueError):
            plan(['scene0001_00', 'scene0002_00'], ['scene0003_00', 'scene0003_01'], self.fixture())


if __name__ == '__main__':
    unittest.main()
