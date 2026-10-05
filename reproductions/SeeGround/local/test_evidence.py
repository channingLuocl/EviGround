import unittest
from evidence import canonical_category, match_categories, distance_evidence


class EvidenceTests(unittest.TestCase):
    def test_aliases_and_no_partial_word_match(self):
        objects=[{'bbox_id':1,'target':'tv'},{'bbox_id':2,'target':'doorframe'},
                 {'bbox_id':3,'target':'door'},{'bbox_id':4,'target':'trash can'}]
        for name in ['TV','television','screen']:
            self.assertEqual([o['bbox_id'] for o in match_categories([name],objects)],[1])
        self.assertEqual(canonical_category('trashcan'),'trash can')
        self.assertEqual([o['bbox_id'] for o in match_categories(['front doors'],objects)],[3])
        self.assertEqual(match_categories(['blackboard'],objects),[])

    def test_distances_preserve_each_reference(self):
        targets=[{'bbox_id':1,'bbox_3d':[0,0,0,2,2,2]},
                 {'bbox_id':2,'bbox_3d':[3,4,0,2,2,2]}]
        evidence=distance_evidence(targets,[targets[0]])
        self.assertEqual(len(evidence['pairs']),1)
        self.assertEqual(evidence['pairs'][0]['center_distance_m'],5.)
        self.assertAlmostEqual(evidence['pairs'][0]['aabb_gap_m'],5**.5,places=4)
        self.assertEqual(distance_evidence(targets,[])['pairs'],[])


if __name__=='__main__':unittest.main()
