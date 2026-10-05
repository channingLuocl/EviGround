import unittest
from evidence import (canonical_category, match_categories, distance_evidence,
                      match_retrieval_categories, distance_rankings, propose_category_checks)
from evidence import validate_distance_relations


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

    def test_retrieval_strips_appearance_but_not_original_query(self):
        objects=[{'bbox_id':1,'target':'tv'},{'bbox_id':2,'target':'door'}]
        self.assertEqual(match_retrieval_categories(['large black screen'],objects),[objects[0]])
        self.assertEqual(match_retrieval_categories(['front doors'],objects),[objects[1]])

    def test_rankings_keep_references_separate_and_ties(self):
        targets=[{'bbox_id':1,'target':'chair','bbox_3d':[1,0,0,1,1,1]},
                 {'bbox_id':2,'target':'chair','bbox_3d':[-1,0,0,1,1,1]},
                 {'bbox_id':3,'target':'chair','bbox_3d':[4,0,0,1,1,1]}]
        anchors=[{'bbox_id':10,'target':'table','bbox_3d':[0,0,0,1,1,1]},
                 {'bbox_id':11,'target':'table','bbox_3d':[5,0,0,1,1,1]}]
        query='The chair nearest the table.'
        parsed={'Relations':[{'target_category':'chair','anchor_category':'table',
                             'relation':'nearest','phrase':query}]}
        ranks=distance_rankings(query,parsed,targets,anchors)['rankings']
        self.assertEqual(ranks[0]['best_ids'],[1,2])
        self.assertEqual(ranks[1]['best_ids'],[3])
        parsed['Relations'][0].update(relation='farthest',phrase='The chair farthest from the table.')
        ranks=distance_rankings('The chair farthest from the table.',parsed,targets,anchors)['rankings']
        self.assertEqual(ranks[0]['best_ids'],[3])
        self.assertEqual(ranks[1]['best_ids'],[2])

    def test_unstated_or_negated_superlative_never_ranked(self):
        chair={'bbox_id':1,'target':'chair','bbox_3d':[0,0,0,1,1,1]}
        table={'bbox_id':2,'target':'table','bbox_3d':[1,0,0,1,1,1]}
        for query,phrase in [('chair by the table','chair by the table'),
                             ('chair by the table','chair nearest the table'),
                             ('chair not nearest the table','not nearest the table')]:
            parsed={'Relations':[{'target_category':'chair','anchor_category':'table',
                                 'relation':'nearest','phrase':phrase}]}
            result=distance_rankings(query,parsed,[chair],[table])
            self.assertEqual(result['rankings'],[])
            self.assertTrue(result['ignored'])
            validated=validate_distance_relations(query,parsed)
            self.assertEqual(validated['Relations'][0]['relation'],'other')
            self.assertEqual(parsed['Relations'][0]['relation'],'nearest')

    def test_proposal_keeps_labels_then_checks_neighbors_with_budget(self):
        objects=[{'bbox_id':10,'target':'tv','bbox_3d':[0,0,0,1,1,1]},
                 {'bbox_id':50,'target':'water cooler','bbox_3d':[1,0,0,1,1,1]},
                 {'bbox_id':60,'target':'chair','bbox_3d':[3,0,0,1,1,1]},
                 {'bbox_id':70,'target':'trash can','bbox_3d':[6,0,0,1,1,1]}]
        proposed=propose_category_checks(objects,[objects[3]],[objects[0]],2)
        self.assertEqual([o['bbox_id'] for o in proposed],[70,50])
        self.assertEqual(objects[1]['target'],'water cooler')


if __name__=='__main__':unittest.main()
