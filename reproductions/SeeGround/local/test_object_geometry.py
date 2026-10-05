import copy
import unittest
from object_geometry import build_geometry, box_measurements, describe_role


def obj(oid, name, x, y, z, w, length, height):
    return {'bbox_id':oid,'target':name,'bbox_3d':[x,y,z,w,length,height]}


class ObjectGeometryTests(unittest.TestCase):
    def setUp(self):
        self.objects=[obj(1,'cabinet',0,0,.5,1,1,1),
                      obj(2,'cabinet',2,0,1.5,1,1,3),
                      obj(3,'cabinet',4,0,.25,1,1,.5),
                      obj(4,'refrigerator',-2,0,2,1,1,2)]

    def test_role_descriptors_bind_same_category_separately(self):
        target=describe_role('lower kitchen cabinet',self.objects)
        reference=describe_role('the tall kitchen cabinet',self.objects)
        self.assertEqual(target['ids'],[1,2,3])
        self.assertEqual(reference['ids'],[1,2,3])
        self.assertEqual([d['word'] for d in target['descriptors']],['lower'])
        self.assertEqual([d['word'] for d in reference['descriptors']],['tall'])

    def test_box_size_and_vertical_rankings_are_explicit(self):
        result=box_measurements(self.objects[:3])
        self.assertEqual(result['rankings']['height_m']['maximum_ids'],[2])
        self.assertEqual(result['rankings']['height_m']['minimum_ids'],[3])
        self.assertEqual(result['rankings']['volume_m3']['maximum_ids'],[2])
        self.assertEqual(result['measurements'][1]['bottom_z_m'],0.)
        self.assertEqual(result['measurements'][1]['top_z_m'],3.)

    def test_next_to_geometry_keeps_target_reference_pair(self):
        query='lower kitchen cabinet next to the tall kitchen cabinet'
        parsed={'Target':['kitchen cabinet'],'Anchor':['kitchen cabinet']}
        result=build_geometry(query,parsed,self.objects)
        self.assertEqual(result['relation_quote'],'next to')
        self.assertEqual(len(result['binding_pairs']),6)
        self.assertNotIn('nearest',result['relation_quote'])
        self.assertTrue(all(p['target_id']!=p['reference_id'] for p in result['binding_pairs']))

    def test_above_below_reports_bounds_without_certifying_contact(self):
        query='cabinet below the refrigerator'
        parsed={'Target':['cabinet'],'Anchor':['refrigerator']}
        result=build_geometry(query,parsed,self.objects)
        pair=next(p for p in result['binding_pairs'] if p['target_id']==1 and p['reference_id']==4)
        self.assertLess(pair['center_z_delta_m'],0)
        self.assertEqual(pair['vertical_overlap_m'],0.)
        self.assertIn('vertical_overlap_m',pair)
        self.assertIn('AABB volume',result['limitations'][0])

    def test_geometry_does_not_mutate_inputs(self):
        original=copy.deepcopy(self.objects)
        build_geometry('cabinet next to refrigerator',{'Target':['cabinet'],'Anchor':['refrigerator']},self.objects)
        self.assertEqual(self.objects,original)


if __name__=='__main__':unittest.main()
