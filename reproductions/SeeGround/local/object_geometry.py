"""Auditable box-size, vertical-position and separate target/reference evidence."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from evidence import canonical_category, match_retrieval_categories

DESCRIPTORS=(r'\b(smallest|largest|biggest|tallest|shortest|lowest|highest|'
             r'larger|smaller|bigger|taller|shorter|lower|upper|tall|short|'
             r'small|large|skinny|narrow|thin|wide|thick)\b')
CONNECTORS=r'next to|beside|near|by|above|below|under|beneath|away from|nearest(?: to)?|closest to|farthest from|furthest(?: away)? from'
METRICS=('volume_m3','footprint_m2','height_m','center_z_m','bottom_z_m','top_z_m')


def category_matches(name, objects):
    # ScanNet often calls kitchen cabinets simply cabinet. Keep this local to evidence.
    normalized=canonical_category(name)
    # Descriptors belong to the linguistic role, not to the ScanNet category.
    normalized=' '.join(word for word in normalized.split()
                        if word not in {'smallest','largest','biggest','tallest','shortest','lowest','highest',
                                        'larger','smaller','bigger','taller','shorter','lower','upper',
                                        'tall','short','small','large',
                                        'skinny','narrow','thin','wide','thick'})
    if normalized=='kitchen cabinet':normalized='cabinet'
    return match_retrieval_categories([normalized],objects)


def describe_role(phrase, objects):
    descriptors=[{'word':m.group().lower(),'phrase':m.group()} for m in re.finditer(DESCRIPTORS,phrase,re.I)]
    name=re.sub(DESCRIPTORS,'',phrase,flags=re.I).strip()
    name=re.sub(r'^(?:the|a|an)\s+','',name,flags=re.I).strip()
    matches=category_matches(name,objects)
    return {'phrase':phrase,'category':name,'descriptors':descriptors,
            'ids':[o['bbox_id'] for o in matches]}


def box_measurements(objects):
    rows=[]
    for obj in objects:
        x,y,z,w,length,h=obj['bbox_3d']
        rows.append({'id':obj['bbox_id'],'volume_m3':w*length*h,'footprint_m2':w*length,
                     'height_m':h,'center_z_m':z,'bottom_z_m':z-h/2,'top_z_m':z+h/2})
    rankings={}
    for metric in METRICS:
        ordered=sorted(rows,key=lambda r:(r[metric],r['id']))
        rankings[metric]={'ascending_ids':[r['id'] for r in ordered],
                          'minimum_ids':[r['id'] for r in ordered if abs(r[metric]-ordered[0][metric])<=1e-6],
                          'maximum_ids':[r['id'] for r in ordered if abs(r[metric]-ordered[-1][metric])<=1e-6]} if ordered else {}
    return {'measurements':rows,'rankings':rankings}


def point_cloud_measurements(objects, xyz, instance_ids):
    """Measure instance AABBs directly from the prepared scene point cloud.

    Missing instance points are reported as unknown instead of inheriting the
    table box, because a fabricated measurement would look like evidence.
    """
    rows = []
    if xyz is None or instance_ids is None:
        point_ids = None
    else:
        point_ids = np.asarray(instance_ids)
        points = np.asarray(xyz)
        if len(points) != len(point_ids):
            raise ValueError('scene xyz and instance ids must have equal length')
    for obj in objects:
        object_id = obj['bbox_id']
        object_points = None if point_ids is None else points[point_ids == object_id]
        row = {'id': object_id, 'point_count': 0,
               'source': 'instance_point_cloud_aabb', 'status': 'unknown'}
        if object_points is None or not len(object_points):
            row['unknown_reason'] = 'no instance points in scene.npz'
            rows.append(row)
            continue
        lower = np.min(object_points, axis=0).astype(float)
        upper = np.max(object_points, axis=0).astype(float)
        extent = upper - lower
        horizontal_short = min(float(extent[0]), float(extent[1]))
        horizontal_long = max(float(extent[0]), float(extent[1]))
        row.update({
            'point_count': int(len(object_points)),
            'min_xyz': lower.tolist(), 'max_xyz': upper.tolist(),
            'extent_x_m': float(extent[0]), 'extent_y_m': float(extent[1]),
            'extent_z_m': float(extent[2]),
            'volume_m3': float(np.prod(extent)),
            'footprint_m2': float(extent[0] * extent[1]),
            'height_m': float(extent[2]),
            'horizontal_short_extent_m': horizontal_short,
            'horizontal_long_extent_m': horizontal_long,
            'horizontal_aspect_ratio': (horizontal_long / horizontal_short
                                        if horizontal_short > 1e-9 else None),
            'height_to_horizontal_long_ratio': (float(extent[2]) / horizontal_long
                                                 if horizontal_long > 1e-9 else None),
            'status': 'known',
        })
        rows.append(row)
    return rows


def build_geometry(query, parsed, objects):
    evidence={'policy':'soft box evidence only; no candidate exclusion or unique reference certification',
              'units':'meters, square meters, cubic meters; world Z points up',
              'limitations':['AABB volume/footprint are alternative size proxies, not object volume.',
                             'Tall/lower/near/away from have no calibrated threshold; not scene-wide extremes.',
                             'Category pools describe the full supplied table; linguistic subsets remain unbound.',
                             'Box distances and vertical overlap do not prove support, contact or appearance.'],
              'category_pools':[],'roles':{},'binding_pairs':[],'missing':[]}
    names=[]
    for field in ('Target','Anchor'):
        value=parsed.get(field,[])
        names.extend([value] if isinstance(value,str) else value)
    for name in dict.fromkeys(names):
        pool=category_matches(name,objects)
        evidence['category_pools'].append({'category':name,'ids':[o['bbox_id'] for o in pool],
                                          **box_measurements(pool)})
    # Whole simple clauses only: preserve distinct noun phrases even for identical categories.
    clause=re.sub(r'^(?:find|choose|select)\s+','',query.strip(),flags=re.I).rstrip('.!?')
    match=re.fullmatch(r'(?P<target>[A-Za-z ]+?)\s+(?P<relation>'+CONNECTORS+r')\s+(?P<reference>[A-Za-z ]+)',clause,re.I)
    phrases={'target':match['target'].strip(),'reference':match['reference'].strip()} if match else {'target':clause}
    for role,phrase in phrases.items():
        info=describe_role(phrase,objects)
        evidence['roles'][role]={**info,**box_measurements([o for o in objects if o['bbox_id'] in info['ids']])}
        if not info['ids']:evidence['missing'].append('unbound '+role+' phrase: '+phrase)
    if not match:
        if parsed.get('Anchor'):evidence['missing'].append('reference phrase requires a simple explicit clause')
        return evidence
    evidence['relation_quote']=match['relation']
    by_id={o['bbox_id']:o for o in objects}
    for target_id in evidence['roles']['target']['ids']:
        for reference_id in evidence['roles']['reference']['ids']:
            if target_id==reference_id:continue
            a=np.asarray(by_id[target_id]['bbox_3d']);b=np.asarray(by_id[reference_id]['bbox_3d'])
            gap=np.maximum(np.abs(a[:3]-b[:3])-(a[3:]+b[3:])/2,0)
            overlap=np.maximum(np.minimum(a[:3]+a[3:]/2,b[:3]+b[3:]/2)
                               -np.maximum(a[:3]-a[3:]/2,b[:3]-b[3:]/2),0)
            evidence['binding_pairs'].append({'target_id':target_id,'reference_id':reference_id,
                'center_distance_m':float(np.linalg.norm(a[:3]-b[:3])),
                'aabb_gap_m':float(np.linalg.norm(gap)),'horizontal_aabb_gap_m':float(np.linalg.norm(gap[:2])),
                'center_z_delta_m':float(a[2]-b[2]),'vertical_overlap_m':float(overlap[2]),
                'horizontal_overlap_m2':float(overlap[0]*overlap[1]),
                'target_bottom_minus_reference_top_m':float(a[2]-a[5]/2-b[2]-b[5]/2),
                'reference_bottom_minus_target_top_m':float(b[2]-b[5]/2-a[2]-a[5]/2)})
    if not evidence['binding_pairs']:evidence['missing'].append('no distinct target/reference pair')
    return evidence


def main():
    from run import load_objects
    from real_scene import object_table_path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir',type=Path,required=True);parser.add_argument('--query',required=True)
    parser.add_argument('--target',required=True);parser.add_argument('--anchor',default='')
    parser.add_argument('--object-table',choices=('pred','gt'),default='pred')
    args=parser.parse_args()
    objects=load_objects(object_table_path(args.scene_dir,args.object_table))
    evidence=build_geometry(args.query,{'Target':[args.target],'Anchor':[args.anchor] if args.anchor else []},objects)
    print(json.dumps({'network_request':False,'object_table':args.object_table,'geometry_conditions':evidence},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
