"""Build a real ScanNet scene from raw files and author-provided OLT. No API calls."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from seeground.rendering.run import read_cloud, ROOT, render

def prepare(downloads, scene, output):
    output.mkdir(parents=True,exist_ok=True)
    xyz,rgb=read_cloud(downloads/f'{scene}_vh_clean_2.ply')
    seg=np.array(json.loads((downloads/f'{scene}_vh_clean_2.0.010000.segs.json').read_text())['segIndices'])
    if len(seg)!=len(xyz):raise ValueError('点云和分割点数不一致')
    groups=json.loads((downloads/f'{scene}.aggregation.json').read_text())['segGroups']
    lines=(downloads/f'{scene}.txt').read_text().splitlines()
    line=next(x for x in lines if x.startswith('axisAlignment'))
    matrix=np.array([float(x) for x in line.split('=',1)[1].split()],dtype=np.float32).reshape(4,4)
    xyz=(np.column_stack([xyz,np.ones(len(xyz))])@matrix.T)[:,:3]
    ids=np.full(len(xyz),-1,dtype=int)
    for g in groups:
        mask=np.isin(seg,g['segments'])
        if (ids[mask]>=0).any():raise ValueError('实例分割区域重叠')
        ids[mask]=int(g['objectId'])
    scene_instances=[]
    for g in groups:
        object_id=int(g['objectId'])
        pts=xyz[ids==object_id]
        if not len(pts):
            continue
        scene_instances.append({'bbox_id':object_id,
                                'target':str(g.get('label','unknown')),
                                'bbox_3d':np.r_[(pts.min(0)+pts.max(0))/2,
                                                pts.max(0)-pts.min(0)].tolist()})
    with ZipFile(downloads/'olt.zip') as z:
        gt=json.loads(z.read(f'seeground_object_lookup_table/nr3d/gt/{scene}.json'))
        pred=json.loads(z.read(f'seeground_object_lookup_table/nr3d/pred/{scene}.json'))
    errors=[]
    for obj in gt:
        pts=xyz[ids==obj['bbox_id']]
        if not len(pts):raise ValueError('查找表编号在点云实例中不存在')
        bounds=np.r_[(pts.min(0)+pts.max(0))/2,pts.max(0)-pts.min(0)]
        errors.append(float(np.max(np.abs(bounds-np.asarray(obj['bbox_3d'])))))
    if max(errors)>.002:raise ValueError(f'查找表与点云坐标不一致：最大误差 {max(errors)} m')
    gt_by_id={o['bbox_id']:o for o in gt}
    for obj in pred:
        if not np.allclose(obj['bbox_3d'],gt_by_id[obj['bbox_id']]['bbox_3d'],atol=.002):raise ValueError('预测类别表与真值表几何不一致')
    # GT labels/queries are evaluation-only, never added to the model prompt.
    (output/'objects.json').write_text(json.dumps(pred,ensure_ascii=False,indent=2))
    (output/'objects_gt.json').write_text(json.dumps(gt,ensure_ascii=False,indent=2))
    (output/'scene_instances.json').write_text(json.dumps(scene_instances,ensure_ascii=False,indent=2))
    np.savez_compressed(output/'scene.npz',xyz=xyz,rgb=rgb,ids=ids)
    kept=xyz[:,2]<xyz[:,2].max()-.3
    overview=render(xyz[kept],rgb[kept],output/'overview.png',ids=ids[kept])
    provenance={'scene':scene,'points':len(xyz),'objects':len(pred),'max_bbox_coordinate_error_m':max(errors),
                'source_olt':'https://github.com/user-attachments/files/18056532/seeground_object_lookup_table.zip',
                'source_scannet':f'http://kaldir.vc.in.tum.de/scannet/v2/scans/{scene}/',
                'geometry':'GT instances with author-provided predicted category labels, as in supplied Nr3D OLT',
                'structural_instances':'ScanNet aggregation labels and boxes; reference evidence only',
                'renderer':'local CPU adaptation, not exact PyTorch3D implementation','overview':overview,
                'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in downloads.glob(scene+'*')}}
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps({k:provenance[k] for k in ['scene','points','objects','max_bbox_coordinate_error_m']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',default='scene0011_00');p.add_argument('--downloads',type=Path,default=ROOT/'local_outputs/downloads');p.add_argument('--output',type=Path)
    a=p.parse_args();prepare(a.downloads,a.scene,a.output or ROOT/'local_outputs'/a.scene)
