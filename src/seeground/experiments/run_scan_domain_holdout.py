"""Use isolated current-Torch CUDA ops, verify old features, then cache holdout."""
from seeground.paths import source_file
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import torch
from torch.nn import functional as F

# Keep the active Torch already imported; borrow only missing Python dependencies.
root = Path(sys.argv[sys.argv.index('--output')+1])
sys.path.insert(0, str(root / 'cuda_ops'))
sys.path.append('/root/paper/reproductions/SeeGround/.venv-v82/lib/python3.12/site-packages')
from seeground.categories.category_common_v82 import file_hash, instance_seed, load_cloud
from seeground.classifiers.pointcloud_classifier_v82 import load_pointnext, prepare_points
from seeground.experiments.run_rgb_experiment import check_frozen
from seeground.evaluation.run_storage import write_json

check_frozen(root, 'MODEL_SELECTION_FROZEN.json')
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_num_threads(4)
model, metadata = load_pointnext('/root/paper/reproductions/SeeGround/weights/pnext_cls.pth')
model = model.cuda().eval()
source = Path('/root/autodl-tmp/seeground-v10')
manifest = json.loads((source / 'manifests/eval.json').read_text())
with np.load(source / 'features/eval/features.npz') as archive:
    expected = archive['global_features'][:, 0]
records = []
for index in (0, len(expected)//2, len(expected)-1):
    row = manifest['samples'][index]
    xyz, _, ids = load_cloud(row['cloud'])
    seed = instance_seed(42, row['scene'], row['bbox_id'])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    points = prepare_points(xyz[ids == row['bbox_id']], None, 'pointnext', seed)
    with torch.inference_mode():
        actual = F.normalize(model(torch.from_numpy(points[None]).cuda()).float(), dim=-1)[0].cpu().numpy()
    np.testing.assert_allclose(actual, expected[index], atol=1e-5, rtol=1e-4)
    records.append({'index': index, 'scene': row['scene'], 'bbox_id': row['bbox_id'],
                    'maximum_absolute_difference': float(np.max(np.abs(actual-expected[index])))})
import pointnet2_batch_cuda
write_json(root / 'pointnext_runtime.json', {'torch': str(torch.__version__), 'bridge': records,
           'cuda_extension_sha256': file_hash(pointnet2_batch_cuda.__file__),
           'cuda_extension': pointnet2_batch_cuda.__file__, 'checkpoint': metadata,
           'wrapper_sha256': file_hash(Path(__file__))})
print(json.dumps({'stage': 'old_feature_bridge_passed', 'records': records}), flush=True)
del model
torch.cuda.empty_cache()
sys.argv = [str(source_file('cache_scan_domain.py')), 'eval', *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name='__main__')
seal = json.loads((root / 'EVAL_FEATURES_FROZEN.json').read_text())
seal['files']['pointnext_runtime.json'] = file_hash(root / 'pointnext_runtime.json')
write_json(root / 'EVAL_FEATURES_FROZEN.json', seal)
