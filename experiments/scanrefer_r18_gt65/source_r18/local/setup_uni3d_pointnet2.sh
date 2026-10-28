#!/usr/bin/env bash
# Build only the pinned FPS/gather extension for this independent Uni3D environment.
set -euo pipefail
experiment=/root/autodl-tmp/uni3d_baseline_20261018
python=/root/autodl-tmp/venvs/uni3d/bin/python
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$(dirname "$python"):$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=4
export TMPDIR="$experiment/tmp"
mkdir -p "$TMPDIR"
"$python" - <<'PY'
import json, tarfile, urllib.request
from pathlib import Path
root = Path('/root/autodl-tmp/uni3d_baseline_20261018')
revision = 'b5ceb6d9ca0467ea34beb81023f96ee82228f626'
archive = root/'pointnet2-source.tar.gz'
if not archive.exists():
    urllib.request.urlretrieve(f'https://codeload.github.com/erikwijmans/Pointnet2_PyTorch/tar.gz/{revision}', archive)
with tarfile.open(archive) as handle:
    handle.extractall(root/'source', filter='data')
directory = root/'source'/f'Pointnet2_PyTorch-{revision}'/'pointnet2_ops_lib'
setup = directory/'setup.py'
original = setup.read_text()
line = 'os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"'
if original.count(line) != 1:
    raise ValueError('Unexpected pointnet2 setup.py: refusing unverified patch')
setup.write_text(original.replace(line, '# Respect TORCH_CUDA_ARCH_LIST set by the build environment.'))
(root/'pointnet2_build.json').write_text(json.dumps({
    'repository': 'https://github.com/erikwijmans/Pointnet2_PyTorch', 'revision': revision,
    'patch': 'Remove hardcoded obsolete CUDA architectures; use TORCH_CUDA_ARCH_LIST=12.0',
    'cuda_home': '/usr/local/cuda-12.8'}, indent=2)+'\n')
print(directory, flush=True)
PY
"$python" -m pip install --no-deps --no-build-isolation "$experiment/source/Pointnet2_PyTorch-b5ceb6d9ca0467ea34beb81023f96ee82228f626/pointnet2_ops_lib"
"$python" - <<'PY'
import torch
from pointnet2_ops import pointnet2_utils
points = torch.randn(1, 1000, 3, device='cuda')
indices = pointnet2_utils.furthest_point_sample(points, 32)
sampled = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), indices)
torch.cuda.synchronize()
assert indices.shape == (1, 32) and sampled.shape == (1, 3, 32)
assert torch.isfinite(sampled).all()
print('pointnet2 CUDA FPS/gather verified', torch.__version__, torch.cuda.get_device_name())
PY
