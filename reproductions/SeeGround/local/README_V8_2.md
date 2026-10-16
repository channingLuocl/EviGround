# v8.2：从点云独立预测物体类别

v8.2 不调用 VLM API，也不使用作者预测类别。推理只读取 `scene.npz` 中的 `xyz`、可选 `rgb` 和 `ids`，然后使用冻结的语义对齐点云模型与固定 ScanNet 类别词表计算相似度。评估入口读取 `objects.json` 时只取实例 ID，忽略其中类别；`objects_gt.json` 的类别仅用于预测后的评分。

输入假定已包含实例 ID/掩码、Z 朝上、坐标以米为单位。这一版本实现实例分类，不包含从整场景自动生成实例掩码的分割模型。使用预训练网络权重，不需要我们重新训练；权重的训练数据来源尚未完成审计，因此暂不宣称严格零样本结果。

## 推荐的服务器安装

下面的推理命令须在登录 NVIDIA GPU 服务器后执行。终端提示符若仍显示 `luocldeMacBook-Air`，说明还在 Mac 本机。Mac 中的 `.venv-v82` 只是准备检查环境，没有安装 PyTorch；即使补装 Mac 版 PyTorch，当前 CUDA 算子也不能在 Apple GPU/MPS 上执行。请在服务器重新创建环境，不要复制 Mac 的虚拟环境目录。

在 NVIDIA GPU Linux 服务器上使用独立环境，建议 Python 3.10 或 3.11。先按服务器 CUDA 版本安装 CUDA 版 PyTorch，再安装 PointNeXt 的推理依赖和算子。编译需要 `nvcc`、C++ 编译器和与 PyTorch 匹配的 CUDA toolkit；只有驱动或 CPU 版 PyTorch 不够。

```bash
cd /path/to/SeeGround
python -m venv .venv-v82
source .venv-v82/bin/activate
pip install --upgrade pip
# 在这个新环境内，按 https://pytorch.org/get-started/locally/ 安装匹配的 CUDA 版 torch，再继续。
pip install -r local/requirements-v82.txt
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
nvcc --version
python -m pip install --no-build-isolation ./models/pointnext/PointNeXt/openpoints/cpp/pointnet2_batch
python -c 'import pointnet2_batch_cuda; print("PointNeXt CUDA extension imported")'
python local/fetch_v82_weights.py
```

`fetch_v82_weights.py` 会校验 SeeGround 官方文件 SHA-256：
`d6551942dda8ddf4af1d840ca8a83ee60f3a7535a618bb5b541e145b95663f6b`。

## 先做不加载模型的选择检查

```bash
SEEGROUND_V82_PYTHON="$PWD/.venv-v82/bin/python" \
local/run_v8_2_categories.sh --limit 10 --prepare-only \
  --output local_outputs/v8_2_pointnext_prepare
```

## PointNeXt 小规模评估

先用 `--limit 1` 和单独的输出目录验证服务器环境，再跑 100 个实例。以下命令在服务器上的 SeeGround 根目录执行；不使用本机 `.venv`，也不需要 API key 或并发参数。

```bash
SEEGROUND_V82_PYTHON="$PWD/.venv-v82/bin/python" \
local/run_v8_2_categories.sh --backend pointnext --limit 1 \
  --output local_outputs/v8_2_pointnext_smoke

SEEGROUND_V82_PYTHON="$PWD/.venv-v82/bin/python" \
local/run_v8_2_categories.sh --backend pointnext --limit 100 \
  --output local_outputs/v8_2_pointnext_100
```

输出每个实例的 `prediction.json`，包括全部类别 `scores`、top-5 和随机采样种子；`results.json` 同时报告严格 top-1、词法等价 top-1、top-5 和类别平均准确率。错误会被保存，重复运行不会重算成功预测；使用 `--retry-errors` 才会归档并重试错误。

评估模式需要将 `local/validation/nr3d_val_full.json` 以及对应场景的 `local_outputs/<scene>/scene.npz`、`objects.json`、`objects_gt.json` 一起放到服务器。`local_outputs` 在本机是外接盘软链接，传输时应复制实际目录内容；服务器也可用 `--scene-root /data/scenes` 指定真实数据目录。

控制台仍然输出下面的格式，并显示累计进度、耗时和准确率：

```json
{"bbox_id": 30, "target": "computer tower", "predicted_object": "computer tower", "correct": true}
```

这是格式示例，不是本轮实测结果。`correct` 只做单复数及少量同义词匹配；不把 `office chair` 与 `chair` 合并。所有模型均使用同一公共 197 类词表；词表外 GT 会在汇总中单独计数。`--limit 0` 评估所选场景的所有实例，不是 7,805 条指代表达。

结果保存：每实例 `prediction.json`（无 GT）、`evaluation.json`（评估）；根目录 `predictions.json`、`results.json`、`config.json`、`selection.json`、`inputs.json`、`runtime.json`。结果写入采用临时文件替换，重试会保留旧错误到 `attempts/`。更换代码、权重、运行环境或已记录输入时须使用新输出目录，避免混合实验结果。

## 纯点云单场景命令

这条命令不需要 GT 文件：

```bash
SEEGROUND_V82_PYTHON="$PWD/.venv-v82/bin/python" \
local/run_v8_2_categories.sh --backend pointnext \
  --cloud local_outputs/scene0131_00/scene.npz --bbox-id 30 \
  --output local_outputs/v8_2_pointnext_scene0131
```

## OpenShape 对照

OpenShape 对照入口已实现，目前未完成 GPU 实测。它需要与服务器 CUDA、PyTorch 匹配的 DGL 环境，不能把它的文本向量和 PointNeXt 混用；建议使用另一个独立环境。先按 DGL 官方安装说明安装匹配的 wheel，再运行：

```bash
pip install -r local/requirements-v82-openshape.txt
SEEGROUND_V82_PYTHON="$PWD/.venv-v82/bin/python" \
local/run_v8_2_categories.sh --backend openshape --limit 100 \
  --output local_outputs/v8_2_openshape_100
```

OpenShape G/14 输入为 10,000 个 XYZRGB 点，单位球归一化，Z 朝上，RGB 为 0 到 1；PointNeXt 输入为 2,048 个 XYZ+height 点。两条路线使用各自配套的文本空间。

## 服务器无法访问 Hugging Face 时

在能访问 Hugging Face 且安装了 PyTorch/transformers 的机器上，用 CPU 生成固定公共类别的文本特征：

```bash
python local/prepare_v82_text_features.py --output weights/v82_clip_text_features.npz
```

将生成的 NPZ 传到服务器同一路径，并为运行命令添加 `--text-features weights/v82_clip_text_features.npz`。此文件只包含 197 个类别的 CLIP 文本向量，不含任何实例或 GT 信息。加载时检查词表、提示模板、模型版本、形状及有限值，运行记录保存文件 SHA-256 和文本模型来源。PointNeXt 仍在服务器上从实例点云独立计算特征；这是同一文本分支的离线缓存，不是另一个分类器。

## 结果解释

PointNeXt/OpenShape 的公开权重是网络参数，不是作者预测标签。当前结果只能作为同一实例子集上的独立基线；不能把论文的 ModelNet40、ScanObjectNN 或 17 类 ScanNet 结果直接当成我们 197 类数据的预期准确率。OccTIP 是处理残缺扫描的后续改进方向，但本轮没有确认可直接下载的官方预训练权重。

## 本机验证边界

本机为 Apple Silicon，不能运行此 CUDA 推理路径。已核验 PointNeXt 文件 SHA-256，并在 CPU 上检查全部 83 项 state dict 严格匹配、256×512 语义投影完整；结构检查临时屏蔽了 CUDA 扩展导入，没有执行模型前向。GPU 算子编译、真实前向和准确率须由服务器试跑确认。

离线测试：

```bash
PYTHONPATH=local python -m unittest local/test_v82_pointcloud.py -v
```
