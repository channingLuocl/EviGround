# Uni3D-B：同实例物体类别预测基线

本实验只输入单个实例的 XYZ 和 RGB，输出固定 197 类的余弦相似度排名。
使用公开 Uni3D-B 权重，冻结编码器，不使用本数据集标签训练。
这是迁移到 ScanNet/Nr3D 场景实例的零样本实验，不是复现 Uni3D 原论文的 benchmark 数字。

## 固定模型与输入

- 官方源码：`baaivision/Uni3D`，commit `64e03c3c42c196e8cb5ed03857810af9fc9ac39c`。
- 官方权重：`BAAI/Uni3D`，revision `3d8233b76aa350d72f6213ecd2123c2026b42355`，`modelzoo/uni3d-b/model.pt`；完整 SHA256 在 `uni3d_classifier.py` 中固定并校验。
- 点编码器：`eva02_base_patch14_448`，768 维主干，512 个组，每组 64 点，投影至 1024 维。
- 文本塔：`EVA02-E-14-plus` / `laion2b_s9b_b144k`。不能复用 OpenShape 的 ViT-bigG 文本特征。
- 文本权重：`timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k`，revision `0162f50c2742be3c90b3a81e4e8066fc5a90e125`。
- 单模板：`a {} in a scene`，沿用此前物体类别实验；未使用官方 64 模板 ensemble。
- 按此前 OpenShape 方式抽取 10000 点，RGB 除以 255，无色时设 0.4；中心化、单位球归一化。
- **坐标适配**：ScanNet 输入为 Z-up，Uni3D 要求 Y-up；应用 `(x,y,z) -> (x,z,-y)` 的正旋转，保持距离、手性和点/RGB 对应关系。不能把 OpenShape G/14 的 Z-up 约定直接用于 Uni3D。
- 依据：官方训练代码绕 Y 轴旋转；NVIDIA WarpConvNet Uni3D 文档明确所有 Uni3D 尺寸均要求 Y-up（`https://nvlabs.github.io/WarpConvNet/models/uni3d/`）。
- 每个实例使用现有 `instance_seed`，输入采样不依赖执行顺序或续跑状态。

## 评估口径

复用 `run_categories_v82.py` 的实例选择、输入校验、缓存和评分。唯一新增接口允许注入 Uni3D 分类器及来源记录。
130 个场景的 3413 个唯一实例与此前 OpenShape 完全相同，类别来自排除 wall/floor/ceiling 后的 ScanNet200 197 类。
GT 只由评分层读取，不进入编码器、文本特征或候选词表构造。

- True：沿用现有词形/别名归一化匹配，不合并 chair 与 office chair 等子类。
- Exact Top-1：类别字符串完全一致。
- Top-5：前五名中存在归一化后匹配的类别。
- Class Mean：沿用现有代码，对评估集中出现的归一化 GT 类别逐类算准确率后取平均；词表外 GT 也保留。
- Errors：读取、编码或输出异常数，不是分类错误数；词表外 GT 不删除，仍保留在总分母中。

## 服务器运行

独立环境：`/root/autodl-tmp/venvs/uni3d`，复用系统 Torch 2.7.1+cu128。
环境内使用 timm 0.9.7、open_clip_torch 2.20.0、easydict 1.13。
`setup_uni3d_pointnet2.sh` 固定官方 Pointnet2 源码，仅将过时的硬编码 GPU 架构改为当前 5090 的 12.0；随后真实执行 CUDA FPS/gather 检查。

```bash
cd /root/paper/reproductions/SeeGround
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:/root/autodl-tmp/venvs/uni3d/bin:$PATH"

python local/prepare_uni3d_assets.py \
  --asset-dir /root/autodl-tmp/uni3d_baseline_20261018/assets

python local/run_uni3d_categories.py \
  --checkpoint /root/autodl-tmp/uni3d_baseline_20261018/assets/weights/BAAI/Uni3D/modelzoo/uni3d-b/model.pt \
  --text-features /root/autodl-tmp/uni3d_baseline_20261018/assets/text_features_eva02e_197.npz \
  --source-dir /root/autodl-tmp/uni3d_baseline_20261018/assets/source \
  --scene-root /root/autodl-tmp/seeground-v10/scenes \
  --output /root/autodl-tmp/uni3d_baseline_20261018/yup/full_3413
```

先使用单独的 `--limit 1` 输出目录验证真实前向，再使用 `--limit 100` 验证，最后执行全量。
不能对同一输出目录变更代码、模型、输入或运行库；失败实例只有显式 `--retry-errors` 才重试，预测错误不会重试。
完整来源记录在 `config.json`、`runtime.json`、`inputs.json`；逐实例分数在各 `prediction.json`，汇总在 `results.json`。

## 验证

本地 CPU 检查：

```bash
PYTHONPATH=local .venv-v82/bin/python -m unittest \
  local/test_uni3d_classifier.py local/test_v82_pointcloud.py -v
```

真实 GPU 前向还必须验证：官方模型严格加载、197 个有限分数、实例列表与此前 OpenShape 一致、完整结果无运行错误。

最初未转换坐标的 Z-up 轮次保留在实验根目录 `full_3413`，仅供接入诊断，不能作为最终 Uni3D 基线；最终使用 `yup/full_3413`。

## 正式结果（2026-10-18）

| 模型 | 评估对象数 | True正确数 | True准确率 | Exact Top-1 | Top-5 | Class Mean | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Uni3D-B（公开权重，零样本） | 3413 | 549 | 16.09% | 15.88% | 27.83% | 4.49% | 0 |

Exact 正确数 542，Top-5 正确数 950。保留 157 个词表外 GT 实例，与此前实验分母一致。
已核验全部 3413 × 197 个分数有限、所有 Top-1/Top-5 与分数排序一致，并从逐实例结果独立重算统计指标。
正式结果 SHA256：`d10b5b9325cb9c2b0ee9c8adbea413eb67729ffa1491d21a9293fe9f2679cb48`。
服务器完整核验记录：`/root/autodl-tmp/uni3d_baseline_20261018/audit_yup.json`。
