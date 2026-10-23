# 历史监督实验：不符合当前无任务标注目标

以下保留早期实现说明。当前默认使用免训练路径，见 local/README_V8_3.md。运行历史head需显式添加 --method supervised-experimental；不属于无监督或严格零样本。

# v8.3：残缺点云的局部语义证据分类器

v8.3 是可训练的研究版本。输入仍是实例点云 XYZ 和实例 ID，输出197类的分数、Top-1/Top-5及部件状态估计；作者现成预测标签不进入模型。假设已存在实例掩码、Z轴朝上。本版不实现实例分割或下游Top-5定位接入，也不改动v8.2。

**现在没有训练好的v8.3权重，不能直接用v8.2权重冒充v8.3结果。** 需先准备独立训练／开发数据及部件状态标注，再缓存特征、训练小模型。现有1000例已作为开发诊断，不应当作未查看的独立测试集。只有类别标签时可先训练global/attention/voting对照；evidence模式会检查每个部件的三种状态均有训练监督，不满足时停止。

## 模型实现与边界

1. 冻结原有PointNeXt及512维语义投影。整体特征使用v8.2预处理；通过8个最远点锚点建立近邻裁块，用同一编码器分别提取局部特征，并保留相对位置、范围和占比。RGB和米制尺度仍未使用，以控制输入预算。
2. 公共部件文本经匹配版本CLIP编码；局部／整体特征与部件文本相似度，以及局部几何布局，输入小型状态头，估计present、absent、unknown。
3. evidence分类头用 `全局类别logits + W × (p_present - p_absent)`。unknown在此局部分支直接贡献为0；W由训练数据学习，不手写chair/stool等测试规则。分类梯度和显式状态监督共同训练状态头。

这些状态是**未校准的统计估计**，不是实测物理可见性。当前点云没有原始相机位姿／射线；同样的残缺几何可能来自不同完整类别，模型无法保证判断真实不存在。softmax输出不是已经验证过的校准概率。

局部编码器目前是对物体级预训练模型输入裁块，**不是已经预训练好的语义部件分割器**。这是首个可训练基线实现；局部域差与准确率收益尚未验证，不宣称实现了论文级创新或一定超过68.70%。研究依据见[文献核查报告](/Users/luocl/Desktop/seeground/research/20261016_partial_classifier_audit/文献核查与方向判断.md)。

## 环境

沿用服务器已经配置成功的 `.venv-v82`：CUDA版PyTorch、PointNeXt算子及v8.2依赖。没有新增Python包依赖。特征提取需要NVIDIA GPU；缓存之后的小模型训练在CPU运行，测试也可在CPU运行。

```bash
cd /path/to/SeeGround
V83_PYTHON="$PWD/.venv-v82/bin/python"
"$V83_PYTHON" local/train_categories_v83.py --help
```

以下命令中的数据路径需指向真实的训练、开发数据，不是当前验证集的默认别名。

## 1. 准备训练／开发manifest

每份manifest的格式如下，路径相对manifest所在目录解析（也可绝对路径）。这是**格式示例，类别与部件状态必须根据实际对象核实**：

```json
{
  "format": "v8.3-training-manifest-v1",
  "split": "train",
  "annotation_protocol": "说明类别和部件标签来源、部件定义、如何核实不存在及证据不足",
  "samples": [
    {
      "scene": "training_scene_A",
      "cloud": "training_scene_A/scene.npz",
      "bbox_id": 12,
      "category": "chair",
      "part_states": {"backrest": "present"},
      "part_masks": "training_scene_A/object_12_parts.npz"
    }
  ]
}
```

开发集使用 `split: "dev"`，其余格式相同。训练／开发按场景分开；程序同时检查场景名和云文件SHA256，ScanNet同一房间的不同扫描（sceneXXXX_00/_01）也视作重叠。独立测试还需与这两组分开；预训练数据重叠尚未审计，所以不声称严格未见类零样本。

已有训练场景的 `scene.npz`、`objects_gt.json` 时，可生成只有类别监督的初始manifest，**不会读取objects.json的作者类别，也不会根据类别猜部件状态**：

```bash
"$V83_PYTHON" local/train_categories_v83.py make-manifest \
  --scenes /data/train_scenes.json --scene-root /data/scenes \
  --split train --output /data/v83/train.json
"$V83_PYTHON" local/train_categories_v83.py make-manifest \
  --scenes /data/dev_scenes.json --scene-root /data/scenes \
  --split dev --output /data/v83/dev.json
```

`train_scenes.json`等格式为 `{"rows":[{"scene":"scene0000_00"}]}`。这里读取的GT是明确指定的**训练／开发监督**；不允许从测试集自动学习。词表外训练标签被排除并计数，测试仍保留词表外实例在分母里。

部件标注规则：

- `present`：当前输入中有该部件的可识别观测支持。
- `absent`：相关部件定义适用，且有完整、穷尽或人工核实的依据支持不存在；仅没找到或没标注不成立。
- `unknown`：已核实当前观测证据不足，无法支持存在／不存在。
- `part_states`没有写的部件：**未标注**，训练loss忽略；不同于显式unknown，也不同于absent。

默认公共部件表在 `local/parts_v83.json`，包含16种部件，不含实例答案或类别规则。概念验证建议先复制成更小的公共部件表（例如仅backrest），完成可信标注后逐步扩大；保持训练、开发、文本特征中的部件顺序一致。每个部件至少出现三种状态只是程序的最低条件，不代表标注量已经足够。

## 2. 公共部件文本特征

在可访问Hugging Face、装有v8.2依赖的机器执行；无需GPU或VLM API：

```bash
"$V83_PYTHON" local/train_categories_v83.py prepare-text \
  --parts local/parts_v83.json --output weights/v83_clip_part_features.npz
```

服务器不能访问Hugging Face时，把这个文件传过去。它只包含公共部件文本向量。原有 `weights/v82_clip_text_features.npz` 继续提供197类文本；加载时检查语义空间、顺序与版本。

## 3. 缓存冻结特征

```bash
"$V83_PYTHON" local/train_categories_v83.py cache \
  --manifest /data/v83/train.json --parts local/parts_v83.json \
  --output local_outputs/v8_3_cache_train
"$V83_PYTHON" local/train_categories_v83.py cache \
  --manifest /data/v83/dev.json --parts local/parts_v83.json \
  --output local_outputs/v8_3_cache_dev
```

默认不自动裁剪增强。每个对象执行1次整体和8次局部编码，速度不能沿用v8.2单次编码的预期；每个实例／视图有固定采样种子。缓存绑定代码、冻结权重、文本、原云、标注文件哈希及标注协议。缓存及训练目录必须是新目录，不覆盖旧结果；当前缓存提取中断后需使用新目录重新提取。

如有**当前输入点上的部件支持mask**，训练缓存可加 `--crop-views 2`。`part_masks`为NPZ：

```python
# N为该实例在scene.npz中按原始顺序取出的点数，P为公共部件数量。
# masks为bool[N,P]；每个True须对应已标注present部件的真实观测点。
np.savez("object_12_parts.npz", parts=np.array(part_names), masks=masks)
```

程序检查每个present部件有观测支持。随机平面裁剪只改变观测：留下足够支持点的present仍为present，支持丢失则变unknown；原absent在额外裁剪视图中也保守变unknown；未标注仍忽略。**裁剪不产生absent监督**，不能用随机删除部件伪造不存在。开发集禁止裁剪增强。

## 4. 训练小模型及对照

```bash
"$V83_PYTHON" local/train_categories_v83.py train \
  --mode evidence \
  --train-cache local_outputs/v8_3_cache_train/features.npz \
  --dev-cache local_outputs/v8_3_cache_dev/features.npz \
  --part-text-features weights/v83_clip_part_features.npz \
  --epochs 50 --output local_outputs/v8_3_evidence_head
```

生成 `head.pt`、配置和逐epoch训练／开发日志；仅按开发集类别交叉熵选择checkpoint。日志还记录有状态标注的开发样本的状态准确率和三类Brier分数。没有状态标注时这些指标为空，不伪报0。训练不调用VLM，不连接服务器服务，也不在线下载其他模型。

使用相同缓存分别把mode改成以下值、输出到不同目录：

| mode | 对照内容 |
|---|---|
| global | 全局语义＋零初始化线性适配头，区别于完全冻结的v8.2 |
| attention | 局部attention聚合＋普通线性融合 |
| voting | 每通道正precision加权局部特征，借鉴PSV思路，非完整PSV复现 |
| evidence | 命名部件状态＋unknown中性、present/absent有符号证据约束 |

attention/voting与evidence使用**同一个状态头和同等状态标注**；两个对照也可自由融合完整状态向量，因此不能把收益简单归因于evidence独享标注。各模式参数数目保存到配置中，并非严格等参数；仍需后续预算匹配实验。无部件标注也可训练前两个局部对照，但不能将其与有额外监督的evidence直接当公平消融。

## 5. 独立点云推理／分类评估

训练完成后执行：

```bash
SEEGROUND_V83_PYTHON="$V83_PYTHON" local/run_v8_3_categories.sh \
  --head-checkpoint local_outputs/v8_3_evidence_head/head.pt \
  --cloud /data/heldout_scene/scene.npz --bbox-id 12 \
  --output local_outputs/v8_3_cloud_smoke

SEEGROUND_V83_PYTHON="$V83_PYTHON" local/run_v8_3_categories.sh \
  --head-checkpoint local_outputs/v8_3_evidence_head/head.pt \
  --manifest /data/test_scenes.json --scene-root /data/scenes \
  --limit 1000 --output local_outputs/v8_3_categories_1000
```

`--cloud`不需要任何类别文件；模型接口只有点云和随机种子。评估模式沿用v8.2的实例选择和匹配规则，GT在预测后评分；输出Top-1、严格Top-1、Top-5、类别平均及词表外数量。预测文件无GT，评分写到独立文件。Top-5只是本分类器输出，尚未接入定位器。

场景与训练／开发集重叠时默认拒绝；若明确要做训练诊断可加 `--allow-development-evaluation`，结果标为 `development_overlap`，不得作为独立测试。纯点云模式用不同文件名或改变场景身份可能绕过简单来源检查，程序无法证明任意几何变换后没有数据重复，仍需人工管理数据来源。

`--prepare-only`校验已训练head、选择和哈希而不加载CUDA编码器（仍需CPU PyTorch）；重复运行复用成功预测。`--retry-errors`只重试执行错误，保留旧尝试，不重试分类错误。权重／输入／代码或环境变化时要求新的输出目录；FROZEN目录受保护。推理时不需要部件标注文件或部件文本NPZ，相关文本向量已经存入训练checkpoint。

## 验证状态

CPU测试包含真实PyTorch前向、反向和四组小模型优化，以及缓存→训练→checkpoint重载→无标签推理的集成检查；其中冻结CUDA编码器用明确的测试替身替代。测试几何与标签均为合成夹具，不是准确率实验。

2026-10-16本机验证：Python3.9、PyTorch2.8.0 CPU，v8.3新增9项及v8.2原有7项共16项测试通过；脚本语法、Python编译及两个CLI的帮助入口检查通过。未改动v8.2源文件或冻结预测结果。

```bash
PYTHONPATH=local python -m unittest local/test_v83_evidence.py local/test_v82_pointcloud.py -v
```

尚未执行服务器真实PointNeXt多裁块前向、真实部件监督训练或独立准确率评估。不能把单测通过解释为v8.3提分。第一轮应先缓存少量真实训练实例，检查状态标注、局部特征数值与速度，再训练／比较四组；无需重跑或覆盖v8.2冻结结果。
