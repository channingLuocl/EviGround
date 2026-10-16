# v8.3：无新增标注、免训练的类别判断

根据用户要求，v8.3默认改为**不使用本任务类别标签或部件标注、不训练新分类头、不更新模型参数**的推理方案。允许使用公开预训练点云模型和公共类别词表；复用现有PointNeXt权重与197类CLIP文本特征。无需部件文本特征、训练manifest或head.pt。

之前实现的监督部件状态头不适用于这个目标，已退出默认路径。[历史说明](docs/history/README_V8_3_SUPERVISED.md)仅用于保留可复核的实验代码；显式选择supervised-experimental才能调用。

## 概念边界

- **无本任务标注**：类别标签、部件状态、作者现成预测不进入分类器，也不用于训练新参数。评估GT仅用于事后评分。
- **免训练**：当前版本不做自监督学习或测试时参数更新，直接冻结预训练模型推理。“无监督适配”允许无标签训练，但此处暂不需要这一步。
- **严格未见类零样本**：还需证实目标类别、形状、场景没有通过不合适的监督进入预训练。当前pnext_cls.pth训练来源尚未完成审计，不能仅凭“不重新训练”就宣称严格零样本；zero-shot是当前工程入口名，结果同时记录这个限制。
- 预训练模型使用过图文或类别监督，不等于“整个系统从零无监督训练”。
- 输入仍假定已含实例ID／掩码。本版本是实例类别判断；掩码可能来自GT，本版没有实现无监督实例分割，不能把整条链路称作完全无标注。

## 当前算法

1. 从每个实例提取整体点云特征，与固定公共类别文本计算余弦分数。
2. 固定8个空间裁块，对整体和同样的局部点集分别做两次随机采样编码；第二次仅改变采样，不改变裁块几何。这是重采样一致性，不是恢复原扫描视角。
3. 整体分数取两次平均。局部只有在两次Top-1一致且有正类别间隔时才提供支持，以特征一致性和类别间隔加权；至少两个不同裁块支持同一类别才生效。重复裁块、几乎整物体的裁块不重复计票。
4. 在整体平均分的前5个类别中加入固定小权重的正支持。无支持不扣分，不预测“某部件不存在”。分类器Top-5重排不涉及后续定位器的Top-5接入。

固定设置：8裁块、两次采样偏移104729、局部权重0.1；无需用GT选择这些设置。它们是尚未验证的研究默认值，不声称是最优参数。较多一致裁块也可能一起认错，不能当成独立证据或校准置信度。

此实现使用物体级PointNeXt编码局部裁块，没有额外预训练的语义部件分割器。输出local_evidence指“局部对类别的支持”，**不等于靠背／轮子等命名部件检测或存在性判断**。多采样／局部融合已有文献基础，本实现不据此宣称新颖性或提分。

## 服务器运行

沿用已配置成功的v8.2环境，不增加依赖，不需要新增模型文件或API key。在SeeGround根目录执行：

```bash
SEEGROUND_V83_PYTHON="$PWD/.venv-v82/bin/python" local/run_v8_3_categories.sh --limit 1 --output local_outputs/v8_3_zero_shot_smoke
SEEGROUND_V83_PYTHON="$PWD/.venv-v82/bin/python" local/run_v8_3_categories.sh --limit 100 --output local_outputs/v8_3_zero_shot_100
```

默认读取与v8.2相同的场景列表、数据根目录、冻结权重及 `weights/v82_clip_text_features.npz`。若数据放在其他目录，增加 `--scene-root /data/scenes`。建议先单例确认数值与速度，再扩大评估。

不加载GPU模型的选择检查：

```bash
SEEGROUND_V83_PYTHON="$PWD/.venv-v82/bin/python" local/run_v8_3_categories.sh --limit 1 --prepare-only --output local_outputs/v8_3_zero_shot_prepare
```

纯点云推理不需要任何类别文件：

```bash
SEEGROUND_V83_PYTHON="$PWD/.venv-v82/bin/python" local/run_v8_3_categories.sh --cloud /data/my_scene/scene.npz --bbox-id 12 --output local_outputs/v8_3_cloud
```

冻结全局对照，与v8.2同一预处理及单实例编码路径：

```bash
SEEGROUND_V83_PYTHON="$PWD/.venv-v82/bin/python" local/run_v8_3_categories.sh --method zero-shot-global --limit 100 --output local_outputs/v8_3_global_100
```

默认zero-shot每实例执行18次编码，全局对照仅1次。额外预算必须报告；不能假定速度沿用v8.2。使用同一选择、词表和评分规则比较准确率、类别平均、Top-5，以及改对／改错数量，才能判断局部支持是否有价值。

## 输出与来源记录

预测文件包含197类scores、Top-1/Top-5、原单次整体baseline_top1／baseline_top5、整体重采样共识Top-1、局部支持及裁块一致性诊断。分数不是校准概率。不会生成present／absent／unknown部件状态，也不会读取部件标注文件。

独立evaluation.json和results.json保存评分；prediction.json无GT。配置记录learning_regime、代码／模型／文本／输入哈希，以及预训练来源未完整审计的限制。成功预测可复用；--retry-errors仅重试执行错误并归档旧尝试，分类错误不自动重跑。代码、输入、选择或权重变化需新输出目录；FROZEN目录受保护。

已查看的1000个实例属于开发证据；不能用于选权重后再声称独立测试。真正未见类别／场景评估还需要预训练来源审计及未参与方法选择的数据划分。之前的68.70%和84.50%属于v8.2，不是v8.3实测。

## 验证

```bash
PYTHONPATH=local python -m unittest local/test_v83_zero_shot.py local/test_v83_evidence.py local/test_v82_pointcloud.py -v
```

上述22项测试已通过。新测试覆盖正支持重排、无支持不产生负证据、重复裁块、候选范围、重采样几何不变，以及默认入口无需head／类别文件的预测与缓存隔离。另已在未安装torch的环境验证默认入口的--help及--prepare-only。CPU测试对冻结CUDA编码器使用测试替身；尚未执行服务器真实点云前向，也没有v8.3准确率结果。
