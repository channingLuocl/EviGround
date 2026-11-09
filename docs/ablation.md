# 消融实验指南

## 当前支持范围

当前代码保留论文 Nr3D 消融所用的 E0–E5 条件。`prepare_ablation_subset.py` 按场景及 Easy/Hard × 视角依赖分层冻结子集；`run_nr3d_ablation.py` 固定要求 1,500 条描述、130 个场景，拆为两个交错分片，并保存源码哈希、协议与逐条输出。它们不能直接接收 ScanRefer 清单。

| 条件 | supplementary inspect | conditional measure | 类别 Top-k | alternative-category check |
| --- | --- | --- | --- | --- |
| E0 | 开 | 开 | 5 | 开 |
| E1 | 关 | 开 | 5 | 关（不适用） |
| E2 | 开 | 关 | 5 | 开 |
| E3 | 开 | 开 | 1 | 关（不适用） |
| E4 | 开 | 开 | 5 | 关 |
| E5 | 关 | 关 | 5 | 关（不适用） |

推理开关为 `EVIGROUND_INSPECT_ENABLED`、`EVIGROUND_MEASURE_ENABLED`、`EVIGROUND_CATEGORY_INSPECTION_REQUIRED`，只接受 `0` 或 `1`；类别由 `--category-top-k` 控制。关闭 inspect 时，类别检查同步关闭；这些设置进入运行身份，不能混用同一输出目录。

## Nr3D 子集复现

先按 README 安装依赖，并使用新的实验目录。下列场景、分类导出和 API 地址必须替换为实际输入；不要覆盖已有实验快照。

```bash
export EVIGROUND_ABLATION_STUDY="$PWD/experiments/nr3d_ablation_new"
export EVIGROUND_ABLATION_SCENES=/absolute/path/to/nr3d_prepared_scenes
export EVIGROUND_ABLATION_PREDICTIONS=/absolute/path/to/verified_classifier_export
export EVIGROUND_ABLATION_MODEL=your_model_name
export EVIGROUND_ABLATION_BASE_URL=https://your-provider.example/v1

python scripts/prepare_ablation_subset.py \
  --manifest data/queries/nr3d_val.json \
  --output "$EVIGROUND_ABLATION_STUDY/nr3d_subset_1500.json" \
  --size 1500 --seed 20261104

PYTHON_DOTENV_DISABLED=1 PYTHONPATH=src \
  python scripts/run_nr3d_ablation.py --workers-per-key 3
```

批处理启动后从标准输入读取一行 JSON：一个共享 API key 或两个分片各自的 API key 组成的数组。密钥不写入命令参数、源码或结果文件。脚本会先进行模型/视觉/结构化输出预检和技术试运行，再开始 E0–E5 正式调用；本命令会产生实际 API 请求。不要在日志或可共享的终端记录中粘贴密钥。

输入场景与分类导出路径可通过上述环境变量配置。API 地址优先取 `EVIGROUND_ABLATION_BASE_URL`，其次取 `EVIGROUND_BASE_URL`，均未设置时使用 `https://api.openai.com/v1`；不会自动使用历史私有转发地址。

完整运行后，用 `python scripts/summarize_nr3d_ablation.py --help` 查看审计汇总参数。`summarize_mixed_nr3d_ablation.py` 仅用于如实记录不同条件使用不同模型的历史比较；跨模型差异不能单独解释为模块贡献。错误、弃答和缺失均不能从固定样本分母中剔除。

## 后续 ScanRefer 子集消融

本轮整理没有启动新的 ScanRefer 模型调用。现有 `scripts/scanrefer.py` 可生成官方格式的验证清单并按 Unique / Multiple / Overall 汇总 Acc@0.25 与 Acc@0.5；主批处理 `eviground --manifest ...` 已支持 ScanRefer 的 IoU 评估。

迁移 E0–E5 时，还需要实现 ScanRefer 专用或通用的子集抽样、六条件调度与汇总：

1. 首次模型调用前固定样本数量、种子和场景/Unique–Multiple 分层，保留来源哈希；不得按模型结果挑选样本。原 `prepare_scanrefer_splits.py` 是历史诊断划分，不能冒充新冻结的消融子集。
2. 所有条件使用同一份子集、模型、预测实例及分类导出、去重策略、初始视图和工具预算，分别保存源码及配置身份。
3. 明确采用 Mask3D 预测实例或 GT 实例。预测实例 ID 与 GT 目标 ID 不可直接比较；GT-proposal 实验须单独标注，不能混入 Mask3D 协议。
4. 用预测框与目标 GT 框的 IoU 汇总两个阈值及三个分组，保留错误/弃答在分母中，并给出相对 E0 的配对差异。旧 Nr3D 汇总器的 `correct` 统计不能替代这些指标。

外部数据、权重、完整实验输出与论文文件不随当前源码仓库分发，论文中的既有数字也不代表本次代码整理后的重新评测。
