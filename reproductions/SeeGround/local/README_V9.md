# v9：使用冻结的 v8.3 类别假设完成指代定位

分类器固定为 `object-category-v8.3-zero-shot-r1`，不再训练或修改。v9 从本地读取其预测和几何信息，再运行 v7.4 的描述解析、证据诊断、渲染与多模态定位流程。推理不读取作者的 `objects.json` 类别，不向模型提供 GT 类别或查询答案。

## 类别门槛

Top-5 是假设上限，不是五个同时成立的类别。每个实例仍只有一个 ID。类别必须同时满足：

- 排名在选定的 Top-1 或 Top-5 内；
- `score >= min_category_score`，默认 `0.20`；
- `top1_score - score <= max_category_gap`，默认 `0.03`。

这些分数来自全局余弦相似度与正向局部支持，不是概率；`0.20` 不表示 20% 置信度。两个默认阈值是尚未校准的试验设置，没有用验证集答案调优。后续如需选择阈值，应在独立开发集上选定，再固定参数比较。

例如五个分数为 `0.36, 0.34, 0.28, 0.25, 0.18`，默认只保留前两项。随后用描述中的目标或参照类别检索这些假设，并用视觉、空间证据复核。标签表中的 `target/category` 仅展示冻结 Top-1，不覆盖其余假设。模型输入保留类别、分数、排名、分差和不确定状态，明确同一物体的类别假设互斥。最终输出一个物体 ID 或 `null`，不会复制成五个物体。

未通过的类别不会经“全场候选兜底”重新进入目标候选；没有匹配目标时，记录 `category_gate_abstention` 并返回不确定。参照类别缺失也保留为证据缺口。原有类别别名和检索规则沿用，例如 screen/monitor/tv 的既有映射；因此检索候选覆盖率不等于严格类别准确率。

## 本地数据

完整下载并校验的目录：

```text
local_outputs/v8_3_zero_shot_full/
  FROZEN.json
  predictions.json
  config.json
  ...逐物体预测、配置、评估与诊断文件
local_outputs/archives/v8_3_zero_shot_full_20261016.tar.gz
```

共有 130 个场景、3,413 个物体。归档 SHA256：

```text
18d6fb03e07b0c2f6aeeeb204c58121cc1cf9ea6b174ca5f43b18a4b058b8174
```

`FROZEN.json` 记录本地文件哈希；v9 校验预测和配置哈希、实例覆盖、每个输入点云哈希。分类预测来自点云推理，场景实例分割仍使用已准备的实例 ID。服务器可以保持关闭。`local_outputs` 位于外接盘，运行时需要连接该盘。

## 运行

从 `reproductions/SeeGround` 执行。使用本地 `.venv` 和 `.env` 配置的模型 API，无需本地 GPU 或 PointNeXt 环境。以下两组命令使用同样的 100 条描述作为配对试验：

```bash
local/run_local_v9.sh --category-top-k 1 --limit 100 \
  --output local_outputs/v9_top1_100

local/run_local_v9.sh --category-top-k 5 --limit 100 \
  --min-category-score 0.20 --max-category-gap 0.03 \
  --output local_outputs/v9_gated_top5_100

PYTHONPATH=local .venv/bin/python local/compare_grounding_v9.py \
  local_outputs/v9_top1_100 local_outputs/v9_gated_top5_100
```

只准备输入、校验数据，不调用模型 API：

```bash
local/run_local_v9.sh --category-top-k 1 --prepare-only \
  --output local_outputs/v9_top1
local/run_local_v9.sh --category-top-k 5 --prepare-only \
  --output local_outputs/v9_gated_top5
```

去掉 `--prepare-only` 可运行同一目录的全量 7,805 条描述。准备时与运行时的 `--limit`、阈值和其他参数必须一致。可用 `SEEGROUND_WORKERS`、`SEEGROUND_API_CONCURRENCY` 设置并发，默认 10 个本地任务、8 个并发 API 请求。每个本地任务的数值计算线程保持为 1，避免线程数成倍增加。

终端输出复用 v7.4 的 `BatchProgress`：每完成一条输出 `{"predicted_id": ..., "correct": ...}`，进度条显示完成数量、百分比、耗时、ETA 和累计 `True` 准确率。续跑时已缓存结果计入初始进度。完整统计写入 `results.json`。

## 评估与复现

每个批次保存 `manifest.json`、独立 `category_inputs`、逐查询日志/模型输入/预测/评估，以及 `results.json`。类别输入逐项记录保留或拒绝原因；预测记录实际匹配到的目标和参照类别假设及其排名。运行配置固定源预测、场景、代码、模型和参数的身份，变更后必须换输出目录。

本入口不导入历史答对缓存，不按 GT 正误重跑。重复执行同一配置会保留所有已完成结果，包括答错、弃答和错误。查询的 `target_id`、候选答案和 GT 标签不进入推理子进程；GT 仅在父进程计算 ID 命中和 IoU，结构实例的标注类别在推理前统一清空为 `object`。

比较脚本要求两个批次使用相同描述、源预测、输入场景、代码、模型和阈值，且完整跑完同一批次。只允许 Top-K 与对应类别表不同。报告净增/净损样本、准确率、覆盖率、IoU、目标候选覆盖率、调用次数和耗时；不能把分类 Top-5 命中率当成最终定位准确率。对有错误的查询，候选覆盖率按未召回计并另外报告可观察查询数量。只有实际配对运行后才能判断 Top-5 是否更好。

## 检查

```bash
PYTHONPATH=local .venv/bin/python -m unittest local/test_v9.py -q
```

测试覆盖阈值边界、低分弃答、第二类别用于目标/参照检索、模型保留互斥假设、无 GT 推理输入、相同配置续跑及配对比较校验。模型调用使用替身；数据准备不等于实际 API 效果评估。
