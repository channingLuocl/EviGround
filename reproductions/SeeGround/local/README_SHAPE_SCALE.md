# PointNeXt 分类修正与真实尺度实验

独立入口，复用冻结 PointNeXt 缓存和此前三种子 global 头；不修改 `category_v10`、旧模型或旧结果。研究设计见仓库 `research/20261019_classifier_experiment_design/实验设计.md`。

## 本轮实验

| ID | 输出 | 作用 |
|---|---|---|
| E0 | 原始语义分数 | 2262/3413 的固定基线 |
| E1 | 基线＋校准后的旧全局残差 | 减少有害修正 |
| E2 | 基线＋尺寸残差 | 检验米制尺度信息 |
| E2-R | 基线＋比例残差 | 控制绝对尺度的作用 |
| E3 | 基线＋旧全局残差＋尺寸残差 | 检验互补性 |

新增尺寸头/比例头各训练 seed42/43/44、40 epoch，AdamW lr0.001。两视图 CE＋0.1 KL，按内部开发集 CE 选 checkpoint；三种子平均 dev True 选择共享融合系数。全部六个头训练完才选系数，冻结选择后才读取 eval 特征。

尺寸来自缓存包围盒后三维，中心坐标不进入推理；训练集拟合标准化。两视图使用相同的完整已观测实例尺寸，形状分支沿用原有两视图增强。绝对跨度不是完整物体的真尺寸。

## 运行

在 `SeeGround` 根目录执行。`python` 应为已有可用 Torch/NumPy 环境，CUDA 只训练小头，不需要重新编译点云算子。

```bash
PYTHONPATH=local python -m unittest test_shape_scale -v

PYTHONPATH=local python -u local/run_shape_scale_experiment.py \
  --feature-root /root/autodl-tmp/seeground-v10 \
  --prior /root/autodl-tmp/pointnext_adaptation_20261019 \
  --output /root/autodl-tmp/shape_scale_NEW_RUN \
  --design /path/to/experiment_config.json --device cuda
```

启动器顺序执行 `train-select`、`evaluate`、独立审计。不要在已有输出目录重复启动。失败时保留目录，先查看 `status.json` 和 `pipeline.log`；不自动覆盖部分训练。阶段入口也可分别调用 `shape_scale.py train-select/evaluate`，参数相同；已有完整小头可校验后复用，部分模型须换输出目录。

源码放在服务器隔离目录时，将该目录及原项目 `local/` 同时加入 `PYTHONPATH`；依赖哈希读取原项目的实际文件。

## 产物和解释

- `models/seed*/size|ratio/`：最优权重、训练集标准化参数、40 epoch 历史及完成哈希。
- `dev/`、`dev_grid.json`、`selection.json`：开发分数、全部融合候选和共享选择。
- `SELECTION_FROZEN.json`：模型/选择/代码配置的冻结哈希，标记未读取评估特征。
- `eval/`：三个种子 × 五种方法的完整分数、无 GT 预测和单独评分。
- `results.json`、`table.tsv`、`RESULTS_FROZEN.json`：统计、纠正/改错、均值±标准差及房间配对 bootstrap 区间。
- `audit.json`：独立重算开发集网格、最终指标、归一化参数、选模历史，以及抽查原始点云的米制跨度。

Errors 不包括词表外目标；它们正常计错并留在分母。Class Mean 包含所有归一化目标类别，另报词表内指标。三种子波动与房间抽样区间是不同的不确定性。该 eval 是此前多次查看的验证集，不能当作新的盲测结果。

RGB/多视图分支和学习式门控尚未包含在本入口；本轮先检验简单融合和尺寸，避免把多个变化混在一起解释。

## 已完成运行

2026-10-19：服务器 `/root/autodl-tmp/shape_scale_20261019_r2` 已完成六组训练、15组评估和独立审计；本机另行重算指标及区间。结果在 `local_diagnostics/shape_scale_20261019/analysis/report.md`。校准全局残差为66.22±0.15%，未超过66.28%基线；尺寸和比例分支的开发集最优系数均为零。

实际执行源码保存在结果包 `source_snapshot/`。运行后工作源码仅移除未使用的import，并为独立审计器缓存纯词法归一化结果以避免重复计算；差异留在 `analysis/post_run_source_cleanup.diff`，不改变已冻结训练或预测。
