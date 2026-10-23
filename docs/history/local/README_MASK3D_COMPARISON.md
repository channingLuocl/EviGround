# Mask3D ScanNet200 前端对照

本实验检查公开 Mask3D 前端在现有 130 个 ScanNet 验证场景、3413 个固定目标上的语义标签表现。它包含检测与几何匹配，不能写成 AgentGrounder 整体复现，也不等同于给定 GT 实例点云的分类实验。

## 固定协议

- 官方 Mask3D 源码：`11bd5ff94477ff7194e9a7c52e9fae54d73ac3b5`。
- 官方 `scannet200_val.ckpt`：SHA256 `6a0c4347616a27348bed32ff1ae225cab0087e0a0148f36d1023ca2d46f82534`。使用验证评测权重，未使用 train+val benchmark 权重。
- 原始网格 XYZ/RGB 与无标签 superpoint 分割作为模型输入。原始网格经官方轴对齐后，与现有评估点云逐顶点核对坐标和颜色。GT 实例 ID、类别与 Nr3D 文本不进入模型。
- 0.02m voxel、150 queries、segment masks、最终 decoder、DBSCAN eps0.95/min_samples1、query/class 概率前750项，置信度乘掩码质量，按官方方式恢复全分辨率并做 superpoint 多数投票。
- 复现官方 ScanNet200 的 `label_offset=2` 及 chair 特例，避免类别编号偏移。
- 完全相同的预测掩码只保留最高置信度类别。随后以点掩码 IoU≥0.5、类别无关的一对一分配匹配固定目标；先最大化有效匹配数，再最大化 IoU。匹配不读取 GT 类别。
- 未匹配目标计错，分母始终3413；额外报告匹配覆盖率及匹配对象上的类别准确率。Errors 指程序错误，不包含漏检。
- True/Exact/Class Mean 沿用已有词法评分；不对 proposal 匹配结果补造 Top-5。原始类别名称列表与现有197类词表的来源一致，但 Mask3D 保留其自身语义输出空间和检测过程。
- 此评估集此前已多次查看，属于验证集分析。

## 运行

在已编译 MinkowskiEngine、pointnet2 且 Torch/CUDA 兼容的独立环境中：

```bash
python local/prepare_mask3d_assets.py --asset-dir /path/mask3d/assets
python local/prepare_mask3d_inputs.py --manifest /path/seeground-v10/manifests/eval.json --output /path/mask3d/inputs
python local/run_mask3d_comparison.py --manifest /path/seeground-v10/manifests/eval.json --source /path/mask3d/assets/Mask3D --checkpoint /path/mask3d/assets/scannet200_val.ckpt --inputs /path/mask3d/inputs --output /path/mask3d/full
```

单场景试跑追加 `--limit 1`，使用独立输出目录。三份并行运行分别追加 `--shard 0 3`、`--shard 1 3`、`--shard 2 3`，输出为同一根目录的 `shard0`、`shard1`、`shard2`。逐场景随机种子只由固定种子和场景名决定，因此分片不改变该场景的随机采样。三个任务全部完成后：

```bash
python local/merge_mask3d_comparison.py --root /path/mask3d
python local/audit_mask3d_comparison.py --root /path/mask3d
python -m unittest discover -s local -p test_mask3d_matching.py
```

合并脚本检查模型、代码、协议、全部掩码哈希和场景集合，然后使用原评分程序汇总。审计程序重读保存的压缩掩码与 GT 几何，核对一对一关联、类别编号、漏检计分及全部指标。每个场景保留 packed masks、类别与置信度，完整结果保留 `FROZEN.json`。

## 本次环境

RTX5090、CUDA toolkit12.8、Torch2.7.1+cu128、Python3.12。MinkowskiEngine 使用作者固定的 `02fc608bea4c0549b0a7b00ca1bf15dee4a0b228`，按 sm120 编译。与新 CUDA/Thrust 的兼容修补、官方源码压缩包和权重哈希保存在实验 provenance 中；补丁涉及缺少的 Thrust 头文件、NVTX 命名冲突及 shared pointer 构造接口，没有修改网络结构或 checkpoint 张量。

训练/分类头对照在独立的 `run_pointnext_adaptation_experiment.py` 与 `audit_pointnext_adaptation.py` 中，不依赖 Mask3D 的检测结果。
