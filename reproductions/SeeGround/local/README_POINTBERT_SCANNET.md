# ScanNet 上的 PointBERT 监督适配

入口为 `train_pointbert_scannet.py`，使用已经安装的 OpenShape G14 PointBERT
及预训练权重。分类任务只输入单物体的 XYZ、RGB，不使用 NR3D 语句。
这是监督适配实验，与原模型零样本迁移的 11.92% 分开报告。

## 数据和模型选择

- 复用 `seeground-v10/manifests`：官方 1,201 个训练扫描按物理房间分为
  1,078 个训练扫描（25,672 个物体）和 123 个内部开发扫描（3,171 个物体）。
- 固定评估为官方验证集 130 个场景里的 3,413 个 NR3D 目标物体。
  已核对其 `(scene, bbox_id)` 与 OpenShape 零样本评估完全相同。
  这是已查看过的官方验证集子集，不能称为全新独立测试集。
- 类别为现有固定 197 类。训练集覆盖其中 196 类，缺少 `wardrobe`。
  开发集 113 个、评估集 157 个词表外目标保留并计错，不缩小准确率分母。
- 训练/开发/评估按房间隔离，并检查云文件哈希交叉。模型选择只看内部开发集
  词表内交叉熵，固定评估集不参与选模。

ScanNet 的场景网格和分割、aggregation 标注可还原每个点的实例 ID；按该 ID
切出物体点云。服务器已经准备好的 `scene.npz` 包含 `xyz`、`rgb`、`ids`，
`objects_gt.json` 提供实例类别。转换流程沿用 `category_v10/data.py`。

## 两阶段训练

1. 冻结预训练编码器，用与现有推理相同的 10,000 点预处理，缓存归一化后的
   1,280 维特征，训练 197 类线性头 40 epoch。AdamW，lr 0.003，batch 256。
2. 从第一阶段内部开发集最优分类头开始，解冻最后 4 个 Transformer block、
   512→1280 投影及分类头。主干 lr 1e-5，头 lr 1e-3，batch 8，梯度累积 4，
   最多 20 epoch；开发集 loss 连续 5 轮未改善则停止。
   前面各层和 PointNet BatchNorm 统计保持冻结。

第二阶段训练物体使用随机绕 Z 轴旋转和轻度 XYZ jitter；开发/评估不增强。
输入缓存为 float32；不需要加载 10.2 GB 的 OpenCLIP 图像/文本网络。
所有设置是预先固定的首轮基线配置，效果需要实际评估确认。

## 运行和恢复

在服务器工程 `/root/paper/reproductions/SeeGround` 下：

```bash
/root/miniconda3/bin/python -u local/train_pointbert_scannet.py \
  --manifests /root/autodl-tmp/seeground-v10/manifests \
  --checkpoint /root/autodl-tmp/.cache/huggingface/hub/models--OpenShape--openshape-pointbert-vitg14-rgb/snapshots/d771a992218de10b967eea690e8f474aa96dd758/model.pt \
  --output /root/autodl-tmp/pointbert_scannet_train_20261018
```

首次正式训练前以独立 output 加 `--smoke-only`，验证真实 batch 的 loss 下降、
冻结权重不变和 Transformer 尾层权重更新。smoke 不保存为正式训练模型。

同一条正式命令可恢复：完成的点缓存和线性训练会复用；微调恢复最后完成的
epoch（优化器和 scheduler 一并保存）。中断的线性训练从头重训，未完成的点
缓存重新生成。恢复不保证逐位复现中断前的随机增强。代码、模型、清单哈希
改变时拒绝复用 output。文件锁避免同一目录被重复并发运行。

输出包括 `config.json`、`status.json`、两阶段 `*_history.json`、`*_best.pt`、
微调 `finetune_last.pt`。两阶段训练结束后才评估 3,413 个目标，生成
`linear_results.json`、`finetune_results.json`、预测/完整 logits 和 `comparison.json`。
`status.json` 的 `completed` 表示整个流程完成；`failed` 含错误原因。

准确率使用已有类别词法规范化，同时报告 exact accuracy、Top-5、类别平均
准确率（含和不含词表外类别两种口径）及词表外数量，便于与旧结果核对。
