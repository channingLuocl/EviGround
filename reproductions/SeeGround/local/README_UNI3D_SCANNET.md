# Uni3D-B 在 ScanNet 上的监督适配

本实验延续 PointBERT 的两阶段训练，输出与现有表格一致的八列结果。
本次是在 ScanNet 训练实例上训练单物体分类器；不输入 Nr3D 语句，
不把3413个固定评估物体用于训练或模型选择。这些物体已在此前实验查看过，
不能称为全新的独立测试集。

- 相同清单：1078个训练扫描、25672个训练实例；123个内部开发扫描、3171个实例。
- 相同评估：130个场景、3413个唯一物体；197类；157个词表外实例仍保留分母。
- 相同输入：每实例10000点，XYZ单位球归一化及RGB/255；复用哈希验证后的点缓存。
- Uni3D坐标约定：在输入编码器前将ScanNet Z-up转为Y-up：`(x,y,z)->(x,z,-y)`。
- 阶段一：冻结Uni3D-B，归一化1024维物体向量＋197类线性头，40 epoch；
  AdamW，lr=0.003，batch=256，weight decay=0.01。
- 阶段二：从第一阶段最优头出发，微调最后4个Transformer block、最终归一化层、
  768→1024投影及分类头；前层及PointNet BN统计保持冻结。
  主干lr=1e-5，头lr=1e-3，batch=8，累积4，最多20 epoch，patience=5。
- 微调增强沿用PointBERT：先在ScanNet坐标绕Z轴旋转和轻度XYZ jitter，
  再转为Y-up，相当于绕Uni3D的竖直轴旋转。
- 两阶段都只用内部开发集词表内交叉熵选模，最后统一评估3413个物体。

分类头训练后的分数来自学习到的197类线性头，**不再以点云与类别文本的余弦
相似度进行分类**。应报告为“Uni3D-B（ScanNet训练，冻结主干＋分类头）”和
“Uni3D-B（ScanNet训练，微调最后4层＋分类头）”，与零样本行分开。
原Uni3D文本特征仅由既有加载器校验模型来源，不参与监督优化或预测分数。

复用 `train_pointbert_scannet.py` 的训练、选模与评分流程；新增可选特征维度、
尾层配置器和编码器工厂，默认PointBERT行为保持一致。旧训练入口先备份再部署。

## 运行

服务器工程 `/root/paper/reproductions/SeeGround`：

```bash
/root/autodl-tmp/venvs/uni3d/bin/python -u local/train_uni3d_scannet.py \
  --manifests /root/autodl-tmp/seeground-v10/manifests \
  --point-bank-dir /root/autodl-tmp/pointbert_scannet_train_20261018 \
  --checkpoint /root/autodl-tmp/uni3d_baseline_20261018/assets/weights/BAAI/Uni3D/modelzoo/uni3d-b/model.pt \
  --source-dir /root/autodl-tmp/uni3d_baseline_20261018/assets/source \
  --text-features /root/autodl-tmp/uni3d_baseline_20261018/assets/text_features_eva02e_197.npz \
  --output /root/autodl-tmp/uni3d_scannet_train_20261018
```

正式训练前用独立输出目录并加 `--smoke-only`：实际batch训练10步，检查loss下降、
梯度有限且非零、尾层权重更新、早期权重和BN统计不变。
正式输出包括 `config.json`、`status.json`、两阶段历史和最优权重、
`linear_results.json`、`finetune_results.json`、完整logits/逐实例预测及 `comparison.json`。
只有状态为 `completed` 且独立复算一致后，才能把结果填入实验表。
复用缓存前校验所有清单、预处理代码、seed和完整点缓存SHA256。
同配置可从最后完成的微调epoch恢复；配置改变会拒绝复用输出目录。

## 本地检查

```bash
PYTHONPATH=local .venv-v82/bin/python -m unittest \
  local/test_uni3d_training.py local/test_uni3d_classifier.py local/test_v82_pointcloud.py -v
```

## 正式结果（2026-10-18）

| 模型 | 评估对象数 | True正确数 | True准确率 | Exact Top-1 | Top-5 | Class Mean | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Uni3D-B（ScanNet训练，冻结主干＋分类头） | 3413 | 2077 | 60.86% | 54.82% | 83.09% | 21.34% | 0 |
| Uni3D-B（ScanNet训练，微调最后4层＋分类头） | 3413 | 2017 | 59.10% | 52.80% | 82.10% | 22.04% | 0 |

分类头最优epoch为40；微调在第9轮触发早停，选择第4轮模型。微调分类头同时训练。
冻结版Exact正确1871、Top-5正确2836；微调版分别为1802和2802。
Class Mean继续包含评估集中出现的词表外GT类别，不改用仅197类内平均。
两个监督实验均比原Uni3D-B零样本16.09%高，但本轮微调未超过冻结版的实例准确率。

16项代码检查在服务器环境通过。真实GPU smoke检查loss从5.2914降至4.7695，
冻结权重和BN统计保持不变，尾层更新且梯度有限非零。
最终独立audit确认：3413实例顺序与PointBERT一致；两组3413×197分数全部有限；
预测、Top-5排序、True/Exact/Top-5/Class Mean统计一致；开发集选模与历史记录一致；
所有冻结张量与公开权重完全一致。作业结束时GPU显存和利用率均为0。

服务器结果目录：`/root/autodl-tmp/uni3d_scannet_train_20261018`。
`audit.json`为完整核验记录；`table.tsv`可直接复制进现有表格。
本地汇总：`local_diagnostics/uni3d_scannet_train_20261018_results.json`。
