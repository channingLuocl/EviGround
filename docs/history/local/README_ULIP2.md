# ULIP-2：相同3413对象的分类与监督适配

本实验使用 Salesforce 官方发布的 `ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt`。
不把论文中 ModelNet40 的 fine-tune 数值当成本数据集结果；实验是单物体类别预测，
不是完整 Nr3D 语言指代表达定位。

- 官方代码：`salesforce/ULIP`，commit `95d480fe2b16c06d0558c60b5cfea981b4cdc8eb`。
- 官方权重：HF dataset `SFXX/ulip`，revision `3f33d96e6bfec2a793d06acf0df4c591ddd47489`。
- 点云权重 SHA256：`a4b5ed9799d5841a1646e0fb7d24cb8dcdd3b3e6fab2e2575de2531f71274adb`。
- 文本：`ViT-bigG-14 / laion2b_s39b_b160k`，严格加载完整官方权重后编码197类。
  复用服务器已有10GB文件，不重新下载；其SHA256为
  `db71bd27925917762c506c1b8a08e191d236e5586b5cd99f77d11aced578ec34`。
  不混用 Uni3D 的 EVA02-E 文本向量。单模板仍是 `a {} in a scene`。
- 保留官方18层、384维PointBERT，512组、每组32点；CLS与maxpool拼为768维，
  严格加载官方768→1280语义投影，不随机替换投影。
- 源码独立包仅修改相对import、延迟未使用DGCNN的knn_cuda及可视化依赖。
  计算路径中的7个类经AST核对保持一致；原始和运行时SHA256均保存。
  官方checkpoint包含NumPy scalar优化器元数据，仅在完整SHA256匹配后使用
  `torch.load(weights_only=False)`，再严格加载全部点云参数。

## 固定协议

130场景、3413唯一物体、197候选类；157个词表外GT实例保留分母。
每实例10000点，XYZ中心化／单位球归一化，RGB/255或缺失颜色0.4；seed42。
官方 `data/dataset_3d.py` 将绕Y旋转描述为绕up direction，并使用 `gravity_dim=1`；
在运行前固定从ScanNet Z-up到Y-up的保持手性旋转 `(x,y,z)->(x,z,-y)`，
不按评估分数选择坐标。
True为既有词法等价匹配，Exact为原字符串一致，Top-5沿用等价匹配。
Class Mean包含评估中出现的所有规范化GT类别，包括词表外类别。
保存全部3413×197分数、排名、逐实例预测、配置、输入哈希和独立复算记录。

监督阶段复用与PointBERT、Uni3D相同的训练／内部开发／评估清单，
按物理房间与点云哈希隔离：25672／3171／3413实例。
不把3413实例用于优化或模型选择；这些实例此前已被查看过，不能称全新测试集。
复用完整哈希校验的Z-up点缓存，不复制7.7GB数据。

1. 冻结主干与投影：归一化1280维向量＋197类线性头，40epoch；
   AdamW lr0.003、wd0.01、batch256。
2. 从第一阶段最优头出发，微调最后4/18个block、最终norm、768→1280投影和分类头；
   主干lr1e-5、头lr1e-3、wd0.01、batch8×累积4、cosine LR；最多20epoch、patience5。
   冻结前层与BN统计；训练增强仍为ScanNet坐标绕Z旋转和XYZ jitter，再转Y-up。
3. 两阶段均只按内部开发集词表内交叉熵选模，最终用固定3413实例评分。
   监督分数来自学习的分类头，单列为监督适配，不称零样本。

## 服务器入口

工程 `/root/paper/reproductions/SeeGround`；复用独立环境
`/root/autodl-tmp/venvs/uni3d/bin/python`，保持Torch2.7.1+cu128和RTX5090兼容。

```bash
/root/autodl-tmp/venvs/uni3d/bin/python -u local/run_ulip2_experiment.py
```

先准备并完整校验权重、源码及文本缓存，再运行上述入口。
入口依次执行11项CPU检查、1／100／3413零样本、零样本独立audit、
10步真实GPU训练smoke、两组监督训练及最终独立audit。
训练smoke检查loss下降、有限非零梯度、尾层更新、冻结权重及BN不变。
结果需状态完成、Errors=0且独立audit通过才能填表。

- 零样本及日志：`/root/autodl-tmp/ulip2_baseline_20261018`。
- 监督训练：`/root/autodl-tmp/ulip2_scannet_train_20261018`。
- 原始源码快照：`local_diagnostics/ulip2_sources_20261018`。
- 不覆写PointBERT、OpenShape或Uni3D的既有结果。

## 正式结果（2026-10-19）

三组均于00:20:07完成并通过独立audit；GPU结束时0MiB、0%利用率。

| 模型 | 对象数 | True正确数 | True准确率 | Exact Top-1 | Top-5 | Class Mean | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ULIP-2官方零样本 | 3413 | 529 | 15.50% | 15.41% | 26.25% | 4.31% | 0 |
| ULIP-2冻结主干＋分类头 | 3413 | 2052 | 60.12% | 53.97% | 82.10% | 21.21% | 0 |
| ULIP-2微调最后4层＋分类头 | 3413 | 1858 | 54.44% | 48.02% | 78.93% | 19.54% | 0 |

冻结头最优epoch40；微调6轮早停、内部开发集最优epoch1。
11项CPU检查和10步真实GPUsmoke通过；smoke loss从5.2733降至4.6612。
三组分数、排名与汇总均独立复算一致；微调全部冻结权重与官方原始checkpoint一致。
本轮冻结版优于微调版，未为提升结果改变既定训练／评估协议。
服务器仅按需安装termcolor2.4.0（--no-deps），Torch维持2.7.1+cu128。
本地完整记录见`../local_diagnostics/ulip2_experiment_20261019.md`，
汇总与可粘贴表格为同名前缀的`_results.json`和`_table.tsv`。
