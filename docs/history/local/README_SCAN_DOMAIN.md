# 扫描域监督分类与双分支融合

本轮实现冻结PointNeXt、冻结CLIP ViT-B/16四视角点云RGB渲染特征上的匹配线性分类头，并在开发集选择融合。它是监督闭集分类实验，不是新的零样本模型，也尚未接入完整Nr3D定位评估。

## 数据与模型选择

- 全部1201官方训练扫描、28730个197类词表内对象用于两个512→197分类头训练。每头101061参数，种子42/43/44，各40轮。
- 旧130房间/3413对象明确作为研究开发集；其余同房间重扫描隔离。它已经多次被查看，不能作为独立测试证据。
- 剩余12个物理房间、22扫描、404对象为本轮预留验证集合；只有在模型、checkpoint、融合权重及主方法冻结后才提取特征和评估。13个词表外目标保留计错。
- S0原3D基线；S1监督3D；S2监督RGB；S3原3D＋监督RGB；S4监督3D＋监督RGB。融合前逐对象对197类分数标准化。
- 按三个训练种子的平均开发True锁定S3（RGB权重0.4）；S4权重0.5。预留表现不用于重新选主方法。

设计文档在训练前保存于研究目录`research/20261019_scan_domain/方案.md`，实际执行版本随结果保存为`design.md`及源码快照。

## 实现入口

1. `scan_domain_protocol.py prepare`：合并训练清单，隔离物理房间，封存协议。
2. `cache_scan_domain.py train`：核对并复用原3D/RGB开发缓存，提取28730训练对象的RGB特征，核对CLIP文本空间。
3. `train_scan_domain.py train-select`：训练六个头，固定checkpoint、完整融合网格、主方法与全部哈希。
4. `download_scan_domain_holdout.py`：官方资产下载，支持HTTP Range续传。
5. `repair_scan_domain_annotations.py`：仅对标签及segment集合完全相同的重复标注保留首个ID；拒绝其他重叠。随后原协议`fetch`转换并封存清单。
6. `run_scan_domain_holdout.py`：使用匹配Torch版本重编译的原始CUDA算子，先核对三个历史特征再运行原特征提取代码。
7. `train_scan_domain.py evaluate`：只读取已锁定模型和权重，导出全部对照。
8. `audit_scan_domain.py --stage full`：独立重算线性输出、选择、融合、指标和物理房间配对bootstrap；`summarize_scan_domain.py`生成表格和诊断。

本轮修复：`scene0217_00`原始标注含31对完全重复对象。未修改官方文件，只修复派生annotation；原始与派生文件、重复映射和哈希保留。旧Torch2.14 CUDA扩展与当前2.7.1不兼容，隔离重编译后三个历史对象特征最大差值均为0，没有更改Torch或模型权重。

## 结果与本地复核

结果根目录：`local_diagnostics/classifier_scan_domain_20261019/experiment`。

开发集True：S0为66.28%，锁定S3为68.84%。预留集合True：S0为73.27%，S3为79.70%±0.25（三种子样本标准差），净增6.44个百分点，按12个物理房间配对bootstrap的95%区间为[+0.51,+11.88]。预留集合较小，原主干预训练房间范围未证实，不能等同官方测试结论。

开发集Class Mean由28.51%降至26.99%；提升整体正确率尚未解决长尾类别。S4预留True为80.45%±0.43，作为事先定义的对照报告，不能根据预留结果替换主方法。

在项目根目录使用带NumPy、Torch的Python环境复核（不需要GPU或原始场景）：

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python local/audit_scan_domain.py \
  --output local_diagnostics/classifier_scan_domain_20261019/experiment --stage full
.venv/bin/python local/summarize_scan_domain.py \
  --output local_diagnostics/classifier_scan_domain_20261019/experiment
```

`analysis/report.md`为完整报告；`analysis/dev_table_8cols.tsv`和`analysis/eval_table_8cols.tsv`必须分表注明数据角色。逐种子正确数均为整数。`class_room_changes.json`记录类别/房间纠正与误改；`*_S3_seed*_predictions.json`为后续接入定位的对象类别预测，未含GT目标标签。

六个头和训练曲线在`models/`，执行源码快照在`source_snapshot/`，验证与修复脚本在`execution_sources/`；三个归档SHA256记录于上一级`archive_checksums.json`。原始扫描、完整主干权重和CUDA二进制仍在服务器，未塞入轻量复核包。此结果是两个冻结编码器＋线性头融合的实验基线，尚不能仅凭准确率提升主张方法创新。
