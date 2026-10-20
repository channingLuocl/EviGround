# PointNeXt主干适配 r5

设计：工作区`research/20261019_classifier_innovation/点云主干微调r5.md`。三组各三种子：冻结主干、微调末两级及语义投影、相同微调加轻量几何增强。28730训练、404验证、3413历史评估；房间隔离。原197词表、XYZ+height输入和原始评测定义保持。

`classifier_backbone_r5.py`有`smoke`、`train-select`、`evaluate`三阶段。`smoke`严格核对原权重、旧特征桥接、指定层梯度/更新、其他权重和BN不变、固定批次损失下降。每模型12轮，前2轮只训练头，4/8/12轮checkpoint在验证集上比较。选型封存后才对3413推理。候选可与S0或r4锁定M融合；保留r4 M，允许全部新模型未胜出。

```sh
python local/classifier_backbone_r5.py smoke \
  --reference /root/autodl-tmp/classifier_scan_domain_20261019 \
  --r4 /root/autodl-tmp/classifier_balance_20261019_r4 \
  --output /root/classifier_backbone_20261019_r5 \
  --checkpoint /root/paper/reproductions/SeeGround/weights/pnext_cls.pth \
  --design /root/classifier_backbone_20261019_r5/design.md
```

保持其余参数，将`smoke`依次换为`train-select`、`evaluate`。已存在的未完成训练或点云缓存会保留并报错，禁止自动覆盖重跑。缓存为float32，部署到有足够空间的系统盘新目录；不升级已有Torch或删除实验。

独立审计：`audit_classifier_backbone_r5.py --output ... --reference ... --r4 ... --checkpoint ...`。检查所有checkpoint冻结状态及实际变化、重做所有验证/历史评估原始前向（头用NumPy）、独立重算验证网格、选择、融合和指标，输出房间bootstrap。审计不声称完整重演优化轨迹。训练日志loss为各批加权CE的对象数加权平均，不等同于整池一次计算的加权CE。

软件检查：`PYTHONPATH=local .venv/bin/python -m unittest local/test_classifier_backbone_r5.py`，覆盖几何增强不修改输入、单位半径/height契约、语义头初始化、参数开放范围和验证约束。真实CUDA反向另由服务器smoke验证。

两评估集合均历史暴露，原预训练场景暴露未知，不能称为独立盲测。True为归一化正确率，Exact Top-1另列。参照S0/r4 M只有一个模型结果；保存为多个种子文件仅为接口一致，不构成三个独立重复。

结果表中的`frozen/last2/last2_aug`为各家族在验证集选定checkpoint及融合后的对照，不能将其直接称为单独3D网络的准确率；具体基线/权重见`selection.json`。若锁定P/M回退到S0/r4 M，应报告保留原方案，不能归功于新增训练。

设计文案中“两分支先标准化”与实现有一处差异：新网络分数与S0按对象z-score，r4 M直接使用已锁定融合分数，不再z-score。因此实际公式为`(1-w)*r4_M_saved + w*z(new)`。已封存源码、网格、预测和独立审计均按此公式，不事后改分数。本轮最终w=0，此差异不改变回退结果。
