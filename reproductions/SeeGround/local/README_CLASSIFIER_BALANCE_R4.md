# 类别均衡训练与模型集成 r4

设计位于工作区`research/20261019_classifier_innovation/均衡采样与集成r4.md`。复用r3冻结PointNeXt/CLIP特征，28730训练池、404验证、3413历史评估对象；三组物理房间隔离，历史暴露限制保持。

`classifier_balance_r4.py train-select`训练12配置×3种子：加权强度.25/.75/1，采样强度.5/1，延后加权.5/1，Focal、label smoothing、128/512隐藏宽度、特征Gaussian扰动。每配置40epoch最后一轮。复用4个r3网络家族，形成16个单模型家族、16个三种子集成和6个固定配对集成。单模型按三种子平均正确数、集成按单次预测正确数公平选择。

P约束验证Macro不低于r3M，最大化平均True；M约束验证平均True不低于r3M，最大化Macro。选型与915行验证网格、原始分数、源码和全部新模型封存后，`evaluate`才读取3413对象特征，所有候选统一评分。不得据历史测试重新替换方案。

```sh
python local/classifier_balance_r4.py train-select \
  --reference /root/autodl-tmp/classifier_scan_domain_20261019 \
  --r1 /root/autodl-tmp/classifier_pair_views_20261019_r1 \
  --r2 /root/autodl-tmp/classifier_pair_views_20261019_r2 \
  --r3 /root/autodl-tmp/classifier_architectures_20261019_r3 \
  --output /root/autodl-tmp/classifier_balance_20261019_r4 \
  --design /root/autodl-tmp/classifier_balance_20261019_r4/design.md
```

保持参数，将`train-select`改为`evaluate`评分，再执行`audit_classifier_balance_r4.py --output ...`。审计需要原GPU环境重放逐epoch采样，网络/集成/选型/分类指标另用独立NumPy重算。保存实际训练对象曝光，不把有放回抽样称为每轮遍历所有对象。确定性集成只报告一条整数正确数，没有三种子标准差。

软件检查：`PYTHONPATH=local .venv/bin/python -m unittest local/test_classifier_balance_r4.py`。核对类别损失、采样分布、延后切换、宽度变体独立前向、集成公式、变体数公平比较、房间bootstrap和实际40轮训练记录。

已完成：锁定P为2391/3413（True70.06%、Exact63.81%、Top5 88.63%、Macro29.28%），M为2396/3413（70.20%、63.96%、88.72%、29.50%）。r3 M为69.70%/Macro30.18%，本轮总体与长尾存在取舍。详细结果见`research/20261019_classifier_innovation/均衡采样与集成r4结果.md`。

审计状态`passed_with_sampling_replay_limit`：sample100 seed44第1/20轮抽样类别计数未完全重放，原因未确证；原始失败及诊断保留，不修改模型和预测。锁定P/M未使用该对照。本地复核138文件哈希、915行验证网格和15个锁定/参照预测数组，通过记录位于`local_diagnostics/classifier_balance_20261019_r4/LOCAL_VERIFIED.json`。不能将其描述为无条件完全重现训练。
