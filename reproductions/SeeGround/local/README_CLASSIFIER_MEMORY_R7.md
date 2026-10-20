# 训练对象近邻与核分类 r7

设计见工作区`research/20261019_classifier_innovation/训练对象检索r7.md`。保留训练对象的冻结shape/RGB特征和标签，查询仅使用待分类特征，不接收query标签。shape、RGB、等权联合三种空间，k20/k100近邻与beta10/beta20全库核、两种训练类别频率校正，共24固定候选。

仅28730训练对象进入记忆库，404验证选择，封存后评估3413历史对象；物理房间隔离。余弦和投票均float64，平票按训练清单索引排序。无随机训练，不报告三种子标准差。内存与延迟随候选记录；计数校正只用训练标签。

```sh
OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 python local/classifier_memory_r7.py select \
  --reference /root/autodl-tmp/classifier_scan_domain_20261019 \
  --r4 /root/autodl-tmp/classifier_balance_20261019_r4 \
  --output /root/classifier_memory_20261019_r7 \
  --design /root/classifier_memory_20261019_r7/design.md
```

保持其余参数，替换`select`为`evaluate`。290行验证网格只在404上选型，锁定P/M及每候选的设置后统一评估。新分支z-score，与S0或r4 M融合；r4 M保留原锁定分数尺度，不再次z-score。P/M允许回退已有参照。

`audit_classifier_memory_r7.py --output ... --reference ... --r4 ...`独立用Torch float64、按类求和重建NumPy近邻/核输出，重算全部验证选择、融合预测、指标与房间bootstrap。原始来源和结果封存哈希均检查。验证和历史数据不加入库；不把包含自身的训练检索正确率当泛化成绩。

`PYTHONPATH=local .venv/bin/python -m unittest local/test_classifier_memory_r7.py`覆盖手算核累积、稳定平票、等权跨模态余弦、筛选约束和Torch/NumPy独立前向。独立预训练场景暴露未知，两个评估集合历史暴露；不称为独立盲测。近邻/核分类是已有技术，不据此宣称方法创新。

首次审计的Torch重放在一个数组出现最大1.405e-10差异，超出初始极严格容差，底层原因未确证；原始失败日志与审计源码保留。修订审计以`passed_with_numeric_replay_limit`限定状态，除1e-8数值边界外，要求全部原始与融合分数的稳定Top5排名完全一致，并使用独立Torch分数重算全部验证网格、选择和指标。未更改保存分数或选型；不得称为无条件精确重放。

已完成：P/M均保留r4 M（2396/3413，True70.20%、Exact63.96%、Top5 88.72%、Macro29.50%）。三个验证选择新分支权重非零的对照在历史评估True为66.22%/66.72%/67.04%，均未提高。详细见工作区`research/20261019_classifier_innovation/训练对象检索r7结果.md`。本地核验64文件、290行验证网格和8条锁定/参照预测；服务器独立核验48原始数组和56条结果。两者范围分别记录，历史候选原数组仍保留服务器。
