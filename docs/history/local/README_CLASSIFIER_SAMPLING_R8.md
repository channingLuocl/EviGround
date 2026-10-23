# 原PointNeXt多次采样投票 r8

设计：工作区`research/20261019_classifier_innovation/点云多次采样r8.md`。保留原PointNeXt和CLIP文本，不训练参数。九个固定输入为对象种子42~48的2048点采样，另加种子42的Z轴±10度。比较single/sample3/sample7/yaw3/sample_yaw9，平均单位语义向量的余弦分数。不会恢复扫描未采到的表面。

404验证选型，封存后3413历史评估；不使用query标签提特征，房间互斥。以r4 M和S0原分数恢复锁定RGB支路，30个投票/RGB权重组合，加2参照，共32行验证网格。P只最大化验证True，B要求Macro不低于r4 M再最大化True，M要求True不低于r4 M再最大化Macro。三个目标分别报告，P可能存在Macro代价。相同分数优先前向次数少。

```sh
python local/classifier_sampling_r8.py preflight \
  --reference /root/autodl-tmp/classifier_scan_domain_20261019 \
  --r4 /root/autodl-tmp/classifier_balance_20261019_r4 \
  --output /root/classifier_sampling_20261019_r8 \
  --checkpoint /root/paper/reproductions/SeeGround/weights/pnext_cls.pth \
  --design /root/classifier_sampling_20261019_r8/design.md
```

其余参数不变，再执行`select`、`evaluate`。正式任务等r6评分和审计结束后占用GPU。预检查使用训练对象3个固定索引，原特征最大差1.49e-8、同输入重复精确一致，所有模型状态不变。原始九次归一化语义特征按对象保存，包含点云输入哈希与推理耗时；不覆盖旧结果。

`audit_classifier_sampling_r8.py --output ... --reference ... --r4 ... --checkpoint ...`重放全部九次编码前向、原缓存桥接和模型状态；独立按“先投影再求均值”复核“先均值再投影”、RGB代数恢复、32行网格、P/B/M选择、保存预测与房间bootstrap。前向允许明确数值容差，排名和最终指标严格核对。

`PYTHONPATH=local .venv/bin/python -m unittest local/test_classifier_sampling_r8.py`检查旋转不改Z/height/半径、输入不被修改、RGB分支恢复、投票线性关系和不同目标选择。两评估集合历史暴露，原预训练场景暴露未知，不能称为独立盲测。True和Exact分别列，OOV保留计错；测试时投票是已有技术对照。

最终审计状态为`passed_with_rotation_precision_limit`。首次独立矩阵旋转重放失败；按封存逐坐标浮点计算顺序重放全部34353次编码，输出与缓存最大差0，同时独立验证数学旋转输入最大差8.225e-8。等价算术并不保证编码输出一致，未定位网络内部原因；原审计失败、诊断及修订源码保留。不改原协议、缓存、选择或预测。详见研究目录的点云多次采样r8结果.md。
