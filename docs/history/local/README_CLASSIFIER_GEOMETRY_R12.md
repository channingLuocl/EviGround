r12在当前28730训练／404验证／3413历史评估协议下重新提取Uni3D-B和ULIP-2原始权重特征。原来的监督分类头/不同清单缓存不复用。设计见research/20261019_classifier_innovation/互补点云表征r12.md。

使用既有`/root/autodl-tmp/venvs/uni3d/bin/python`（已确认Torch2.7.1+cu128、pointnet2_ops3.0.0、timm0.9.7），不要升级Torch。Uni3D的EVA02-E文本与ULIP-2的bigG文本各自加载；只有监督分类头可以拼接不同编码器的特征。

统一入口：`classifier_geometry_r12.py`，所有阶段带`--output/--reference/--r4/--r8/--uni3d/--ulip2/--design`。资产根路径为先前两组baseline目录。

顺序：preflight → cache train/uni3d → cache train/ulip2 → cache eval/uni3d → cache eval/ulip2 → train-select → cache dev/uni3d → cache dev/ulip2 → evaluate → audit_classifier_geometry_r12.py。cache额外指定`--split`和`--encoder`。dev缓存会要求验证选型已封存；旧目录或已有缓存禁止覆盖。

六组×三个种子、固定40轮；111行验证网格锁定P/B/M与各组选择。全部纯模型、融合、参照和失败结果都保留；回退不能称新模型收益。原始浮点特征持久保存，约300MB，不使用r11临时token。

运行`python -m unittest discover -s local -p test_geometry_classifier_r12.py`核验匹配文本空间、对照分支隔离、独立NumPy前向和可学习性；GPU preflight另在3个真实训练对象核验官方predict桥接、完全相同重复采样及冻结状态。审计抽样重放两编码器共18个对象，复算36组分类头、精确Top-5、完整选型及房间区间；不把抽样重放称为全量主干重演。

排队任务必须先确认r11完成且释放memfd，避免并行占用GPU；实际PID/CHILD.json用于确认进度，超时不构成重启理由。封存后仍禁止修改原模型、协议和预测，失败须保留记录并先诊断。
