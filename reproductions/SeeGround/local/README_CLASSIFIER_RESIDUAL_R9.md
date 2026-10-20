# 冻结主干的输入残差 r9

设计：research/20261019_classifier_innovation/冻结主干残差r9.md。原stem输出叠加7→64→32点级MLP（2592参数），末层零初始化；原主干/BN/投影及10*cosine文本分类头全部冻结。三组geometry_kd/color_ce/color_kd分离RGB与教师约束，教师仅为训练对象原始特征。三种子×8轮，checkpoint4/8；111行验证选择后才评分3413。

`classifier_residual_r9.py`支持preflight/train-select/evaluate。共用参数为output/reference/r4/r8/color/checkpoint/design，服务器目录见local_diagnostics/classifier_residual_20261019_r9/LAUNCH.json。借用r6已封存7通道点缓存，不额外复制大数据。新结果存储到/root/autodl-fs，避免继续挤占本地系统盘。

`audit_classifier_residual_r9.py`核对完整旧参数和BN冻结、全部18 checkpoint/36原始前向、NumPy文本投影与精确Top-5、全部111行验证选择及融合/指标/房间bootstrap。不等于完整优化轨迹重演。r8参照继承旋转浮点限制。

CPU契约检查：`PYTHONPATH=local .venv/bin/python -m unittest local/test_pointnext_residual_r9.py`。服务器128对象20步预检查已通过：初始函数差0，教师缓存最大差1.64e-7、RGB入口梯度非零、总loss下降、原参数与BN完全不变，峰值15.3GB。这里只证明接线可训练，不证明评估改善。

参照保留原分数尺度，新分支z-score。P无Macro下限；B保Macro再最大化True；M保True再最大化Macro。所有目标保留并报告失败/取舍，不能据3413分数更换部署。True与Exact分列，OOV保持计错；历史评估已暴露，原预训练场景暴露未知，非独立盲测。
