# 冻结特征的ExtraTrees r10对照

设计见research/20261019_classifier_innovation/随机树分类r10.md。RGB512与shape+RGB1024两种特征，各普通/均衡class_weight；三个种子，共12个森林。256树、max_leaf_nodes1024、min_samples_leaf2；8线程拟合，单线程固定树序概率累加。输出classes_显式映射197类，缺类不移位；log(max(p,1e-6))标准化后与S0/r4 M/r8 M融合。

`classifier_forest_r10.py train-select/evaluate`，参数output/reference/r4/r8/design。87行验证网格，P/B/M选型在历史评分前封存，保留纯森林诊断和回退。`audit_classifier_forest_r10.py`逐树直接取叶节点类别分布独立累加，核验所有概率、精确Top-5、网格/选型/指标/房间bootstrap；不重拟合训练过程。

依赖只复制旧环境已有sklearn1.5.2/joblib1.6.0/threadpoolctl3.7.0/cloudpickle到实验deps_existing并核验源文件哈希。活动Torch/NumPy不升级。初始新版本下载失败日志、同步复制超时后确认进程终止及对现有复制结果的校验记录均保留，未盲目重复复制。

远程结果目录/root/autodl-fs/classifier_forest_20261019_r10。启动句柄见local_diagnostics中同名目录LAUNCH.json。GPU残差训练独立并行，CPU实验OPENBLAS/MKL/OMP限制2，森林自身拟合8线程。

CPU类别映射契约检查：`PYTHONPATH=local .venv/bin/python -m unittest local/test_classifier_forest_r10.py`。历史评估暴露及原主干预训练场景未知限制保持；r8参照继承旋转精度限制。True与Exact分列，参照接口重复不当作独立种子。
