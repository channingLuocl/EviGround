# 点级RGB与几何联合分类 r6

设计见工作区`research/20261019_classifier_innovation/点级颜色接入r6.md`。扩展PointNeXt stem输入4→7通道，复制原4通道权重，新增RGB权重为零。`ColorPointNext.forward`传递完整7通道，避免原PcdClassifier将RGB截掉。初始原函数保持；同时开放stem、末两级、原预测层与语义投影，BN冻结。

三组各42/43/44种子：geometry（RGB为零）、color（真实RGB）、color_regularized（训练时颜色扰动/整物体颜色丢弃，评估自然RGB）。仅本轮内部对照具有相同批次/参数开放范围；不能将跨r5差异全部归因于RGB。固定batch128、12轮、4/8/12轮checkpoint，前2轮只训头。

28730训练、404验证选型、3413封存后评估。三组各epoch三种子平均，在110行验证网格中选择与S0/r4 M的融合。既有参照保留，允许不胜出。XYZ+height逐对象与r5缓存精确一致，RGB用相同点采样索引。输入缓存位于系统盘新目录，不删除旧数据。

```sh
python local/classifier_color_r6.py train-select \
  --reference /root/autodl-tmp/classifier_scan_domain_20261019 \
  --r4 /root/autodl-tmp/classifier_balance_20261019_r4 \
  --r5 /root/classifier_backbone_20261019_r5 \
  --output /root/classifier_color_20261019_r6 \
  --checkpoint /root/paper/reproductions/SeeGround/weights/pnext_cls.pth \
  --design /root/classifier_color_20261019_r6/design.md \
  --smoke /root/classifier_color_20261019_r6/preflight_gpu128.json
```

保持其余参数，将`train-select`替换为`evaluate`评分。`audit_classifier_color_r6.py`使用output/reference/r4/r5/checkpoint参数，核对所有checkpoint、冻结状态、RGB通道梯度和权重，独立比较完整XYZ+height缓存和抽样RGB对应，重放评估前向、验证选型和指标。审计不等于完全重演优化轨迹。

CPU：`PYTHONPATH=local .venv/bin/python -m unittest local/test_pointnext_color_r6.py`。服务器32/128对象固定批次预检查均通过；128批次峰值13.5GB、原始输出差0、RGB梯度非零、损失下降。该检查证明接线与优化可用，不证明泛化提升。

S0/r4 M每个参照仅一组实际预测，多种子文件为接口重复；不将其当独立重复。True与Exact分别列出。历史评估暴露及原预训练场景暴露未知的限制保持。直接加入RGB是已有技术的对照探索，不据此声称方法创新。

结果中的`geometry/color/color_regularized`是各家族在验证集选择checkpoint及融合后的对照，不能误称为单独点云网络的分数；设置见`selection.json`。若锁定方案为旧参照，应明确报告新方法未胜出。

设计文案“两分支按对象标准化”与已封存实现不同：S0及新网络按对象z-score，r4 M保留已有融合分数尺度。实际为`(1-w)*r4_M_saved + w*z(new)`。r6训练启动后发现此差异，保留原代码/协议/分数，在结果报告中明确披露；不根据历史评估重算另一套融合再换型。
