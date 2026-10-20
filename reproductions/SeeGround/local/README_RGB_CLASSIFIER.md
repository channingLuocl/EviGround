# 冻结 CLIP RGB 多视角分类实验

`rgb_classifier.py` 仅从已分割对象的真实XYZ/RGB生成四幅固定点云图像；冻结CLIP ViT-B/16与现有197类文本空间对齐。`run_rgb_experiment.py` 按训练对象渲染检查、开发集选权重、冻结评估三个阶段运行。训练分割和已知对象实例与原对照相同，不能将本实验视为检测分割方法。

新增依赖使用服务器已有的open_clip、Pillow、NumPy和Torch；不升级Torch/CUDA，不下载其他权重。HF到OpenCLIP的转换必须严格加载全部参数，并与现有全部类别文本向量核对（最大绝对误差≤1e-4、最小余弦≥.99999）；QuickGELU显式启用。

渲染参数、融合公式与搜索网格见冻结的`config.json`，事前设计见工作区`research/20261019_classifier_experiment_design/rgb_phase2/预设方案.md`。无CLIP分类头训练；复用第一阶段三个尺寸头作V3消融。每对象跨197类标准化分数后融合，不能把选择出的权重直接解释为概率。

运行：

```bash
python local/run_rgb_experiment.py \
  --feature-root /path/seeground-v10 \
  --prior /path/pointnext_adaptation_20261019 \
  --scale-root /path/shape_scale_20261019_r2 \
  --clip /path/clip-vit-base-patch16/pytorch_model.bin \
  --output /path/fresh_rgb_experiment --stage qa
```

人工或代理检查固定训练索引的三个QA拼图之后，使用`freeze(output,'QA_FROZEN.json', ['config.json','text_alignment.json','qa/records.json','qa/object0.png','qa/object1.png','qa/object2.png'])`固定QA结果，再将`--stage`改为`dev`运行。开发集搜索完成自动写`SELECTION_FROZEN.json`，然后才运行`eval`。每个场景的提取结果单独保存以便在相同代码、参数、权重和清单下恢复；代码或输入变化必须使用新输出目录。

```bash
python -m unittest discover -s local -p test_rgb_classifier.py
python local/audit_classifier_data.py --feature-root /path/seeground-v10 --prior /path/pointnext_adaptation_20261019 --output /path/fresh_data_audit
python local/audit_rgb_experiment.py --output /path/rgb_experiment
```

数据审计检查全部场景云文件哈希、颜色、缓存尺寸与准备好的标签；仅在原始aggregation存在时额外核对其标签，报告核对范围。结果审计独立重算全部开发集网格、评估分数、标签归一化指标、纠正/误改和房间配对bootstrap。保留OOV对象，`Errors=0`仅表示执行无错误。

现有eval为此前已查看的验证集；开发集已被用于此前训练/选择。PointNeXt发布权重的确切预训练场景范围未知。不要宣称独立测试集或严格未见主干预训练场景。渲染为稀疏点云视图，不等同原始RGB相机帧；oracle仅为互补性诊断。
