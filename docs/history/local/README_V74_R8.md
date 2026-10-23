# v7.4 + r8 预测类别：Top-1 主标签与 Top-5 类别假设

保留 v7.4 的解析、几何证据、渲染、多模态定位和复核流程，复用 v9 已有的多类别接口。
新增 r8 输入适配，版本记为 `predicted-category-grounding-v9-r8-r1`。

## 接入方式

- 一物体一个 `bbox_id`，`target/category` 仍为分类器 Top-1；不复制物体，也不拼接五个类别为一个标签。
- 默认把完整 Top-5 保留为互斥假设；描述解析词表、目标检索和参照检索均使用保留假设。
- 模型收到每项原始 `score`、`rank` 和 `gap_from_top1`，知道 Top-1 是分类器主假设，后续排名是备选。最终结合完整描述、视觉和空间证据选择一个物体。
- Top-5 的作用是扩大召回，Top-1/排名提供分类先验；没有添加未经开发集验证的融合权重、概率转换或硬阈值，也没有“存在 Top-1 匹配就丢弃其他物体”的短路规则。
- r8 分数是 `weighted_standardized_3d_rgb_fusion_not_probability`，不是旧版余弦值。r8 路径拒绝 `--min-category-score` 和 `--max-category-gap`，避免误用旧 `0.20/0.03`。
- Top-1 对照使用完全相同的分类预测和定位流程，仅把可绑定类别缩减为第一名。无匹配类别时保留原有 v9 弃答行为，不引入全场候选兜底。

入口不会导入 v7.4 的历史答对缓存，所有查询重新推理；续跑则保留该次运行的所有结果，包括答错、弃答和错误。旧 v7.4 的“历史曾答对”指标不能直接作为本次单轮基线。

## 数据

本次 ZIP 已解压到项目根目录下：

```text
outputs/r8_grounding_input_20261020/20261020_r8_object_predictions/
```

加载器只读取 `EXPORT_VERIFIED.json` 的状态与文件哈希、`predictions.json`、`config.json`。
不把导出中的 `evaluation_only/`、真值类别或查询答案用于推理。
检查预测/配置 SHA256、Top-1/Top-5 一致性、排名、重复 ID、场景覆盖，以及运行点云的 SHA256 和实例 ID。
也可通过 `--predictions` 指定原 ZIP，加载时无需另行解压。
仍使用已准备的实例分割和几何；本次只替换类别来源。
父进程使用验证集答案和 GT 框计算指标，模型子进程只获得场景路径和描述。

## 运行

以下命令均从 `reproductions/SeeGround` 执行。启动脚本会加载现有 `.env`。

```bash
# 仅准备两组完整输入，不调用模型 API；各 7,805 条描述。
bash local/run_local_v74_r8.sh --category-top-k 1 --prepare-only \
  --output local_outputs/v74_r8_top1_full_20261020
bash local/run_local_v74_r8.sh --category-top-k 5 --prepare-only \
  --output local_outputs/v74_r8_top5_full_20261020

# 完整评测：相同设置下去掉 --prepare-only 即可续用已准备目录。
bash local/run_local_v74_r8.sh --category-top-k 1 \
  --output local_outputs/v74_r8_top1_full_20261020
bash local/run_local_v74_r8.sh --category-top-k 5 \
  --output local_outputs/v74_r8_top5_full_20261020

# 先跑小批次时必须使用另一个输出目录；--limit 在准备与实际运行时必须一致。
bash local/run_local_v74_r8.sh --category-top-k 5 --limit 100 \
  --output local_outputs/v74_r8_top5_100_20261020

# 两组都完成相同查询后，统计 Top-5 净增/净损和各项定位指标。
PYTHONPATH=local .venv/bin/python local/compare_grounding_v9.py \
  local_outputs/v74_r8_top1_full_20261020 local_outputs/v74_r8_top5_full_20261020
```

默认并发沿用 v9：10 个本地任务、8 个 API 槽。可通过 `SEEGROUND_WORKERS`、
`SEEGROUND_API_CONCURRENCY` 调整；配对两组请保持一致。代码、模型、输入或其他设置改变后需要新目录。

## 指标与改进依据

`results.json` 记录 ID 准确率、IoU、弃答、调用数/耗时、目标候选召回率及候选数量。
新增 `target_category_best_rank`：真值物体与解析目标匹配的最佳假设排名，**仅在评估父进程计算**。
`target_category_rank_groups` 分别汇总 `top1_match` 与 `top2_to_5_rescue` 的数量和定位准确率。
这里使用现有类别别名检索，因此匹配率不等于 exact 分类准确率；“rescue”只表示额外召回，未必定位正确。

先比较 Top-1 与完整 Top-5，确认收益是召回增加还是干扰增加，再在独立开发集上选择排名/分差策略。
Top-5 分类命中率不能当定位准确率。这 130 个场景属于历史已暴露验证集，不作为新独立盲测。

## 测试

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=local .venv/bin/python -m unittest \
  local/test_categories_r8.py local/test_v9.py local/test_v7_4.py -q
```

覆盖 r8 分数保真、拒绝旧阈值、Top-1 对照、第二类别用于目标和参照、模型收到分数语义、
无 GT 类别输入、ZIP/目录一致性、文件篡改、点云错配，以及原有 v9/v7.4 回归。
