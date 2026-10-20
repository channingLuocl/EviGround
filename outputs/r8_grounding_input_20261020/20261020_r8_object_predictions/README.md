# r8 最佳已核验物体预测导出

固定使用 r8 验证集预先锁定的 B/M 方案：九次点云采样/旋转投票，RGB 权重 0.6。
本次仅导出既有结果，没有重新训练、重新预测或按测试真值修改标签。

- 130 个场景，3413 个物体，197 个候选类别。
- True：2411/3413 = 70.64%；Exact Top-1：64.34%；Top-5：88.60%；Class Mean：29.64%；Errors：0。
- `predictions.json`：完整预测列表，以 `(scene, bbox_id)` 标识物体，不含真值。
- `scene_tables/<scene>/objects.json`：分场景物体表，`target` 是预测显示类别，不是真值；Top-5 未经过类别门槛筛选。
- `object_predictions.csv`、`object_predictions.xlsx`：便于查看的平表，保留全部物体、Top-5 和几何。
- `all_scores.npy`：原始 float64 融合分数，行与 predictions.json 一致，列与 vocabulary.json 一致。
- `evaluation_only/`：评估统计和真值核对，不能用作定位模型输入。
- `config.json`、`EXPORT_VERIFIED.json`：输入身份、分数语义和导出校验记录。

## 接入后续定位流程

本导出的 schema 明确标识为 r8，不冒用 v8.3 版本。旧 v9 入口强制检查 v8.3 版本，
需要适配新源格式和分数语义后再接入；单纯替换旧目录不能直接运行。
融合分数不是概率或原始余弦值，不能直接沿用旧版 0.20 最低分和 0.03 分差。
本次不选择或校准新门槛，也不启动端到端定位。后续应在开发集固定门槛，再比较相同描述集。

物体几何取自已冻结的原输入记录，按场景/物体 ID 对齐，并逐场景核对点云 SHA256。
源 r8 审计状态为 passed_with_rotation_precision_limit，保留既有旋转浮点复现限制。
3413 对象集合为历史已暴露评估集，不属于新独立盲测。
