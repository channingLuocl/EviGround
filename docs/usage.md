# 运行指南

## 输入

`--predictions`：外部分类器导出的 ZIP 或目录，包含 `predictions.json`、`config.json`、`EXPORT_VERIFIED.json`。适配器校验文件哈希、实例覆盖与五类别排名，不读取 `evaluation_only` 标签。保留导出格式标识 `pointnext-sampling-vote-r8-export-v1` 以读取已有文件，该标识不启用任何历史策略。

`--scene-root` 下每个场景目录需包含：

- `scene.npz`：`xyz`、`rgb`、`ids` 数组，必须与分类器所用点云一致。
- `scene_instances.json`：实例 ID 与包围框；补充实例的标注类别会被清除。
- `objects_gt.json`：仅供协调进程计算 IoU 的评估框。

`--manifest` 默认读取仓库 `data/queries/nr3d_val.json`；在仓库外使用已安装的包时，请显式指定该文件的绝对路径。

## 检查与运行

```bash
eviground --predictions /path/to/predictions.zip --scene-root /path/to/scenes \
  --output outputs/input_check --limit 10 --prepare-only

eviground --predictions /path/to/predictions.zip --scene-root /path/to/scenes \
  --output outputs/top5_single_pass --category-top-k 5 \
  --workers 8 --api-concurrency 8 --review-limit 3
```

`workers` 控制本地并行样本；`api-concurrency` 控制进程之间共享的模型调用上限；`review-limit` 是每条描述的证据工具预算，另有最终决策。

继续同一运行时重复完全相同命令。切换代码、模型、Top-K、输入或工具预算时换新输出目录。环境变量只接受 `EVIGROUND_` 前缀，旧项目配置需手动迁移。

## 输出

- `manifest.json`：本次代码、输入和配置的身份。
- `category_inputs/`：从外部预测准备的类别假设。
- `qXXXXX/prediction.json`：推理结果及观察 / 测量轨迹。
- `qXXXXX/evaluation.json`：独立评估结果。
- `qXXXXX/run.json`：调用记录与诊断，不保存 API 密钥。
- `results.json`：固定分母汇总、分组及逐条结果。以 `summary.complete=true` 作为全量结束标志。

API 请求恢复文件仅限这次运行的失败阶段，不合并其他实验结果。更改源代码后需要新输出目录，不能直接续跑清理前产生的结果。
