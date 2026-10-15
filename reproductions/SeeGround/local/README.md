# SeeGround 本地代码

## 已冻结的 v7.4 结果

2026-10-15 完成：**5,435 / 7,805 = 69.63%**，GT 类别。
原目录为 `local_outputs/v7_4_nr3d_val_full_gt`，保留所有原始文件，并用 `FROZEN.json` 标记冻结。
当前全量批处理、重试、汇总、导入缓存、直接推理和 API 并发锁目录均检查该标记，
拒绝向冻结目录及其子目录写入。

- [完整归档说明](/Volumes/KESU/论文/paper/reproductions/SeeGround/archives/v7.4_20261015_6963/README.md)
- [电脑本地关键结果备份](/Users/luocl/Desktop/seeground/archives/v7.4_20261015_6963/README.md)

完整归档包含原批次和被引用的历史来源批次：预测、评估、错误日志、API 记录、图片、轨迹、检查点、重试目录。
`source_before_refactor.tar.gz` 保存重构前实际代码，`git_history.bundle` 保存提交历史。
`verification.json` 保存所有归档文件的 SHA-256；完整归档逐文件读取后确认才标记完成。
本地关键备份另存完整最终评估汇总、正确缓存、配置、查询清单与代码，完整图片归档在 KESU 上。

统计策略仍为 `cumulative_ever_correct`，混有历史版本的成功条目；该冻结分数用于开发分析。

## 重构后的入口

本次是保持推理行为的结构重构，继续使用 v7.4 规则，输出使用独立目录
`local_outputs/v7_4_refactored_nr3d_val_full_gt`。没有将重构后的代码或新结果写入冻结批次。

只准备资源并导入历史正确缓存，不发起模型请求：

```bash
cd /Users/luocl/Desktop/seeground/reproductions/SeeGround
local/run_local_full.sh --prepare-only
```

后续确实需要运行新实验时：

```bash
local/run_local_full.sh
```

默认本地并发 **10**、API 并发 **8**。可用 `SEEGROUND_WORKERS` 和
`SEEGROUND_API_CONCURRENCY` 覆盖；`SEEGROUND_RUN_DIR` 指定新的输出目录。
同一目录用于断点续跑；代码或模型设置改变后使用新目录。
旧的版本启动脚本保留兼容；指向已冻结 v7.4 目录的命令会明确拒绝执行。

正确缓存策略保持原样：历史正确不重跑；普通续跑保留已完成错误；`--retry-errors`、
`--retry-null`、`--retry-false` 三者选一。重试的旧尝试保留在各自目录。
这类缓存用于开发累计结果，不能作为不依赖 GT 的正式单次评测。

## 模块分工

| 模块 | 职责 |
|---|---|
| `run_local_full.sh` | 当前启动入口、GT 类别与并发设置，输出到新目录 |
| `run_size_validation.py` | CLI、任务调度、子进程与重试；保留旧导入兼容 |
| `run_storage.py` | 冻结目录检查和原子 JSON 写入 |
| `validation_cache.py` | 跨版本历史正确缓存、来源与被替换评估归档 |
| `validation_reuse.py` | 显式复用的兼容性检查；隔离 v7.1→v7.3 历史特例 |
| `validation_reporting.py` | 评测端目标比较、计分、分组汇总 |
| `real_scene.py` | 单条场景输入、解析、检查点与推理流程 |
| `active_diagnostic.py` | 有限预算的主动证据控制流程 |
| `computed_evidence.py` | 位置、距离、尺寸证据汇总及候选更新，不调用模型或渲染 |
| `planner_prompt.py` | 外层 planner 的原有 prompt 与能力说明 |
| `v74_grounding.py` | 全场景多模态补充推理 |

`v4_policy`、`v5_runtime`、`v7_evidence` 等名字中带旧版本号的模块仍被当前规则继承，
保留其功能与兼容接口，避免通过删除历史名称意外改变预测。旧设计和运行说明集中在
[docs/history](docs/history)，不再与当前操作指南混排。
生成结果继续放在 `local_outputs`，归档放在工作区 `archives` 及 KESU 的独立归档目录；均不作为源码提交。

## 离线验证

以下命令隔离本机 `.env`，测试中的网络调用使用 mock，不产生付费推理：

```bash
PYTHON_DOTENV_DISABLED=1 SEEGROUND_API_PROVIDER=openai \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=local OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m unittest \
  test_correct_cache test_retry_null test_v5_runtime_batch test_full_validation \
  test_v7_2 test_v7_3 test_v7_4 test_v7_evidence test_frozen_run
```

重构前后的全测试集状态、函数结构一致性和归档验证见
[本次验证记录](../local_diagnostics/refactor_20261015/report.md)。
