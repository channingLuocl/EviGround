# R18 Top-5 visual-fallback development snapshot — 2026-10-22

本次只定格当前代码与数据，不修改定位算法，不重新启动推理。

- Branch: `codex/r18-visual-fallback-snapshot-20261022`
- Base: `fcab59e289f68a3d2b10a7d34733c2e80947e2b0`。旧冻结分支保持原提交。
- 当前 7 个未提交的实验代码改动保存到此分支；源代码 SHA-256 见 `freeze_manifest.json`。
- 使用预测类别 Top-5，不将 GT 类别提供给 grounding，不复用历史正确答案。GT 仅用于评测；归档的结果文件可含评测标签。
- R8 预测输入的 140 个文件已经由父提交纳入 Git，且与本地可用导出逐一核对一致。本次继承原始数据，无需重复上传。
- 三批开发实验的原始日志、预测、错误现场、图像、运行配置及选择清单打包于 `experiments.tar.gz.part*`。每个文件的哈希见 `experiment_files.json`；排除临时 API 锁、符号链接及 `.DS_Store`。
- 凭据和 `.env` 不纳入版本。场景点云仍在外置盘，未上传 Git；对应运行清单保留场景文件哈希。

## 已有结果的正确解释

| 实验 | 实际 revision | 已记录/选择 | 正确数 | 备注 |
| --- | --- | --- | --- | --- |
| r18_extreme_dev32_20261022 | r18 | 5/32 | 3 | 提前停止，不能报告为完整 32 条得分 |
| r18_extreme_fallbackguard_dev32_20261022 | r1 | 32/32 | 11 | 配置误用 r1，不能用于判断 R18 guard |
| r18_extreme_fallbackguard_r18_dev32_20261022 | r18 | 32/32 | 14 | Overall 43.75%；16 条格式/schema 失败计入分母 |

本版本尚未完成全部 7,805 条单轮验证。归档实验并非全部由本次最终源码或相同 thinking 配置产生，应以各自 `manifest.json` 为准。当前结果不代表达到 65%～66% 目标，也不能与累计保留历史正确结果的 v7.4 指标直接比较。

## 恢复数据

在仓库根目录运行（恢复前确认同名输出目录没有需要保留的新内容；建议在新 clone 中恢复）：

```bash
cat outputs/r18_visual_fallback_snapshot_20261022/experiments.tar.gz.part* | tar -xz
```

预测输入在 Git 中的路径：

```text
outputs/r8_grounding_input_20261020/20261020_r8_object_predictions
```

如使用 sparse checkout，可在新 clone 的仓库根目录物化该路径：

```bash
git restore --source=HEAD --worktree --ignore-skip-worktree-bits -- outputs/r8_grounding_input_20261020/20261020_r8_object_predictions
```

## 校验

在仓库根目录运行：

```bash
shasum -a 256 -c outputs/r18_visual_fallback_snapshot_20261022/SHA256SUMS
cd reproductions/SeeGround
PYTHONPATH=local .venv/bin/python -m unittest local.test_r8_grounding local.test_r18_react local.test_gpt61_action_format
```

本次定格验证：47 项测试通过；分片和归档内容逐项哈希核验。`.venv` 及 API 凭据需要在运行机器上单独配置。
