# R12 Top-5 冻结快照（2026-10-20）

分支：`codex/r12-top5-freeze-20261020`。快照开始：2026-10-20T23:36:46.327616+08:00。

- 当前 `reproductions/SeeGround/local` 源码、启动脚本和测试直接保存于 Git；正在运行的 148 个模块与启动清单逐个 SHA256 一致。
- 原始 R8 ZIP 和解压输入直接保存于 `outputs/r8_best_object_predictions_20261020.zip` 与 `outputs/r8_grounding_input_20261020/`。
- `audited_challenge.tar.gz` 保存此前已完成的开发、累计结果、冻结源码和审计记录；69.55% 是评估 oracle 累计，完整 Top-1 首轮为 53.43%。
- `top5_started_manifest.json` 保存当前 r12 Top-5 全量单轮的启动配置、7805 条查询、代码及场景资产哈希。
- `top5_in_progress_final_files.tar.gz` 保存快照时 711 条已完成查询的最终预测、运行记录、评估和类别输入；`top5_in_progress_results.json` 为同批评估集合。完整单轮仍在运行，这些部分结果不是最终成绩。
- `freeze_manifest.json` 记录源码、数据、归档及归档内文件的 SHA256，包含源文件追溯限制。

本次 Git 冻结不修改推理代码、不停止当前任务，不在原输出目录写入冻结标志。后续完成的结果不属于本次快照。
场景点云、网格、模型权重与 Python 环境仍在本地或外置盘；启动清单保留场景资产哈希。归档跳过外部符号链接、凭证、缓存和 macOS 元数据。未将其他分类器的大型诊断目录及无关研究素材纳入本次数据快照。

恢复归档：在仓库根目录执行以下命令（进行中的最终文件归档建议解到单独目录，避免覆盖当前任务）：

```bash
tar -xzf outputs/r8_r12_freeze_20261020/audited_challenge.tar.gz -C outputs
mkdir -p /tmp/seeground-r12-top5-snapshot
tar -xzf outputs/r8_r12_freeze_20261020/top5_in_progress_final_files.tar.gz -C /tmp/seeground-r12-top5-snapshot
```

核对归档：`shasum -a 256 outputs/r8_r12_freeze_20261020/*.tar.gz`，与 `freeze_manifest.json` 的 `archives` 字段对照。
