# v7.3 修复与验证记录

后续用户将复用规则改为“任意历史运行曾答对就永久保留”。当前已建立
KESU 上 `.correct_results_gt.json` 的 233 条正确缓存，向当前目录补入 21 条、
替换错误/null 11 条，替换前的评估归档至 `superseded_by_correct_cache`。
当前汇总是 334 条已完成、233 条累计正确；以下 300 条原样复用的描述为
实施正确优先策略之前的验证记录。当前统计标注
`evaluation_policy=cumulative_ever_correct`，不代表单个版本独立准确率。
缓存跨代码版本生效，GT 和预测类别分开保存，新版本用新输出目录自动继承。
新增测试覆盖历史正确恢复、错误归档、重试跳过、0 号物体、历史重试记录、
样本身份校验及两个运行互不丢失正确缓存。

本次修复 planner 共享 JSON schema 与分维度执行器之间的字段冲突：
appearance 校验后忽略 inspect；size/position/distance 校验后忽略 context；
viewpoint 校验后移除不执行的视觉字段。保留类型、编号、候选范围和视角校验，
不修改证据不足时的拒答规则。planner 提示与 schema 描述同步更新。

## 验证

- `test_v7_3`、`test_v7_2`、`test_v7_evidence`、`test_retry_null`、
  `test_v5_runtime_batch`、`test_full_validation`：55 项，54 项通过、1 项跳过。
- 最后一次复用日志调整后，重跑 `test_v7_3`、`test_retry_null`、
  `test_v5_runtime_batch`：28 项，27 项通过、1 项跳过。
- Shell 语法检查及 `git diff --check` 通过。
- 旧日志的 144 次参数拒绝回放：135 次在修复后的校验器中有效。
  剩下 9 次来自 q00118，其 focus_ids=[14,28] 不含剩余候选 [4,5]，仍拒绝。
  这是请求校验回放，不代表 135 条查询的定位结果会改善。
- 最终版联网短测 q00012、q00020 各成功执行 3 个证据工具，拒绝动作均为 0；
  两条最终均为 null。缺少对复杂方向、参考物体和 grouped together 描述的完整
  条件绑定。不能以此次参数修复宣称全量 null 已解决或准确率已经提高。

曾运行包含 `test_active_diagnostic` 和 `test_v62_structured_outputs` 的较旧综合套件：
125 项中有 20 处失败、38 处错误。输出包括手写 parsed 夹具缺少 query_structure、
旧的 API 不重试/不修复预期与当前实现不一致。这些失败未在本次范围内修复，
上述通过数字仅指列出的相关套件。初次参数修复的联网轨迹保留在 KESU 的
`local_outputs/v7_3_contract_smoke_initial_20261014`，未导入最终运行目录。

## 复用与续跑

原始 300 条 GT 结果全部保持原评估字段和来源文件哈希，200 条正确。
错误、null 和 API 失败结果也原样导入，不选择性保留正确项。
旧全量新增的 19 条未导入；最终版两条短测已缓存，可直接续跑其余 7,503 条。
原文件及图片轨迹留在源目录；新结果保留 v7.1/v7.3 来源，并分别统计版本成绩。
这是显式跨版本实验，不是推理等价的缓存复用。

```bash
cd /Users/luocl/Desktop/seeground/reproductions/SeeGround
local/run_local_v7_3_full.sh
```

GT 全量清单共 7,805 条，输出固定为 KESU 上的
`local_outputs/v7_3_nr3d_val_full_gt`。中断后重复同一命令。
`--prepare-only` 可在不发起推理的情况下重新检查资源、设置和缓存。
续跑日志区分 newly imported 与 already cached，避免将 0 条新导入误读成未复用。

详细机器可读记录为 `v7_3_contract_replay.json` 和 `v7_3_verification.json`。
