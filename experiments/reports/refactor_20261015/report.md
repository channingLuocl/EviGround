# v7.4 冻结与结构重构验证

日期：2026-10-15。重构前代码和结果已独立保存；本次没有发起新的模型推理。

## 结果归档

- 冻结成绩：**5435 / 7805 = 69.63%**，GT 类别、跨版本累计正确策略。
- [完整归档](/Volumes/KESU/论文/paper/reproductions/SeeGround/archives/v7.4_20261015_6963/README.md)：26 个来源批次、**68,683 个文件**，逻辑大小 8,260,968,095 字节，压缩后 4,721,254,713 字节（约 4.40 GiB）。
- 原始目录未删除；日志、API 记录、图片、轨迹、检查点、所有现存重试和来源批次均包含在归档中。
- [电脑本地关键备份](/Users/luocl/Desktop/seeground/archives/v7.4_20261015_6963/README.md)：完整最终评估汇总、正确缓存、运行 manifest、查询清单、重构前代码及 Git 历史。与外盘对应的 8 个关键文件逐一校验一致。
- 原始 7,805 条单独评估均存在于完整归档。原批次 4 条推理错误没有生成 prediction.json；它们的评估和 console.log 已保留，并非归档丢失。
- 打包时对源文件内容计算 SHA-256，再逐文件读取压缩包校验；校验清单保存在归档的 `verification.json`。代码的 Git bundle 亦通过 `git bundle verify`。

完整压缩包 SHA-256：

```text
be96c573753b5051ae59ce199907b75cce0821dd070e52e861e5e48453bdc88f
```

原 results.json 在重构前后均为：

```text
9fd6b1cc86a9ff3001531b1f36d232a90dcac2e9b47f7ac046e23e38971e38a8
```

## 代码变化及行为边界

批处理从 639 行减为 307 行，按职责拆到 `validation_cache`、`validation_reuse`、
`validation_reporting` 和 `run_storage`；历史哈希例外离开主调度文件，旧导入接口仍保留。

主动控制器从 2,219 行减为 1,748 行，确定性的证据计算迁到 `computed_evidence`，
原有 prompt 与能力常量迁到 `planner_prompt`。历史设计文档集中到 `local/docs/history`，
当前说明统一到 `local/README.md`。

没有删除仍被 v7.4 继承的尺寸、关系、视觉证据和恢复逻辑；没有改 GT/预测类别策略、
模型请求参数、候选筛选、推理预算或历史正确缓存策略。

新增 `FROZEN.json` 检查，覆盖当前全量 CLI、重试、汇总、缓存导入、单条推理和 API
并发锁目录，包括子目录和符号链接别名。实测旧 v7.4 命令在准备资源前拒绝写入。
该检查防止这些程序入口误写；原始文件及独立备份都仍保留。

新入口 `local/run_local_full.sh` 使用独立的 `v7_4_refactored_nr3d_val_full_gt` 目录，
默认本地并发 10、API 并发 8。使用临时 Python stub 核对了真实启动参数，未调用服务。

## 验证

| 检查 | 结果 |
|---|---|
| 原主动推理与证据函数 AST 对照 | 26 个函数体全部一致 |
| 原 planner prompt/常量 AST 对照 | 5 项全部一致 |
| real_scene、qwen_api 的函数及批处理 main | 除新增冻结检查外，函数体一致 |
| 当前缓存、重试、全量选择、v7.2—v7.4 与冻结保护回归 | **74 项：73 通过、1 跳过、0 失败** |
| 所有本地 Python 编译及启动脚本语法 | 通过 |
| Git diff 空白检查 | 通过 |
| 原结果文件 SHA-256 | 未变化 |

全测试集尚未全部通过：在隔离 `.env` 的相同环境下，重构前 780 项含 **53 个失败、24 个错误、4 个跳过**；
重构后 785 项仍为相同的 53 个失败、24 个错误、4 个跳过。逐项对比问题名称，没有新增失败或错误。
旧测试涉及已演变的证据契约、结构化输出重试和旧 GPT 参数预期；本次没有为使旧测试通过而改变 v7.4 推理行为。

额外将 API 测试并入当前回归的 93 项检查中，有 2 个原有 GPT 参数断言失败：
旧断言不接受 `enable_thinking`，而当前客户端原本就发送 `enable_thinking=false`。
这两个问题亦在重构前全测试集问题列表中，不是本次引入。

原始输出见本目录的 `tests_before_isolated.log`、`tests_after_isolated.log`、
`tests_current_after.log`、`tests_api_and_current.log` 和 `frozen_entrypoint.log`。
机器可读结论见 `summary.json`、`structure_verification.json`、`archive_verification.json`。
