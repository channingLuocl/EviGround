# R18 Top-5 当前代码定格（2026-10-22）

冻结分支：`codex/r18-json-top5-freeze-20261022`。

本快照以提交 `89ec46ba66113d6c27c752bc0f8480b7dd39b4a9` 为父提交，包含当前全部已跟踪的工程源码，以及 `1e3d7135b43528b78561b60d27275df681ebbc9b` 中的 ReAct JSON 动作接口修正。之前的冻结分支 `codex/r18-top5-single-pass-freeze-20261022` 和提交 `cfe9623cbe7c2e3fcc28b7fa9b22d18165178f6f` 保留。

## 已保存的内容

- 工程源码：`reproductions/SeeGround/local/`，完整文件清单、SHA256 和 Git blob ID 见 `freeze_manifest.json`。
- 预测类别数据：`outputs/r8_grounding_input_20261020/20261020_r8_object_predictions/`，140 个文件已在本分支的 Git 历史中，逐个核验与本机实际运行输入 `outputs/20261020_r8_object_predictions/` 一致，保留 Top-1、Top-5 和分数。
- 全量评测清单：`reproductions/SeeGround/local/validation/nr3d_val_full.json`，共 7,805 条、130 个场景。
- 当前固定运行副本 `runtime_1e3d713` 的 245 个 Python 文件逐个匹配当前 checkout 和 Git。运行副本可以从本分支源码重建，无需重复提交。
- 23 项 ReAct 单元测试通过。此次仅新增冻结说明与校验清单。

## 评测口径与复现边界

本版本显示类别为预测 Top-1，并允许使用预测 Top-5；完整单轮不累计历史正确答案，错误和弃答计入分母。

本快照是代码和输入数据定格，不是全量成绩认证。先前 full7805 任务使用旧的 `runtime_eb261ce`，其结果不能归为此处的新源码。已有五条 JSON 接口压力测试也不能代表 7,805 条准确率。相关原始记录位于 `outputs/r18_json_action_contract_20261022/`，包含服务故障说明。

外置盘原始场景资产未上传 Git；既有资产路径和哈希保留在清单中。密钥与外部私有配置不提交。复现仍需准备工程依赖、对应场景资产和凭据。旧的 `audit_single_pass.py` 固定核验旧源码，不能直接作为本次新源码通过全量审计的证明。

此次不启动、停止或重新运行评测任务，也不修改其固定源码和输出。
