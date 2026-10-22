# R18 ReAct context and budget development — 2026-10-22

目标仍为预测类别 Top-5、无 GT 类别输入、不累计历史正确答案，全部 7,805 条单轮 Overall 达到 65%～66% 以上。本目录记录开发过程，不表示已经达到目标。

## 已实施的修改

源码提交：`eb261cea6f4e91afa9dd01ee523303efaa58b73f`，分支 `codex/r18-react-context-budget-fix-20261022`。

- 测量结果向模型以列名与数值行传递，省去重复字段；完整原始证据仍保留在轨迹中。同一参照系下被后续完整测量覆盖的旧测量不再重复发送。
- 每次最多附带概览与最近三张图片；全部图片保留，可再次 inspect 旧对象。
- 模型取证、controller 补证和提醒分别计数，提醒不占取证预算。参照物和 fallback 的补证需求一并处理。
- 预算耗尽时，system、state 和 schema 一致要求最终选择；不将 inspect 改写成 finish，不编造答案或证据。
- inspect/measure 必须包含明确且非空的 focus_ids；纯粹“我要看场景”的文字不能冒充有效动作。格式问题由本地校验与格式重试处理。
- 同一测量必须覆盖整个比较池；几次互不相交的测量不能冒充完成全部两两比较。额度不足不会作为临时限流重试。

56 项单元测试通过。选取旧失败轨迹做离线编码比较，测量文本由 132,266 字符降至 52,018 字符，原始数值完整保留。这是文本大小改善，不是服务端截断已获证实。

扩展回归也做了隔离比较：冻结版本执行 101 项，当前执行 110 项；两者均有相同的 12 处失败和 3 处错误，另有 1 项跳过，本次未新增失败检查。旧测试包含远端 `json_schema`、温度参数和重试次数等既有接口契约，与当前中转配置存在差异，不能声称扩展套件全通过。详细对比见 `compatibility_test_comparison.json`。新增的 R18 测试显式固定 provider，避免依赖工程 `.env`。

## 运行记录

- `smoke8/`：启动时密钥未导出，8 条均在 `_settings()` 校验失败，尚未发出推理请求；保留现场。
- `smoke8_exported/`：重新配置后，在新目录完整执行 8 条固定压力案例。3/8 正确，8/8 有预测，0 致命格式错误，实际候选召回 8/8。该集合集中于旧失败案例，不能估计总体准确率。逐项审计见 `smoke8_audit.json`。
- `dev32/`：当前固定源码完整重新执行 32 条，26/32 正确（81.25%），32 条均有预测，致命错误和格式重试均为 0。本轮所有结果单独保存，没有混入旧正确结果；审计见 `dev32_audit.json`。实际候选召回 32/32，纯分类候选召回 29/32。
- `fresh_validation192/`：固定随机种子 20261022，从官方 7,805 条清单排除旧开发 96 条和旧验证 192 条后抽取 192 条。选择清单在任何新验证结果生成前写入，不按正确与否挑选样本。
- 验证任务使用 `runtime_eb261ce/` 内的源码副本，逐个核验哈希，后续开发不会改变正在执行的验证源码。运行配置和实际代码哈希以各运行目录的 `manifest.json` 为准。

8 条压力测试的最大请求仍报告 93,742 prompt tokens，最多附带 4 张图。文本压缩和图片上限不足以证明中转服务已经稳定；后续 32、192、7,805 条都必须保留失败分母。剩余错误包括视觉类别判断、复合限定词和观察者参照系，尚待进一步分析。

完整开发 32 条的 149 个推理源码哈希均匹配启动清单，逐条核验输出未提供 GT 类别或 query target ID；实际模型取证动作最多 4 次（基础 3 次加复核保留 1 次），controller 补证最多 2 次。与旧 14/32、16 条格式失败的 R18 实验相比，代码和 token ceiling 同时变化，不能将差异单独归因于某一项修改。开发集 81.25% 不能替代独立 192 条或全量 7,805 条成绩，65%～66% 的全量目标仍未验证。

## 启动方式

密钥只从 `/Users/luocl/.config/seeground/gpt61-relay.env` 导出，不写入本目录。启动器绕过工程 `.env` 的覆盖，强制固定 GPT 中转、R18、Top-5、JSON object + 本地 schema 校验、8192 token floor。

本地 `.env` 只记录 `SEEGROUND_API_PROFILE` 路径，不保存密钥。旧 `run_local_v9.sh` 也支持读取这一外部 profile；运行中的 Python 推理源码哈希保持不变。

在工程目录外也可启动；每次使用全新输出目录：

```bash
bash /Users/luocl/Desktop/seeground/outputs/r18_context_budget_fix_20261022/run_gpt61_r18.sh \
  --manifest /Users/luocl/Desktop/seeground/outputs/r18_extreme_dev32_20261022_manifest.json \
  --output /Users/luocl/Desktop/seeground/outputs/r18_context_budget_fix_20261022/dev32
```

## 2026-10-22 版本定格

冻结分支：`codex/r18-top5-single-pass-freeze-20261022`。`freeze_manifest.json` 固定源码、预测数据、全量清单和运行配置的 SHA256；预测数据 140 个文件已包含在 Git 的 `outputs/r8_grounding_input_20261020/20261020_r8_object_predictions/`，与本机运行输入逐个核对，不重复提交同一份数据。外置盘的场景资产以启动清单中的 `scene_sha256` 固定，原始点云和运行中持续产生的图片不在此次 Git 快照中。

开发 32 条完整通过稳定性检查后，全量 7,805 条已与独立验证 192 条并行启动，使用相同的 149 个推理源码哈希。全量输出为 `/Volumes/KESU/论文/paper/reproductions/SeeGround/local_outputs/v74_r18_gpt61_context_budget_full7805_20261022`。`full7805_started_manifest.json` 保存实际启动配置；`checkpoint_at_freeze.json` 仅记录定格时进度，不是最终成绩。现有任务继续运行，不要重复启动。任何开发集、部分结果或成功子集都不能替代 7,805 条 Overall。

全量结束后运行审计（审计不会调用模型）：

```bash
/Users/luocl/Desktop/seeground/reproductions/SeeGround/.venv/bin/python \
  /Users/luocl/Desktop/seeground/outputs/r18_context_budget_fix_20261022/audit_single_pass.py \
  --run /Volumes/KESU/论文/paper/reproductions/SeeGround/local_outputs/v74_r18_gpt61_context_budget_full7805_20261022 \
  --runtime /Users/luocl/Desktop/seeground/outputs/r18_context_budget_fix_20261022/runtime_eb261ce/local \
  --output /Users/luocl/Desktop/seeground/outputs/r18_context_budget_fix_20261022/full7805_audit.json
```

`verified_complete` 只有在全部 7,805 条记录齐全且结构检查通过时才为 true；失败仍留在分母。65%～66% 的全量目标尚未验证。

从本冻结分支的完整 checkout 复现时，先将 canonical 预测目录恢复到启动器使用的位置：

```bash
cd /Users/luocl/Desktop/seeground
cp -R -n outputs/r8_grounding_input_20261020/20261020_r8_object_predictions outputs/
```

须另备外置盘场景数据、工程依赖和外部凭据 profile。启动器默认使用 checkout 的 `reproductions/SeeGround` 源码，源码已与当前固定运行副本逐个比对。下面仅为将来复现使用，输出必须换成不存在的新目录：

```bash
bash /Users/luocl/Desktop/seeground/outputs/r18_context_budget_fix_20261022/run_gpt61_r18.sh \
  --manifest /Users/luocl/Desktop/seeground/reproductions/SeeGround/local/validation/nr3d_val_full.json \
  --workers 8 --api-concurrency 8 \
  --output /Volumes/KESU/论文/paper/reproductions/SeeGround/local_outputs/r18_top5_fresh_reproduction
```

审计器默认核验 checkout 源码；使用固定副本时显式传入 `--runtime`。四项审计测试覆盖不完整样本、失败分母、伪造正确标记和 GT 输入标记。
