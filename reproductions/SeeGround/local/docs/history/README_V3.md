# Relative size v3

实现目录：`/Volumes/KESU/论文/.worktrees/relative-size-evidence-v3/reproductions/SeeGround`。
分支：`codex/relative-size-evidence-v3`。从当前 v2 源码复制后独立修改，v2 工作区保留。

## 行为与边界

- 取证正常结束且原输出为空时，最终候选恰好一个才输出该 ID，标记 `prediction_source=singleton_fallback`。继续保留原 `decision_status`、缺失条件、停止原因；fallback 的 `accepted=false`。不提前结束取证，错误和人工中断不补答案。
- `prediction_source=model` 表示原有决策输出，`abstained` 表示仍为空。`accepted` 表示严格证据认证。`answer_coverage`、`answered_accuracy` 与 `accepted_coverage`、`accepted_accuracy` 分别统计。
- 默认 `SEEGROUND_SYNTAX_BACKEND=spacy`。英语依存分析给现有语言模型提供词组、数量、名词、介词附着、从句及完整原文跨度；语法结果本身不证明场景对象或几何条件。
- 结构化尺寸条件、查询重写与条件审计复用比较信息。只有完整条件已表示，才消解重复的 `of the two` / `two doors` 等 Unhandled 记录。两者比较的成员不足两个或多于两个时，范围仍 unknown。
- 对明确修饰比较集合的 above/below 从句，使用现有垂直关系谓词绑定子集并冻结成员；保留未确定成员。其他未支持关系、`than` 的场景参照比较、歧义附着、否定和复杂并列继续保留待处理条件。单独的独立位置条件不改变比较集合。
- 仅完整简单请求中的 `in the room` 按所提供 NR3D 场景解释，并记录 `scene_scope_assumption`。命名子房间如 `in the bedroom` 不按全场景消解；用于多房间数据时需要另行实现房间绑定。
- spaCy 和 Stanza 已用 9 个代表句作依存结果比较，选择 spaCy 的小型英语模型作为运行依赖。两者都有附着误差；v3 保留有疑问的解释。Stanza 仅用于开发比较，无需生产安装。
- 未修改尺寸阈值或后续测量策略。运行记录与缓存 manifest 记录 v3、语法后端、模型版本和代码哈希，后端/代码改变须用新输出目录。

## 环境

当前 `.venv` 已配置，实际环境在本机 SSD 缓存，避免外置盘上安装大量小文件。新机器使用 Python 3.12 建立虚拟环境后安装：

```bash
cd /Volumes/KESU/论文/.worktrees/relative-size-evidence-v3/reproductions/SeeGround
python3 -m venv .venv
.venv/bin/python -m pip install -r local/requirements-lock.txt
```

锁定 spaCy 3.8.16 与 `en_core_web_sm` 3.8.0（模型 wheel 含 SHA256）。推理期间不自动下载模型。

## 离线验收结果

`spacy` 与 `none` 后端各 603 项测试通过；网络请求与外部子进程在回归期间被阻断。

固定 186 条已保存记录的 singleton-only 策略回放：原 15 条非空预测不变，新增 41 条中 37 对、4 错；总非空 56、正确 52、null 130；输出覆盖率 30.1%、全样本准确率 28.0%、输出答案准确率 92.9%。严格认证仍 15 条，准确率 100%。错误 fallback：8614、11901、12579、26783，如实计错。

五条保存解析＋场景几何的独立语法回放：7764、30152、7755、4091、29515 从 missing 变为 resolved，候选 ID 分别仍为 13、25、34、17、9，均与评估答案一致。未重新调用解析模型，未复用视觉判定；这不能代替完整主动取证重跑。

以上是同一批观察数据的离线回放，不是新的 300 条真实推理成绩。详细结果见 `local_outputs/v3_validation/saved_replay.json`；依存对比见 `syntax_comparison.json`；回归日志为 `regression_none.log` 和 `regression_spacy.log`。

重放命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=local .venv/bin/python local/replay_v3_validation.py \
  --snapshot /Volumes/KESU/论文/outputs/relative_size_v2_diagnosis_20261011_174935/results_snapshot.json \
  --saved-root /Volumes/KESU/论文/.worktrees/relative-size-evidence-v2/reproductions/SeeGround/local_outputs/relative_size_20261011_162853_v2_gpt_full300 \
  --scene-root /Volumes/KESU/论文/.worktrees/relative-size-evidence-v2/reproductions/SeeGround/local_outputs \
  --output local_outputs/v3_validation/saved_replay.json
```

## 后续真实验收命令

以下命令会调用已配置 API。本次开发没有启动真实推理。统一使用 `--workers 10`，读取 v2 的已准备场景，输出写入 v3 新目录。

先加载原有配置并选择后端：

```bash
cd /Volumes/KESU/论文/.worktrees/relative-size-evidence-v3/reproductions/SeeGround
set -a
source /Users/luocl/.config/seeground/relative-size-v2-relay.env
set +a
export SEEGROUND_SYNTAX_BACKEND=spacy
```

先验收 dev10：

```bash
.venv/bin/python local/run_module_acceptance.py \
  --manifest local/validation/relative_size_dev_10_20261010.json \
  --scene-root /Volumes/KESU/论文/.worktrees/relative-size-evidence-v2/reproductions/SeeGround/local_outputs \
  --output "local_outputs/relative_size_$(date +%Y%m%d_%H%M%S)_v3_spacy_dev10" \
  --review-limit 5 --workers 10
```

小批验收通过后，固定 300：

```bash
.venv/bin/python local/run_size_validation.py \
  --manifest local/validation/nr3d_size_300.json \
  --scene-root /Volumes/KESU/论文/.worktrees/relative-size-evidence-v2/reproductions/SeeGround/local_outputs \
  --output "local_outputs/relative_size_$(date +%Y%m%d_%H%M%S)_v3_spacy_full300" \
  --review-limit 3 --workers 10
```

对照组将环境变量改为 `export SEEGROUND_SYNTAX_BACKEND=none`，将输出名中的 `spacy` 改为 `singleton_only`，使用同一 manifest、review-limit 与 `--workers 10`。该组保留 singleton 输出并关闭语法改进，供独立归因；真实模型调用仍可能存在随机差异。最后用未参与诊断的样本验证泛化。
