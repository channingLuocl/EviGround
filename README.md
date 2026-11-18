# EviGround

**ReAct-Based Visual–Geometric Reasoning for 3D Visual Grounding**

EviGround 根据语言描述定位 3D 场景中的物体。输入是点云、实例几何和外部分类器预测；大模型自主选择 `inspect`（观察）、`measure`（测量）和 `finish`（作出选择）。

## 安装

需要 Python >= 3.10。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

默认通过 `http://127.0.0.1:18000/v1` 调用 Qwen2-VL-72B 的 OpenAI 兼容接口；先用 ssh-skill 将本地 18000 端口转发到 AutoDL 的 8000 端口。无认证的本地 vLLM 使用占位密钥 `EMPTY`。可在 `.env` 覆盖接口与模型。

ScanRefer 原始点云预处理及完整离线测试还需要 `pip install -e '.[scanrefer]'`。Mask3D 和 r8 分类器的 GPU 环境、权重及冻结外部源码需另行准备，不包含在该可选依赖中。

## 运行

```bash
eviground \
  --predictions /absolute/path/to/classifier_predictions.zip \
  --scene-root /absolute/path/to/prepared_scenes \
  --category-top-k 5 \
  --workers 8 --api-concurrency 8 \
  --output outputs/top5_single_pass
```

- `--predictions` 必须显式指定，支持已校验的分类导出 ZIP 或目录。
- `--category-top-k 1` 只使用第一类别；`5` 保留五个候选类别，保留排名和分差，结合视觉与空间证据判断。
- 默认评估 `data/queries/nr3d_val.json` 中的 7805 条描述。用 `--limit 10 --prepare-only` 检查输入，不调用模型。
- 同一输出目录、相同代码和配置可继续未完成任务；已完成的正确、错误和 null 都原样保留。新实验必须使用新输出目录。
- 单轮结果保存在 `results.json`；错误和 null 计入总分母。没有跨运行正确结果合并、累计命中或 GT 类别推理入口。

本机调用 AutoDL 上当前服务的模型（先在服务器切换到对应模型）：

```bash
bash scripts/run_qwen2_vl.sh 72b \
  --predictions /absolute/path/to/classifier_predictions.zip \
  --scene-root /absolute/path/to/prepared_scenes \
  --category-top-k 5 --review-limit 10 --workers 8 --api-concurrency 2 \
  --output outputs/qwen2_vl_72b_full
```

2B 使用 `2b` 和独立输出目录。两模型顺序运行，各使用全部 7,805 条描述。Qwen2-VL 启动脚本默认完整开启 Monitor/DAG、连续策略退火和 Top-5，按成功工具调用计 10 轮取证，另有 6 次修正额度，最多 17 次动作决策；每次动作上限 4,000 token。旧云模型的 thinking 参数关闭，输出参数使用 vLLM 的 `max_tokens`。`api-concurrency=2` 与当前单卡服务的并发容量一致。历史 `run_eviground_local.sh` 仍用于原冻结实验。

本地模型启动脚本将 `EVIGROUND_API_READ_TIMEOUT` 设为 900 秒，保留 15 秒连接超时。单卡 72B 的长 DAG 请求可能超过原 120 秒读取上限；过早重试会增加排队。该设置进入运行清单，显式传入的请求超时优先。

Qwen2-VL 启动脚本默认使用 `EVIGROUND_PROMPT_PROFILE=compact`，并记录在运行清单中；直接 CLI 的默认 `standard` 保留原提示。适配版缩短指令、强调空间关系需要比较测量，并将重复的测量字段编码为无损表格。两种配置均使用严格 JSON 结构；非空文本的正则只在客户端校验，以规避当前 XGrammar 通配符可能越过 JSON 字符串边界的问题。服务应使用 `--generation-config vllm`，避免权重自带的 `max_new_tokens=512` 覆盖应用请求。提示或服务配置改变后必须使用独立输出目录。

Qwen2-VL 脚本同时开启 `EVIGROUND_ACTION_NO_REPEAT=1` 和 `EVIGROUND_GUIDED_DECODING_BACKEND=guidance`；服务端须以 `--guided-decoding-backend guidance` 启动。已成功取得的相同观察会在 token 解码时被禁选，ID 顺序、默认相机角度及理由／DAG 改写不能绕过；不同对象组、视角、上下文或测量参考系仍可选。完整原生 Monitor 子语法、客户端 schema 校验及最终核验保留。该开关使用 compact 动作分支；不支持的后端直接报错，不静默退回无约束生成。CPU 禁选验证通过不等于 Monitor 能推进或准确率提高，必须另做独立小批推理验收。

`compact` 可显式设置 `EVIGROUND_DIALOGUE_HISTORY=1` 试验原生多轮消息：历史动作作为 assistant 消息保留，新增观测/图像与最新完整 DAG、证据目录和预算作为后续 user 消息发送。初始场景只发送一次，全部历史仍在消息链中；内部 DAG 校验与评分不变。默认关闭，开关进入运行清单。v18 的 2B/72B 诊断仍出现重复 pending 计划，72B 的历史上下文明显增长；该试验未通过可靠状态推进验收，也没有准确率提升证据，暂不用于新全量。

合成协议诊断发现72B可选对简单高度目标，却保留错误当前节点及未完成verification，因而被Monitor拒绝。补强非空finish解码约束的候选通过离线校验，但在线短测出现任务膨胀、弃答且未通过验收，已回退，不用于新全量。2B在有效合成渲染后仍重复pending计划。原始报告与审计保留于`outputs/qwen2_vl_20261116/improvement/`；这些控制题不构成Nr3D或ScanRefer成绩。

实验开关 `EVIGROUND_DAG_WIRE_FORMAT=delta` 仅适用于 compact + DAG Monitor：首轮仍生成完整图，后续只输出明确变更的节点状态；结构修改使用 `replace_plan` 提交完整新图。适配器展开为完整内部图，再执行原证据、依赖、回滚、候选恢复及最终核验检查，不自动完成节点或补写结论。输入仍保留完整图和观测。默认 `snapshot`，开关进入运行清单，必须使用独立输出目录；离线等价测试不代表模型精度改善。

2026-11-17 的 delta 实验通过本地/远程各 223 项测试及两个模型 tokenizer 的 XGrammar 路径检查，但固定高度/距离控制均未通过有界完成验收（每题先种入一次 measure，最多两次真实后续请求）。72B 高度题选对 ID2 后仍缺少最终核验或排除，距离题弃答；2B 出现重复请求、同时排除两个候选及状态/动作冲突。该开关保持默认关闭，不用于新全量，不能将这些诊断当作数据集准确率。

后续仅增加一个无关的完整完成示范，候选本地/远程各224项测试通过，但在线控制仍失败：72B两题均选对目标，却未完成最终verification；2B在最终节点请求inspect，且混淆位置/尺寸。额外示范已回退，回退后223项测试通过；原始响应、92个冻结源文件和CPU重放审计保留于`outputs/qwen2_vl_20261116/improvement/protocol-example-20261117`及对应归档。Nr3D/ScanRefer的新12条清单仅通过输入预检，未进行模型推理；无新全量，实例已恢复无卡模式，仍无准确率改善结论。

2026-11-17后续诊断将动作/状态约束提前到delta解码，仍出现2B状态冲突及72B弃答。CPU重放确认先生成monitor的n0更新会提前剪掉非空finish分支。字段顺序候选将action、predicted_id放到monitor前；本地/远程各227项测试、22条完整token路径及8条前缀检查通过，但两模型有界在线控制仍因重复工具和completed当前节点请求工具而失败。新的pending当前节点checkpoint候选本地229项测试通过，正进行无卡CPU检查，尚未在线验证；原Monitor与ground流程不变。delta继续默认关闭；未获得Nr3D/ScanRefer精度改善证据，无新全量。完整记录见`outputs/qwen2_vl_20261116/improvement/诊断与改进.md`。

2026-11-17 06:56 最新核对：另行试验同一模型先评估 Monitor、再选动作，两阶段仍共享 4,000-token 决策额度。固定 4 条真实诊断再次启动 2 条、完整评估 0 条，因重复相同提案止损；这一顺序变化未解决状态推进，未推广至默认流程。240 项本地／远程单测、隔离协议回归和两套 tokenizer 的 20 项解码检查通过，但不构成准确率提升证据。compact 列表边界及此前属性归属、相机范围、等效取证修复保留。未恢复全量，实例已实际切回无卡模式，权重与环境保留。详见`outputs/qwen2_vl_20261116/improvement/进一步诊断与定向修复-20261117.md`。

动作请求每次最多包含 8 个对象，较大的测量组需分次请求，以限制成对测量的二次增长。`compact` 还使用按动作区分的结构约束：检查/测量时答案为空，非空最终答案必须附证据，最终动作不携带相机或检查参数。工具调用总预算仍由 `--review-limit` 控制；这些是新实验的接口限制，不能与旧版本结果混合。

2026-11-17 07:35 最新核对：证据分析→原协议的真实诊断仍因重复取证和状态回退止损，启动 2 条、完整评估 0 条，不能报告准确率。另修复 entrance door(s) 未检索到 door 假设的缺口：真实输入从空集合恢复 7 个待核验候选，本地／远程各 241 项单测通过，独立修复版已上传。没有改动原 Monitor 语义、预算、Top-5 或旧结果；原全量仍为 10/33，未恢复新全量，实例已核实无卡／服务停止。原实时查看入口已更新。详见 `outputs/qwen2_vl_20261116/improvement/证据推理诊断与入口检索修复-20261117.md`。

2026-11-17 08:01 最新核对：仅放宽服务端为 JSON 语法的原生传输对照触发技术止损，6 次 HTTP 尝试／5 个完整响应、0 个合格原生动作、0 个完整预测；该候选未进入默认流程。另修复初始俯视图隐藏全部补充实例几何的问题，保留无类别标签的点与数字 ID，不改变 Monitor、预算或评分。本地／远程各 242 项单测通过，真实输入 CPU 回放确认恢复 6 个补充实例；尚无修复后 GPU 精度收益。原全量暂停于 10/33，实例已恢复无卡／服务停止。详见 `outputs/qwen2_vl_20261116/improvement/JSON传输诊断与场景上下文修复-20261117.md`。

2026-11-17 08:44 最新核对：函数标记接口独立候选通过 242 项核心测试、4 项适配测试和两套 tokenizer 共 12 项路径检查；真实固定四条开发诊断中，72B／2B 分别返回 7／5 个 schema 合格动作，但都因连续重复取证止损，完整预测均为 0，准确率不可计算。原始 HTTP 请求确认新增图像、观测和最新 DAG 已传入，尚未恢复可靠状态推进。该候选未推广至默认流程，未启动新全量；188 个证据文件已归档，服务停止，实例已确认无卡。详见 `outputs/qwen2_vl_20261116/improvement/函数接口试跑与当前阻塞-20261117.md`。

## 目录

```text
src/eviground/   当前方法、类别适配、渲染与评估
data/queries/   固定评估描述；答案只供评估进程使用
docs/           使用说明与方法说明
tests/unit/     当前流程的离线回归测试
scripts/        命令启动脚本
```

详细说明见 [运行指南](docs/usage.md) 和 [方法结构](docs/architecture.md)。外部模型、分类器训练代码和旧实验结果不随本仓库分发。依赖及来源说明见 [NOTICE](NOTICE)；许可证见 [LICENSE](LICENSE)。

## ScanRefer 与消融实验

- ScanRefer 清单转换与评分：`scripts/scanrefer.py`；主推理仍使用 `eviground --manifest ...`，预测实例按目标 GT 框的 3D IoU 评分，报告 Acc@0.25 / Acc@0.5。
- ScanRefer 原始输入、预测实例、分类导出、策略与审计脚本位于 `scripts/prepare_scanrefer_*.py`、`scripts/classify_scanrefer_r8.py` 和 `scripts/audit_scanrefer.py`。
- Nr3D 子集抽样、A0–A4 五组模块消融与统一审计见 [消融实验指南](docs/ablation.md)：完整模型、去退火、去 Monitor（含 DAG）、去按需证据获取、Top-5 改 Top-1。当前调度器仅支持 Nr3D。
- 五组采用相同的成功工具预算、修正额度和决策上界；去 Monitor 仍保留共享轮次退火。普通推理入口默认仍使用原 ReAct，可按指南显式开启 Monitor 与退火。旧 E0–E5 不与新配置混用。
- `scanrefer_ready.sh`、`scanrefer_gt_ready.sh`、`run_scanrefer_r18.sh` 和 `run_eviground_local.sh` 保留历史本机路径及冻结源码入口，不能视为新机器上的通用启动命令。新实验应显式传入清单、预测导出、场景和独立输出目录。

`experiments/`、`output/`、`outputs/`、`tmp/`、`local_outputs/` 与 `docs/figures/` 保存本地实验快照、模型输出或论文素材，不纳入当前源码提交；已有文件保留在本机。

## 测试

```bash
PYTHON_DOTENV_DISABLED=1 PYTHONPATH=src python -m unittest discover -s tests/unit
```

清理后的版本需要独立运行以报告新指标；既往实验数字不是本版本的重新评测结果。
