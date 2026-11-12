# 消融实验指南

## 五组高层模块配置

当前 Nr3D 消融与论文第 4.5 节一致：A0 为完整模型，每次移除一个功能模块，共五组、一张结果表。

| 条件 | 移除模块 | Monitor / DAG | 退火 | inspect | measure | Top-k | 备选类别检查 |
|---|---|---|---|---|---|---|---|
| A0 | 无（完整 EviGround） | 开 | 开 | 开 | 开 | 5 | 开 |
| A1 | 策略退火 | 开 | 关 | 开 | 开 | 5 | 开 |
| A2 | Monitor（含 DAG 组织与修订） | 关 | 开 | 开 | 开 | 5 | 开 |
| A3 | 按需证据获取 | 开 | 开 | 关 | 关 | 5 | 关（无补充观察） |
| A4 | 多类别候选覆盖 | 开 | 开 | 开 | 开 | 1 | 关（无备选类别） |

A2 使用普通 ReAct，不输出 Monitor/DAG，共享调度器仍提供轮次退火。
A3 保留初始场景视图、基础 AABB 几何与 Top-5 候选，动作集合只有 finish；可依据初始证据完成、弃答或修正最终计划，不要求调用已关闭工具。
A4 同时收缩目标及参照物的类别假设为 Top-1，实例框和工具接口不变。
主要比较均为 A0 对 A1–A4；合并移除只能说明模块整体作用，不能分别归因于 DAG、视觉、测量或单条检查，各行差值不可相加。

旧 E0–E5 不是 A0–A4 的别名，不重命名旧结果或复用旧源码快照。
本调度器只接受新编号；归档 E 系列须使用相应旧 Git 版本与冻结协议复现。
汇总器保留读取旧 E 系列协议的能力，历史跨模型比较脚本也独立保留。

## 共享预算与策略退火

五组统一按**成功工具调用**计预算，默认 B=12；无效计划、无效工具请求及被拒绝的提前结束使用修正额度，默认 R=6。
总动作决策上界 T=B+R+1=19；修复也推进轮次。解析和客户端有限重试另行记录。
A2 与 Monitor 配置使用相同口径、修正额度和输出上限。
A3 的工具可用额度为 0，但共享决策上界仍为 T；零工具调用是配置定义，不能单独视为效率提升。
所有组均可提前结束，无需用满预算；证据不足时返回 null，不强制选择目标。

退火按 `lambda_t=t/max(T-1,1)` 改变每轮策略提示，固定阶段为：

| 进度 | 策略优先级 |
|---|---|
| lambda < 1/3 | 比较合理的候选与解释，补充能区分它们的证据。 |
| 1/3 <= lambda < 2/3 | 聚焦合理竞争者及未决条件，减少无关扩展和重复观察。 |
| lambda >= 2/3 | 优先最终核验，仅为可能改变选择的关键缺口继续取证。 |
| 工具用尽或最后一次决策 | 动作 schema 只允许 finish；证据不足则弃答。 |

这是**阶段条件化提示的策略退火**，不改变采样温度，也不自动按轮次裁掉候选。
阶段内的语义取证范围由策略提示约束，资源截止由程序强制执行。
关键反证始终允许改选、重开任务和恢复候选；A1 仅去掉阶段提示，其预算、采样温度和结束检查不变。
阶段边界及指导文字进入源码快照和 `protocol.json`，不能看过确认性结果后修改并续跑同一目录。

直接推理仍默认不开启 Monitor 或退火；调度器按上表显式覆盖各组开关，旧的全局 Monitor 变量不能改写新矩阵。

- `EVIGROUND_DAG_MONITOR_ENABLED=0|1`、`EVIGROUND_POLICY_ANNEALING_ENABLED=0|1`。
- `EVIGROUND_SUCCESSFUL_TOOL_BUDGET=1`：普通 ReAct 也按成功工具计预算；Monitor 始终采用该口径。
- `EVIGROUND_DAG_REPAIR_LIMIT=6`：该口径下的通用修正额度。
- `EVIGROUND_DECISION_MAX_TOKENS=4000`：五组统一的动作请求参数，runner 同时统一设置 thinking floor=8192。
- `EVIGROUND_INSPECT_ENABLED`、`EVIGROUND_MEASURE_ENABLED`、`EVIGROUND_CATEGORY_INSPECTION_REQUIRED`：工具与检查，均为 `0|1`。
- `--category-top-k 1|5`、`--review-limit B`：类别与取证上限。

## Nr3D 运行

按 README 安装依赖，使用**全新实验目录**并设置实际路径和 API 配置。
调度器固定使用覆盖 130 个场景的 1,500 条描述，按场景及 Easy/Hard × 视角依赖分层冻结。

```bash
export EVIGROUND_ABLATION_STUDY="$PWD/experiments/nr3d_module_ablation"
export EVIGROUND_ABLATION_SCENES=/absolute/path/to/nr3d_prepared_scenes
export EVIGROUND_ABLATION_PREDICTIONS=/absolute/path/to/verified_classifier_export
export EVIGROUND_ABLATION_MODEL=your_model_name
export EVIGROUND_ABLATION_BASE_URL=https://your-provider.example/v1
export EVIGROUND_ABLATION_REVIEW_LIMIT=12
export EVIGROUND_ABLATION_DAG_REPAIR_LIMIT=6

python scripts/prepare_ablation_subset.py \
  --manifest data/queries/nr3d_val.json \
  --output "$EVIGROUND_ABLATION_STUDY/nr3d_subset_1500.json" \
  --size 1500 --seed 20261104

# 仅预检及五组固定的子集外技术样本。
PYTHON_DOTENV_DISABLED=1 PYTHONPATH=src \
  python scripts/run_nr3d_ablation.py \
  --start-condition A0 --end-condition A4 --workers-per-key 3 --technical-pilot-only

# 技术样本通过后复用它们，运行五组正式评估。
PYTHON_DOTENV_DISABLED=1 PYTHONPATH=src \
  python scripts/run_nr3d_ablation.py \
  --start-condition A0 --end-condition A4 --workers-per-key 3 --skip-technical-pilot

PYTHON_DOTENV_DISABLED=1 PYTHONPATH=src \
  python scripts/summarize_nr3d_ablation.py --study "$EVIGROUND_ABLATION_STUDY"
```

运行命令会产生实际 API 请求。密钥在等待提示后经标准输入提供一行 JSON 数组：一个共享 key 或两个分片 key，不写入命令参数、源码、协议和结果。
共享 key 的两次预检顺序执行，必须全部通过。API 地址优先取 `EVIGROUND_ABLATION_BASE_URL`，其次 `EVIGROUND_BASE_URL`，最后才是公共默认地址。

直接运行调度器默认完成预检、技术样本及 A0–A4。可用 `--start-condition A2 --end-condition A2` 单独执行一组。
技术样本不按正确率放行；类别候选为空的合法弃答保留，不当作技术故障。已冻结目录只续跑缓存，不挑选正确答案重跑。

## 输出与审计

- `protocol.json`：矩阵、共享预算、输出限制、退火阶段、固定清单哈希与比较对象。
- `source_identity.json` 与 `source/`：推理源码及哈希；更改设置或源码后另开目录。
- `technical_pilot/A*/diagnostics.json`：是否回答、已接受 DAG、预算和校验反馈。
- `runs/A*/key_slot_*/q*/prediction.json`：动作、证据、DAG 修订、退火进度与阶段。
- `react_budget`：五组统一记录工具、修正、决策次数及上限；Monitor 同时保留兼容字段 `dag_monitor_budget`。
- `termination_cause`：正常结束、弃答、预算截止或修正耗尽；技术错误另列。
- `run.json`：请求用量、实际模型及执行轨迹，包括失败前已完成的工具操作，避免自动把技术错误样本开销记零。
- `verified_summary.json`、`ablation_results.csv`、`ablation_results.md`：完整五组审计后生成统一表，包含准确率、配对差值、平均工具/词元和耗尽率。

错误与空预测保留在固定分母。API 失败可能缺少用量，汇总另记缺失用量调用数；未上报 tokens 不代表实际免费。
候选召回另报缺失统计，技术失败不解释为已证实的候选遗漏。不同模型结果不能单独解释为模块作用，跨运行稳定性需重复实验。
程序可校验依赖、引用与对象范围，不能保证视觉和语言判断正确。

## ScanRefer 与既有结果

五组调度、抽样及统一审计目前仅支持 Nr3D。ScanRefer 需冻结相应 Unique/Multiple 子集、预测实例和去重策略，
按目标框 IoU 报告 Acc@0.25 / Acc@0.5，不能复用 Nr3D 实例 ID 正确率。
论文既有主结果与旧 E 系列数值不代表新 A0–A4 的结果。本次代码修改未启动模型评估。
