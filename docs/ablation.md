# 消融实验指南

## 当前支持范围

当前代码保留论文 Nr3D 消融所用的 E0–E5 条件。`prepare_ablation_subset.py` 按场景及 Easy/Hard × 视角依赖分层冻结子集；`run_nr3d_ablation.py` 固定要求 1,500 条描述、130 个场景，拆为两个交错分片，并保存源码哈希、协议与逐条输出。它们不能直接接收 ScanRefer 清单。

| 条件 | supplementary inspect | conditional measure | 类别 Top-k | alternative-category check |
| --- | --- | --- | --- | --- |
| E0 | 开 | 开 | 5 | 开 |
| E1 | 关 | 开 | 5 | 关（不适用） |
| E2 | 开 | 关 | 5 | 开 |
| E3 | 开 | 开 | 1 | 关（不适用） |
| E4 | 开 | 开 | 5 | 关 |
| E5 | 关 | 关 | 5 | 关（不适用） |

推理开关为 `EVIGROUND_INSPECT_ENABLED`、`EVIGROUND_MEASURE_ENABLED`、`EVIGROUND_CATEGORY_INSPECTION_REQUIRED`，只接受 `0` 或 `1`；类别由 `--category-top-k` 控制。关闭 inspect 时，类别检查同步关闭；这些设置进入运行身份，不能混用同一输出目录。

## Nr3D 子集复现

先按 README 安装依赖，并使用新的实验目录。下列场景、分类导出和 API 地址必须替换为实际输入；不要覆盖已有实验快照。

```bash
export EVIGROUND_ABLATION_STUDY="$PWD/experiments/nr3d_ablation_new"
export EVIGROUND_ABLATION_SCENES=/absolute/path/to/nr3d_prepared_scenes
export EVIGROUND_ABLATION_PREDICTIONS=/absolute/path/to/verified_classifier_export
export EVIGROUND_ABLATION_MODEL=your_model_name
export EVIGROUND_ABLATION_BASE_URL=https://your-provider.example/v1

python scripts/prepare_ablation_subset.py \
  --manifest data/queries/nr3d_val.json \
  --output "$EVIGROUND_ABLATION_STUDY/nr3d_subset_1500.json" \
  --size 1500 --seed 20261104

PYTHON_DOTENV_DISABLED=1 PYTHONPATH=src \
  python scripts/run_nr3d_ablation.py --workers-per-key 3
```

批处理启动后从标准输入读取一行 JSON：一个共享 API key 或两个分片各自的 API key 组成的数组。密钥不写入命令参数、源码或结果文件。脚本会先进行模型/视觉/结构化输出预检和技术试运行，再开始 E0–E5 正式调用；本命令会产生实际 API 请求。不要在日志或可共享的终端记录中粘贴密钥。

看到等待输入提示后，粘贴 `["你的完整API_KEY"]`（使用英文双引号和方括号），再按回车。
误按回车产生的空行会被跳过并继续等待；格式错误或标准输入关闭会给出不含密钥的明确提示。

输入场景与分类导出路径可通过上述环境变量配置。API 地址优先取 `EVIGROUND_ABLATION_BASE_URL`，其次取 `EVIGROUND_BASE_URL`，均未设置时使用 `https://api.openai.com/v1`；不会自动使用历史私有转发地址。

两个分片共享同一 key 时，模型预检顺序执行，两次均通过才继续；两个不同 key 仍并行预检。
预检失败会在 `preflight_key_*.json` 保存脱敏的 HTTP 状态、服务端错误响应和 request ID，
不会因另一个槽位成功而忽略 401。顺序预检用于排除预检请求相互影响，不能据此认定
供应商有并发限制，也不改变正式推理的并发配置。

完整运行后，用 `python scripts/summarize_nr3d_ablation.py --help` 查看审计汇总参数。`summarize_mixed_nr3d_ablation.py` 仅用于如实记录不同条件使用不同模型的历史比较；跨模型差异不能单独解释为模块贡献。错误、弃答和缺失均不能从固定样本分母中剔除。

## 动态 DAG Monitor 与原 ReAct 对比

原 ReAct 仍为默认。直接调用推理入口时，用 `EVIGROUND_DAG_MONITOR_ENABLED=1`
开启；使用本页消融调度器时，必须设置
`EVIGROUND_ABLATION_DAG_MONITOR_ENABLED=1`，由调度器显式传给冻结源码的 worker。
两个开关只接受 `0`、`1`，设置会进入运行身份及新实验协议。

Monitor 和执行器由同一次多模态模型调用共同完成，没有另训练一个模型，也不额外
增加一轮独立 critic 调用。每轮除原动作外，模型输出完整任务 DAG、当前节点、
要重开的旧节点及修订理由。程序校验无环依赖、语言条件覆盖、已有证据引用、
当前任务是否就绪以及动作是否属于该任务。

`bound_object_ids` 表示当前绑定结果，工具的 `focus_ids` 可包含尚未绑定的候选、
参照物和比较成员。排除记录按该节点引用的实际观察范围检查，不要求被排除的竞争者
仍在绑定结果里。`initial` 覆盖已提供的初始视图与全场景基础几何；工具观察只覆盖
实际查看或测量的对象。证据范围检查不等于证明视觉结论正确。

- 节点可表达参照物绑定、属性、空间关系、候选组比较和最终核验，支持并行分支。
- 回退或修改已完成节点的语义/绑定时，该节点及受影响后继自动失效；旧结论、
  证据关联和排除记录清空，原始观察继续保留，相关目标自动恢复比较资格。
  结构合法的回退即使附带了尚不能执行的动作，也会保留失效后的图，并通过
  `ready_node_ids` 告知下一轮可重核的节点；无环、条件覆盖、证据引用等结构检查
  未通过的提案仍整体拒绝。历史中的 `action_error` 表示回退已接受、动作被拒绝。
- 排除只改变目标资格，不删除用于判断“中间”等关系的物理比较成员。
- 选择目标须位于最终核验节点、全部任务完成、条件覆盖且活动目标唯一，并通过原有
  类别观察检查；有未决项则不能作为成功完成。无效 DAG 会返回具体缺失条件或证据 ID，
  修正额度耗尽后仍无法通过则输出 null，不强行制造唯一候选。
  仅当上述完成、唯一性、无未决项检查均通过，且最终节点绑定列表为空时，使用
  显式 `predicted_id` 补齐该冗余绑定，并记录 `normalized_final_binding`；非空冲突
  绑定不会被覆盖，已有类别观察检查也不会跳过。回退导致任务失效后仍不能直接结束。
- DAG 模式按成功执行的工具调用计算取证预算；计划校验失败、无效工具请求和被驳回的
  提前结束使用独立的有限修正额度，不占用取证次数。默认取证上限仍为 5，额外修正额度
  默认为 6，总动作决策调用上限为 `取证上限 + 修正额度 + 1`。工具用完后仍可修正最终
  核验，但不能继续取证。关闭 DAG 的原 ReAct 仍按原有非终止决策轮数计费。
  DAG 模式单次请求输出上限从 1,600 调至 4,000 tokens；现有 runner 统一使用的
  thinking floor 仍是 8,192。
  图和历史会增加上下文成本，对比时同时报告实际 tokens、时延、弃答和准确率。

直接推理用 `--review-limit` 设置取证上限，`EVIGROUND_DAG_REPAIR_LIMIT` 设置额外
修正额度；消融调度器对应使用 `EVIGROUND_ABLATION_REVIEW_LIMIT` 和
`EVIGROUND_ABLATION_DAG_REPAIR_LIMIT`，显式传给 worker 并写入实验协议及运行身份。
以下新实验使用 12 次取证和 6 次修正，至多 19 次动作决策；解析及客户端重试另计。
与历史 5 轮结果的差异同时包含预算变化，不能单独解释为 DAG 的贡献；公平对比须
另设相同预算的 baseline，或同时报告成本。

程序只能检查结构和证据关联，不能保证视觉结论或语义解释正确。本实现没有新增
“between/on/middle”确定性几何判定器，也没有把分类分数当概率设置动态阈值。
本轮要验证的是整体计划、阶段状态及依赖回退机制。

先只跑同一子集的 E0，无需重跑 E1–E5。沿用原消融的场景、预测分类、模型与 API
设置，但使用**全新的实验目录**；不要复用或改写旧 source 快照。已有冻结子集时
原样复制；如果只拿到了源码，可用前面的抽样命令和相同 seed 重建，并核对选中行。

```bash
# 在仓库根目录、已安装依赖的 Python 环境中执行。
export EVIGROUND_ABLATION_STUDY="/Volumes/KESU/EviGround/experiments/nr3d_dag_monitor_v3"
export EVIGROUND_ABLATION_SCENES="/Volumes/KESU/论文/paper/reproductions/SeeGround/local_outputs"
export EVIGROUND_ABLATION_DAG_MONITOR_ENABLED=1
export EVIGROUND_ABLATION_REVIEW_LIMIT=12
export EVIGROUND_ABLATION_DAG_REPAIR_LIMIT=6
# 沿用原消融的分类预测导出、模型和 API 配置；勿指向按 query 保存的推理结果。
: "${EVIGROUND_ABLATION_PREDICTIONS:?请设置为原消融使用的已验证分类导出目录或 ZIP}"
mkdir -p "$EVIGROUND_ABLATION_STUDY"
cp /Volumes/KESU/EviGround/experiments/ablation_20261104/nr3d_subset_1500.json \
   "$EVIGROUND_ABLATION_STUDY/nr3d_subset_1500.json"

PYTHON_DOTENV_DISABLED=1 PYTHONPATH=src \
  python scripts/run_nr3d_ablation.py \
  --start-condition E0 --end-condition E0 --workers-per-key 3 --technical-pilot-only
```

这里的场景、子集和新输出路径是本机已挂载的 KESU 路径，未搬移或覆盖旧数据。
换机器运行时修改相应路径。外盘可用空间需覆盖新增渲染图与日志；不要将数据提交 Git。

按原方式从标准输入提供 API key 数组；上述命令只预检并运行两条固定的子集外技术样本，
不会启动 1,500 条正式评估。DAG 技术试运行还要求每条至少接受过一版任务图，且没有
耗尽修正额度；不设置正确率门槛，合法弃答仍如实记录。检查
`technical_pilot/E0/diagnostics.json` 后，移除 `--technical-pilot-only` 可启动正式评估。
若出现技术错误先查日志，不按准确率修改样本或重跑。E0 的汇总保存在
`$EVIGROUND_ABLATION_STUDY/runs/E0/results.json`，每条
`runs/E0/key_slot_*/q*/prediction.json` 的 `active_trace` 内有
`dag_monitor_history`、`dag_monitor_current_node_id`、`dag_monitor_final_plan` 和
`dag_monitor_active_target_ids`，可检查回退、失效节点及恢复的目标；`dag_monitor_budget`
另记实际工具次数、失败检查数与决策调用数。
原始请求、响应和 token 用量仍在同目录 `run.json`。

若要在本次源码下重新跑原 ReAct 对照，改成另一个新目录
`nr3d_react_baseline_v1`，设置 `EVIGROUND_ABLATION_DAG_MONITOR_ENABLED=0`，
复制**同一份**冻结子集，再执行相同 E0 命令。保持模型、输入、工具开关和预算一致。
本节只运行单条件 E0，不要对该目录使用要求 E0–E5 全部完成的六条件汇总器。

## 后续 ScanRefer 子集消融

本轮整理没有启动新的 ScanRefer 模型调用。现有 `scripts/scanrefer.py` 可生成官方格式的验证清单并按 Unique / Multiple / Overall 汇总 Acc@0.25 与 Acc@0.5；主批处理 `eviground --manifest ...` 已支持 ScanRefer 的 IoU 评估。

迁移 E0–E5 时，还需要实现 ScanRefer 专用或通用的子集抽样、六条件调度与汇总：

1. 首次模型调用前固定样本数量、种子和场景/Unique–Multiple 分层，保留来源哈希；不得按模型结果挑选样本。原 `prepare_scanrefer_splits.py` 是历史诊断划分，不能冒充新冻结的消融子集。
2. 所有条件使用同一份子集、模型、预测实例及分类导出、去重策略、初始视图和工具预算，分别保存源码及配置身份。
3. 明确采用 Mask3D 预测实例或 GT 实例。预测实例 ID 与 GT 目标 ID 不可直接比较；GT-proposal 实验须单独标注，不能混入 Mask3D 协议。
4. 用预测框与目标 GT 框的 IoU 汇总两个阈值及三个分组，保留错误/弃答在分母中，并给出相对 E0 的配对差异。旧 Nr3D 汇总器的 `correct` 统计不能替代这些指标。

外部数据、权重、完整实验输出与论文文件不随当前源码仓库分发，论文中的既有数字也不代表本次代码整理后的重新评测。
