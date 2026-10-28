# 交接目标：累计约 67% 停止、表格指标、固定模型子集消融

## 给目标模式模型的指令

请完整执行本计划：先接管当前 ScanRefer 重试的只读审计与外部停止控制，使累计 Overall Acc@0.25 首次达到至少 67% 后停止继续派发查询；保留已经发生的所有结果和消耗，生成完整 9508 条口径的累计快照与可填表六项指标。然后在独立目录实现并执行本计划规定的固定模型、小子集配对消融，交付结果与解释。不要为了刚好显示 67.00% 丢弃已经完成的成功样本。不要承诺必须得到有利的消融结果。用户已要求达到目标后停止当前实验、并让另一模型执行后续工作；不必仅重复计划或再次询问常规实现细节。

先读本文件与 README_FREEZE_20261028.md，再核验 freeze_20261028/CODE_MANIFEST.json。当前分支是 20261028-dev。后续控制器和消融实现放在新目录，不修改本分支的冻结快照字节，也不修改活跃运行所依赖的代码、输入、环境文件及历史输出。运行前重新检查实时进度；文中的数字是带时间的观察，不是未来接管时的实时状态。

## 一、路径、基线和不可混淆的口径

- 活跃实验根目录 E：/Volumes/KESU/experiments/scanrefer_r18_gt65。
- 当前输出 R：E/outputs/r18-all-false-gpt61-sol-v1；逐条轨迹在 R/round001/qNNNNN/。
- 冻结基线 B：E/outputs/r18-visual109-v1/results.json。9508 条，correct25=6131，correct50=6114；B 的 SHA-256 为 124993ca593825357842f6f3e15202fc6bf555858d7b4a0fa9ec9b54449294ce。
- 本轮依据 B 的 Acc@0.25 失败标签选中 3377 条，其余 6131 条保持基线。选中的终态结果一律替换，包括错答、弃答、API/运行错误；不按新结果对错择优保留。
- 本轮冻结协议 R/protocol.json；完整基线副本 R/baseline_results.json；本轮清单 R/selection.json；完整评测清单 E/inputs/full_manifest.json。
- 既有 Python：/Users/luocl/Desktop/seeground/reproductions/SeeGround/.venv/bin/python。使用 -B 或 PYTHONDONTWRITEBYTECODE=1。
- 私有凭据只从 $HOME/.config/seeground/scanrefer-gpt61-sol.env 读取；不展示、不复制进 Git、不放入命令行参数。原启动器还依赖既有 source_r18/.env；不要打印其内容。
- 模型请求标识固定 gpt-6.1-sol；API base http://106.14.25.135:3000/v1；参数、schema、thinking、最大生成预算、传输重试设置从当前协议复制。网关身份或模型变更时停止并报告，不能悄悄换模型。
- E/inputs/scenes 与 E/inputs/r8_predictions 只读。对象 proposal 是 GT 几何。target_id、IoU、unique、历史正确性仅供评测和抽样父进程，不能进入推理 worker。
- 不改 /Users/luocl/Desktop/EviGround，不接触同工作区其他 Stage Two/RL 工作，不使用 pkill python，不停止其他任务。

这是依据已知失败标签追加推理、并按评测准确率早停的跨版本累计开发结果。不得称为新模型全量单轮准确率、独立测试集提升或无需 GT proposal 的结果。消融则是另一次固定模型的子集单轮方法比较，不能把两种口径混成同一主表。

## 二、停止目标和起始记录

目标使用 Overall Acc@0.25；Acc@0.5 同时报告，但不是额外停止门槛。

9508 × 0.67 = 6370.36，因此至少需要 6371 条正确，6371/9508 = 67.0067%。从本轮基线需新增 240 条正确。用户截图 6282/9508=66.07%，还差 89 条；本计划编写时于 2026-10-28 15:40:13 UTC（23:40:13 上海）只读扫描已见 301 条 evaluation，其中新增 155 条 @0.25，因此该批快照为 6286/9508=66.11%，距阈值 85 条。文件扫描不是原子快照，接管时必须重读并验证。

默认在 correct25 >= 6371 时触发停止，期望最终约 67.00%–67.20%。允许在途查询造成小幅超出；如接管时已经超出，立即停止继续派发并如实保留当前全部结果，不回滚到挑选出的 67%。不能把达到阈值的“正确条数”误当成要再运行多少查询。运行速度、恢复比例和费用均不得从前缀线性承诺。

## 三、阶段 A：先完成外部监测与安全停止

1. 核验代码及基线哈希、分母、3377 个选择 ID 的唯一性；检查真实进程树、父子关系、命令行和输出目录。只识别 retry_all_false_gpt61.py、其 progress/批量调度子进程和该树下的具体 worker，不读取或输出进程环境中的密钥。记录 PID、PPID、启动时间，防止 PID 复用。
2. 新建独立控制/审计目录，例如 E/operations/target67_20261028/ 与 E/outputs/r18-target67-20261028/。控制器的源码、测试与计划更新可提交 Git；当前运行的冻结源码保持原样。监测默认间隔 10 秒，缓存已核验 evaluation 的哈希，避免反复重读所有大文件。
3. 累计统计从 R/baseline_results.json 开始，用已经完整写入且验证通过的 evaluation 按 csv_row 替换。核对 scene、query、target_id、ann_id、object_table 与清单；reject 重复/未选择 ID、非法 IoU、缺失字段。错误或 predicted_id=null 按失败；普通错答是有效终态。读取中遇到尚未写完的 JSON 应稍后重试，不覆盖源文件。不能只看控制台，也不能假设 round001/results.json 实时包含全部完成查询（该聚合约 30 秒更新，最终还会重写）。
4. 实施前用假的进程/文件做离线验证：阈值未到不发信号、达到阈值仅触发一次、错误/弃答正确计入、重复和半写文件不导致虚增、PID 复用拒绝、没有广泛进程匹配。禁止为了测试向当前真实 worker 发送试探信号。

### 必须解决的现有调度限制

visual109_runtime/run_grounding_v9.py 使用 ThreadPoolExecutor，一次性 submit 所有 pending 查询；每个线程用 subprocess.run 启动 worker。当前运行没有可热加载的 stop flag；把新条件写进磁盘源码不会改变已经运行的进程。直接 Ctrl-C 可能留下没有 prediction/evaluation 的目录，而当前入口会拒绝这种目录的自动续跑。原入口的 merge 要求 3377 条完整替换，因此部分停止后不能调用原入口伪造正常 completion。

可优先采用外部暂停调度、完成在途、终止调度、离线结算的最小方案，具体 PID 必须现场验证：

- 阈值达到后，用 SIGSTOP 暂停“执行线程池的批量调度进程”本身的所有线程，使它不能从已排队的 future 中启动新查询。不要暂停整个实验进程组或正在执行 API 的 worker；暂停外层等待父进程也不能阻止子调度器工作。
- 暂停后重新核对该调度器的现存 worker 子进程/输出目录。它们继续按原预算自然结束；父进程暂停期间不会替这些 worker 写 evaluation，需要事后离线评测。记录触发时已完成 ID、在途 ID、尚未启动 ID 和 API ledger 的状态。
- 根据既有 API 超时/请求上限设定并记录有界等待（默认最多 15 分钟）。每 10 秒检查真实子进程退出与文件状态，不因有新的失败就额外重试。超时后只终止已审计的这几个 worker PID，并记录中断及 usage 未知部分；不能删除轨迹或按零成本处理。
- 在所有在途 worker 结束或被审计终止后，向已暂停的调度器发送 SIGTERM，再发送必要的 SIGCONT 使终止信号生效。须先核对该程序无自定义 SIGTERM 恢复逻辑；否则选择经离线测试的更保守终止方式。只终止这一棵已审计运行树的调度器，确认没有新请求出现。外层入口可能以非零退出结束，这是有记录的用户目标早停，不是 3377 条全跑完。
- 不依靠固定 sleep 推断 worker 已完成，不凭目录存在判断成功，不丢弃同一批在途最终完成的结果。若无法确认准确的调度进程或暂停语义，保留读监测并报告具体阻碍，不能用宽泛 kill 代替。

如果接管时运行已经结束，直接核验全量结果并报告实际准确率，不重跑或删减结果。若进程已经异常退出，先保全已有结果与账本；仅在确需继续达到 6371 时，在全新版本目录实施可逐条受控派发的续跑器，复用所有有效终态，严禁重发整批 3377 条。终态 API 错误也是完成结果，不自动重试；在途不确定请求须先审计，不能静默再发。

### 部分运行的正确结算

停止后把稳定的结果、prediction/run/ledger 状态和源哈希写入独立 snapshot_manifest；使用临时文件加原子 rename 发布汇总，源结果不改。已有 evaluation 原样复用；仅 prediction 已完整终态、缺 evaluation 的查询，使用同一冻结 evaluator 离线补评。若复用 run_case，必须保证 prediction 已验证存在，并禁用/拦截所有新进程推理和网络调用；不能让“补评”触发缺文件重跑。被中断且无有效终态 prediction 的查询记录为 unfinished，保留其基线行，另报中断和已消耗请求，不能冒充新模型错答或完成。

worker 如果在停止信号之前自然以错误退出、没有有效 prediction，且退出码/日志/账本足以确认终态，应按冻结 run_case 的错误构造规则离线记为本轮 error 并替换，不能借“没有 prediction”保留旧答案。只有明确因停止而中断、终态尚不可确定的轨迹保留 baseline 并标记 unfinished；两者必须分开审计。离线错误结算同样禁止启动推理子进程。

最终 cumulative_results.json 必须有恰好 9508 个唯一 ID：所有有效本轮终态都替换（包括错误/弃答），未启动及未完成者保留冻结基线。另存每行来源映射、替换 ID、未完成 ID、结果哈希、停止触发时间、最终结束时间、overshoot、进程操作记录。标注 status=stopped_at_target、evaluation_policy=cumulative_known_failure_retry_early_stop、full_query_coverage=true、selected_retry_complete=false（若确未全跑完）、selection_uses_evaluation=true、stopping_uses_evaluation=true。覆盖 9508 条不等于本轮完成 3377 条。不要给原输出写 completion.json，也不要在报告里保留 helper 自动生成的 complete_single_pass 标签。

验证：按完整清单独立重算 @0.25/@0.5，保留基线的 6131 条成功不变；所有有效终态已纳入，@0.5<=@0.25，Overall 与分组计数相加一致。若最终未达阈值，明确写尚未达标及原因，不能修改阈值伪称完成。

## 四、阶段 B：生成可粘贴飞书的六项指标

复用 scanrefer_metrics.py 的固定分母及官方 unique 标记，不能以本轮完成数或非空答案数为分母。基于阶段 A 的 cumulative_results.json 和 E/inputs/full_manifest.json，离线生成：

- target67_metrics.json：Unique、Multiple、Overall 各自 total、correct25、correct50、acc25、acc50、错误/弃答；另列新模型重试 completed/selected、未启动/未完成数与累计各阶段。
- feishu_row.tsv 与 feishu_row.csv：方法、模型、Unique@0.25、Unique@0.5、Multiple@0.25、Multiple@0.5、Overall@0.25、Overall@0.5、发表/会议。六项百分数保留两位小数，同时在 metrics.json 留整数分子分母和未舍入值。
- FEISHU_NOTES.md：方法建议写“ours（GT proposals，累计定向重试）”，模型写“Qwen 历史阶段 → gpt-6.1-sol”；准确的历史 Qwen 名称应由协议核验，不能仅据界面 qwen3.7-plus 猜测。发表/会议填“未发表”。注明已知失败选择、按评测早停、跨版本复用、GT proposals 和本轮未全量完成。

可使用如下实际已有命令离线产出基础报告，再由新导出器补齐累计标签和来源：

    python -B /Volumes/KESU/experiments/scanrefer_r18_gt65/scanrefer_metrics.py report \
      --manifest /Volumes/KESU/experiments/scanrefer_r18_gt65/inputs/full_manifest.json \
      --results /Volumes/KESU/experiments/scanrefer_r18_gt65/outputs/r18-target67-20261028/cumulative_results.json \
      --proposals gt \
      --output /Volumes/KESU/experiments/scanrefer_r18_gt65/outputs/r18-target67-20261028/target67_metrics.json

此 helper 的默认 complete/evaluation_policy 只反映结果覆盖，需要在最终报告明确改为累计早停口径。helper 从结果目录自动读取的 accounting 也不包含所有历史路径；账本必须单独从真实物理轨迹审计，再按阶段合并，避免重复计算基线累计账本。所有请求尝试含失败请求都计入；无 usage 的列未知数量，价格未知则不报虚构金额。错误与弃答仍留在分母。

只交付填表数据和备注；没有必要操作用户的飞书文档。不要用独立子集消融数据填充这个 9508 条主表。

## 五、阶段 C：预注册固定模型的省 token 消融

目的：检验候选召回、初始实例 atlas、动态证据、显式实体绑定的作用。全程固定 gpt-6.1-sol、网关、temperature、thinking、schema、输出上限、分类器/Top-5 分数、GT 场景、渲染参数、工具预算上限及传输重试策略。只按下面各臂规定改变一个方法组件。不做模型替换、不做 RL、不训练、不自动扫参，不把旧 Qwen 或累计结果当作新模型的控制组。

### 数据与预算（默认方案）

1. 从完整 9508 清单取 8 条 pilot 和与其不相交的 128 条 main；固定 seed=20261028。抽样算法和 ID 列表在任何消融请求前保存并哈希。不得依据旧/新正确性、错误原因或阈值贡献挑样本，也不能使用按 csv_row 排序的运行前缀代替抽样。
2. main 按全量 Unique/Multiple 比例用最大余数法分配 128 个名额；各层按 seed 打乱场景顺序，再打乱场景内查询，轮询场景取样，每层每场景最多 2 条。如无法满足层配额，按预定规则逐轮将上限增加 1，抽样前记录实际上限。pilot 用同规则单独 seed=20261029，先从总体移除 pilot 再抽 main，main 配额仍按原总体组比例确定。
3. 这是场景多样的分层开发子集，不能声称是未见独立测试或全量精度估计。报告实际 Unique/Multiple 数、场景数、每场景样本数、场景实例数分布和原始 manifest 哈希。主要统计为配对差值；Overall 报原始子集值，另可按原总体 Unique/Multiple 比例给加权组值，明确它仍是子集描述，不把加权等同于消除场景抽样偏差。
4. 五臂共享相同 main 128 条，每臂单轮一次，共 640 条 query-arm 轨迹。pilot 五臂各 8 条，共 40；总上限 680 条轨迹。query-arm 数不是 API 请求数；同一轨迹可能多次调用，须用实际 ledger 统计 tokens。不得因结果不显著或不符合预期而扩样。
5. pilot 只检查开关、数据泄漏、schema、账本和错误归因。跑完后可修实现 bug，但不得根据 pilot 准确率选消融或调提示词；修复须新版本哈希，冻结后才运行 main。pilot 不并入主统计。先报告 pilot 实测 tokens 与按同设置估出的剩余消耗区间，不能推算货币费用；不因模型弃答就增加输出预算。

### 五个预注册实验臂

| ID | 实验臂 | 唯一方法变化 | 保持项与实现检查 |
|---|---|---|---|
| A0 | Full | 无，当前 all_instances_atlas_binding_v1 | all-instance 候选、3 overview、完整 atlas、3 动态动作、显式绑定/强竞争者核对 |
| A1 | w/o atlas | 不向模型提供初始实例 atlas 图片 | 保留相同 all-instance 候选、文本/几何信息、3 overview、动态工具；只删 atlas 图片及失效的 atlas 指示，不删候选对象或降低动态图像质量 |
| A2 | w/o dynamic evidence | 禁用 inspect 和 measure，只有固定初始证据 | 保留 3 overview 与全 atlas、候选和绑定；schema 只允许 finish，移除必须调用工具的前置指示，不能暗中把预算留给额外推理/重试 |
| A3 | w/o explicit role binding | 删除新增 target/reference/support/accessory/part-whole 绑定提示及要求显式填写这些绑定解释的句子 | 保留 atlas 全览说明、全候选、3 动态动作、相同 target 与 strongest rival 的动态核对要求及 schema；不回退整个 r18 prompt，也不同时删除竞争者检查 |
| A4 | original category gate | 目标候选恢复为当前 recover_candidates 扩充前的原始 predicted-category 匹配集合 | “Top-5”是每实例的类别假设，不是全场只取 5 个物体；全 atlas/overview 和参考对象可见性保留，finish target 必须在原始集合，空集按既有 abstention 处理。只改变目标候选约束，不按 GT 补候选 |

A2 的动作集合和相应指令必须一起同步，否则会把不可执行提示造成的失败混入结果；这是禁用动态证据机制的必要实现。A3 只检验显式语义绑定提示，不能把它命名为“移除全部 reasoning”。A4 在保持初始图像相同的前提下检验硬类别 gate 的影响，可能节省很少 token，但仍有方法价值。

所有臂均用独立 runtime 副本和新输出目录，例如 E/outputs/ablation-gpt61-20261028/{pilot,main}/{A0,A1,A2,A3,A4}/。每个臂独立保存 effective 配置、prompt/schema 哈希、allowlist、输入哈希、实际请求模型标识。不能通过修改 E/visual109_runtime 或正在运行进程的 env 实现开关。新增运行器必须可在离线 fixture 检查各臂有效载荷差异，并验证 GT 字段未流入 worker。

### 执行顺序与复用规则

- 67% 主结果稳定封存后再开消融，避免争用当前 8 个 API 槽。默认消融全局最多 2 个在途查询、最多 2 个 API 请求，所有臂共享限流，不能每臂各用 2 槽造成隐性 10 并发。
- 用 seed 对 main ID 排序，每个 ID 用平衡轮换顺序运行五臂（第 i 个查询按 i mod 5 轮换臂顺序），以减小网关时间漂移；记录时间和网关返回的模型字段。不能先跑完整 A0 几小时后才跑全部 A4。
- 精确相同 query、臂、输入、模型设置、prompt/schema 和代码哈希的终态可断点复用，成功/失败/弃答同等缓存。不得复用旧模型、不同方法、失败选择前缀或 pilot 的结果作为 main 某个臂。为简单审计，默认 main 五臂全部新跑；不运行第二轮 benchmark。
- 复用只读渲染和静态 atlas 资产以节约本机计算；必须核对点云/相机/实例/渲染参数哈希，不能复用旧模型答案。静态图像复用不等于网关不计 image tokens；如网关报告 cached_tokens 则单独记录。
- 遵守已冻结的传输重试上限，拒绝因预测错误或弃答再试。运行错误和 API 错误均保留，分母固定为该臂 128；另可提供非错误条件下描述，不能替代主指标。持续鉴权失败或网关模型漂移时停止该批并报告，不以无限重试完成数量。

## 六、阶段 D：消融统计与最终交付

每臂报告 Unique/Multiple/Overall 的 @0.25/@0.5 分子、分母与百分比；主要比较 A0-A1、A0-A2、A0-A3、A0-A4 的配对 @0.25 百分点差。列出 Full 对/消融错与 Full 错/消融对的两个 discordant 计数，不只报告净提升。以 scene 为聚类单位对配对查询进行 2000 次 bootstrap，seed=20261030，报 95% 区间；子集小或跨 0 就写证据不足，不包装为显著全量结论。@0.5 和分组为次要描述，不能按最好看的指标挑结论。

同时报告各臂 API attempts、失败请求、弃答、runtime/API/schema 错误、prompt/completion/reasoning/cached/total tokens（仅 provider 有报告时）、未知 usage 尝试数、每 query tokens、动态动作数、atlas 页数、图像数和耗时。比较 token 节省使用配对均值/中位数，未知账本不能当 0。不要重复计入共享基线账本或把本机图片缓存称作 token 免费。

最终文件：

1. 主结果目录的 stop_audit.json、snapshot_manifest.json、cumulative_results.json、row_provenance.json、target67_metrics.json、feishu_row.tsv、feishu_row.csv、FEISHU_NOTES.md。
2. 消融目录的 preregistration.json、pilot_ids.json、main_ids.json、各臂冻结配置/代码哈希、每条结果/ledger、ablation_metrics.json、ablation_table.tsv、ABLATION_REPORT.md。
3. 控制器/离线汇总器/消融开关的必要测试，以及 VERIFICATION.json：模型调用次数、冻结源码未变、源历史结果未变、分母/去重/分组核验、停止进程范围、680 轨迹预算、是否有未完成或未知 usage。
4. 简短中文总结：最终真实 @0.25/@0.5 与新恢复条数、当前重试完成量、可直接粘贴的六项数值、四项消融结果及证据强弱、tokens 和未知部分、所有累计开发/GT/早停口径限制。失败实验也完整报告。

验收标准：当前重试达到目标后已停止继续派发，全部有效在途终态已结算；主结果恰好 9508 行且可从哈希和来源重算；六项表格指标真实；五臂 main 各 128 条且同一子集，固定同模型，结果不择优、不自动扩样；冻结源码/旧结果/凭据均未改写或公开。若运行/网关阻碍使某项无法完成，明确列出已经完成与阻碍，不能伪造数据或静默变更实验设计。
