# 直接千问003：单条真实Nr3D验收

2026-11-20（北京时间），隔离分支codex/qwen-direct-fresh-20261120。
用户已明确弃用Laya，改为同一qwen3.8-max直接解释证据、选择单一动作并提出Monitor。
本条从原始输入全新运行；旧Laya/直接001/002代码冻结和结果均保留，未拼接预算。
既有纯千问主线未读取/修改，GPU由用户报告关机，本路线未发服务器/Laya请求。

## 分项结果

| 验收项 | 结果 |
|---|---|
| 合法终局/流程完成 | 是，原严格finish路径 |
| 严格native Monitor | 通过，最终active=[25] |
| 停滞暂定出口 | 未触发；仅本地合同验证，不声称实测成功 |
| 原模型最终ID | 25 |
| 独立定位正确性 | 正确，冻结后读取该条GT=25 |
| 完整证据充分性 | 未完全建立，存在下列明确限制 |
| 正式成绩/论文 | 不计入、不更新 |

原native字段仍为prediction_source=dag_monitor_tentative、decision_status=tentative；
这里的strict_monitor_passed表示native状态/排除/结构门禁通过，非完整语义证据认证。
本轮没有使用新增tentative_stagnation_final旁路。native命名与新旁路模式不混同。

## 实现和验证

同一Qwen请求保留原始文字、实际附图、工具原文和历史，单次返回唯一原生动作。
原Schema/Monitor、ReAct工具、回滚、退火保留；B/R/T和人为32K只记录不硬停止。
用户授权的新出口：候选集合、有效已接受计划和真实证据连续3个决策转换均不变，
或已有精确重复native拒绝时，下一个实际决策由同一个Qwen做一次暂定最终评估。
三是明确工程值，非预算/论文结论。新证据/有效状态进展重置，不能只看候选数。
暂定目标限fixed_allowed∩current_native_active，可null；需原始证据、主要对手与
unresolved。暂定不产生任何Monitor更新；旧基本ID/引用/对象证据结构验证通过后，
accepted=false并保留原计划/未排除候选。只允许一次精确结构修复，非法则失败。
程序不按分数/频次选目标、不自动补排除或节点。普通严格finish保持原Monitor。

简洁candidate_contract将固定合法域、当前未有效排除域、known参照物范围、
已接受计划/修订号、原始证据覆盖、最近拒绝/回滚并列，避免把多active误读为非法ID。
不把分类器Top5当真标签。保留同帧原始行/符号/排序和引用对象覆盖要求。

五项必要本地检查通过，覆盖直接动作、重复拒绝转下一轮暂定、三次无进展计数与
新证据/有效计划重置、合法交集、暂定不改Monitor、至多一次结构修复。
一次现有002静态轨迹核查显示新增真实工具即使候选未减也重置计数。
没有API探针/控制矩阵。候选003冻结SHA：
e2330d325c50ba3181a834afe95f3e76707e08bd1d32fb1b8d337435f2166dbe。

## 原始证据与限制

- 初始原始表：desk12 height1.0453/top_z1.1582，desk13 height0.8164/top_z0.8202；
  实际场景图显示12/13和monitor3/5。支持两张主要桌子间的高度比较，部分结论把top_z
  写成height，术语不严谨；其他低rank桌子假设未逐个明确排尽。
- 场景observation1/6的实际像素支持monitor3与desk12关系；初始底高/顶高接近。
  observation2只测1/12/25，没有3和12的专门pair测量，不能捏造测得overlap。
- observation2模型显式frame=origin[12],facing[3]，同帧right行25=-0.7097、1=+0.1549。
  本轮最终解释没有反转此符号/顺序。该帧是模型假设，原句未明确给观察者，仍需语义论证。
- 最终对0/24的排除依赖“靠近短桌13”的场景判断；同帧数值测量未覆盖0/24。
  初始原始坐标虽全部可见，但模型没有明确给出同帧的完整对手数量比较。
  “在另一桌附近”本身不足以证明不是原句要求的far-left成员。
- observation5的隔离实际像素支持0/24是椅子、2/3/4/5等屏幕及15/17类物体；
  observation7覆盖1/25/0/24/27/40，含27/40场景形状。覆盖字段通过不等于每个
  分类排除都充分清晰；27/40未单独隔离验真。不能把“看过”当作语义自动正确。

所以本条定位正确且strict Monitor通过，但完整语义证据充分性不升级为通过。
不能用GT正确倒推上述排除/帧假设已证明。未用GT反馈规划/调提示/重跑此样本。

## 完整计账

B5/R30/T8（上限全部null）。19个Qwen HTTP200响应：正常解析1、语义咨询18；
10次结构非法提案另记R、2次native反馈另记R，合计18+10+2=30。
五次成功证据工具：场景查看、测量、隔离查看（两个面板）、场景查看、场景查看。
原Monitor曾丢弃不被引用覆盖的排除并实际回滚n0/n1/n11，最终由模型重新提出完成。
无自动完成或替换目标。没有未知usage/传输失败/供应商截断或人为容量拒发。

prompt_tokens合计472104，其中报告cached_tokens219648；completion_tokens12333。
单次实际prompt最高79623，未施加人为32K gate；不据此宣称供应商无限上下文。
没有核实本路线单价，因此不估造人民币费用。附图字节均与每次receipt SHA一致。
原result.parser_calls=9是共享native ledger长度，不是9次解析HTTP；解析实际1次。

## 独立评分和归档

预测冻结时间北京时间14:40:26，评分14:42:32（均在模型运行自然结束后）。
冻结result SHA b351918e02411ffd4a860e2106193d2a0ef04fa28c5c0ea3ec9a3e33dd284b48，
trace SHA de478722737f8e36bf737e0328234ff5941b7065e58b59e0e8f702fc3c8513af。
查询scene0565_00/csv_row12精确匹配一条；之后独立读取本工作区nr3d_val.json，GT25。
评分后无新增模型调用。冻结源/输入副本哈希与原manifest逐项一致。

记录：outputs/qwen-direct/nr3d-003-tentative/：
result.json、trace.json、calls/、frozen-source/、frozen-inputs/、
prediction-freeze-before-evaluation.json、independent-correctness-evaluation.json、
independent-acceptance-audit.json。目录名预先为003-tentative，实际terminal_mode=strict_monitor；
不重命名或事后更改原始结果。此前两个失败独立保存。无全量/ScanRefer/正式分数。
