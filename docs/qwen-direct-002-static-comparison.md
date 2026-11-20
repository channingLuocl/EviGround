# 直接千问002：独立诊断与旧finish静态对照

2026-11-20（北京时间）。独立工作区qwen-direct-fresh-20261120，未接触既有纯千问
主线工作区。旧Laya源码冻结和失败结果原样保留；服务器由用户报告关机，未调用GPU。

当前qwen3.8-max单模型直接行动已真实运行两条。002自然失败，B4/R19/T8，
B/R/T仅记录、未封顶；13个真实Qwen响应（解析1、语义12），无未知usage。
4次成功证据工具：查看、测量、查看、查看。所有实际附图字节SHA核对通过。
没有Laya或第二Qwen复述候选。源码/输入冻结副本哈希逐项验证通过。

流程：未有效完成。证据充分性：未通过。预测正确性：无冻结最终预测，不可评分。
未读GT、不做正式成绩/论文更新/全量。模型中间提出ID1或0不是最终合法预测，
不能抽取其中一个当成成功或事后按GT挑选。001外部无进展停止另档，不拼接预算。

002在t03提交ID1，原Monitor返回：
Finish needs one active target matching predicted_id; active targets: [1,2,3,4,5,15,17,27,40]。
t06/t07提交ID0，仍因剩余活跃候选未排除而拒绝。t08再提出相同可执行动作和Monitor，
在同一已接受状态、真实证据和同一拒绝下触发精确无进展停止，非调用次数上限。
模型修改了叙述不能视为新取证。所有新咨询、结构修复和native拒绝均留账。

此次只做一次静态对照：git007a77b中的原r18_react.py137–267及
v74_grounding.py81–99。不读取当前样本GT。只抽取纯validate_decision函数在
已存002的三次finish上验证，未重新运行旧模型/工具/旧轨迹。
该检查要求合法目标ID、非空reason、原文引用、已知对象ID、非空观察文字，
并要求至少一条证据提到所选目标。旧循环另对rank2–5目标要求先 visibly inspect。
002这三次目标1/0均rank1；基本检查全部通过，旧分支本来可以返回暂定预测。
但旧输出是prediction_source=r18_react_tentative、accepted=False，随后独立评分；
这不是通过当前DagMonitor语义验收。旧版不要求节点完成/回滚或每个活跃候选显式排除。
旧版历史模型为qwen3.7-plus+thinking，不是本次qwen3.8-max；本报告不重算或继承
旧7805条指标，不用历史准确率证明本条/当前合同成功。

对照区分：
1. 三次实际finish拒绝的直接原因属于新增全候选唯一性门禁，不是旧基本ID/引用检查失败。
2. 后续模型把“active候选仍多”误读为“所选ID不合法”，属于Qwen证据/反馈解释错误。
   固定allowed明确包括1和25，前次反馈的active也包括1。t04却说1不在allowed，
   t05再声称合法chairs只有0/24，并按此缩组。排除不能删除物理比较成员。
3. 原始测量frame=null，另有工具提供的conditional observer frames；模型采用某条件帧
   解释方向仍需语义论证，不能凭任意坐标系较小值宣称匹配原句。
4. 旧基本validate只验证字段与引用存在，不验证这些数值/比较组/视觉类别结论正确。
   所以“旧检查通过”不足以证明本条定位正确或证据充分。

当前只保留现有v2执行源码：在002结束后尝试添加的allowed/active解释提示已撤回，
scheduler SHA与002冻结值58bf766d7fd6e8037af5e5d585e5b98993ca462db454904ebf30a91b9a1aae84一致。
没有第三候选冻结/新API调用，没有静默关闭原Monitor，也未自动补排除/换目标。
如以后采用旧式暂定预测协议，必须独立冻结、清晰标记tentative/accepted=False，
预测冻结后独立评分；不能把放宽终止合同叫当前证据验收通过。

机器可继续保持关闭。本报告只依赖本地Git对象和现有002轨迹。
数据：outputs/qwen-direct/nr3d-002/result.json、trace.json、calls/、
legacy-finish-static-comparison.json。该JSON保存旧源码哈希、逐finish对照、
所有实际图片哈希核查、冻结源/输入验证和已知usage，原receipt未覆盖。
