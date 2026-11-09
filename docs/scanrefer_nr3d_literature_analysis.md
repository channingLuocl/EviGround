# Nr3D 与 ScanRefer 差距：文献核验与本地诊断

核查日期：2026-10-26。研究问题：保持 qwen3.7-plus、预测类别 Top-5 和自主 inspect / measure / finish，为什么现有方法在 Nr3D 较好，在 ScanRefer 较低；下一步应该验证什么？

最新R125真实试验没有提分：12条开发描述×两组×两次，原版合计15/24、新版14/24（@.25与@.5均相同），机制门槛亦未通过，关闭且不进入未用验证。q3067的新版独立距离计算正确、模型也确实引用了观察，但以“更近桌子”替代原文完整关系而退步。此例直接区分了证据计算/使用与语义有效性；不能由一例定量解释全部分差，也不能把字段增加当有效改进。见[R125全部结果与成本](/Volumes/KESU/datasets/ScanRefer/runs/r125_prospective_evidence_pilot12_v1/REVIEW.md)。

最新执行证据：R123有界扩查发现3条明确朝向算术矛盾，仅1条来自ScanRefer原文；R124在两个早期错误前缀各4次真实反馈续跑中，0/8自主取得新观察，全部直接finish。4次数学条件自洽不等于正确视角或目标。冻结输入、16条自身流与实际原响应核验通过后关闭该方案，不采用、不追加变体，不升级准确率试验；已有分差归因和正式成绩保持。[完整负结果](/Volumes/KESU/datasets/ScanRefer/runs/r124_declared_axis_mechanism_v1/REVIEW.md)。

R117目录计算复核未建立新的复发执行错误：199次数值文本已读，限定32事件的64项数值/排序独立重算均一致；其余167事件不能据此认证无错。完整筛查分母和199次实际原响应核验通过，0新模型请求/准确率。不新增计算拒绝或强制measure，也不把数值一致推成参照/比较组正确。[范围与限制](/Volumes/KESU/datasets/ScanRefer/diagnostics/r117_catalog_calculation_scope_v1/REVIEW.md)。

便于阅读的原因排序、文献机制和实验优先级见[简版分析](/Users/luocl/Desktop/EviGround/docs/scanrefer_diagnosis_brief.md)。R115补完原成功scene inspect的71图/286标号核查，没有标号被后续标注完全抹掉，1个部分遮挡；独立覆盖计算、历史PNG和实际交付核验通过。不支持重做布局或新增回执机制，不证明标号可读性/理解，没有新模型调用或准确率。[复核](/Volumes/KESU/datasets/ScanRefer/diagnostics/r115_annotation_receipt_audit_v1/REVIEW.md)。

补充原文核查见[SPAZER/IVSGround与证据选择](/Users/luocl/Desktop/EviGround/docs/scanrefer_evidence_selection_supplement.md)：分别标明250条子集、视图选择器训练和Nr3D消融的边界。本次只补充文献与开发集内算术，不改变正式推理或既有准确率。

58.33% 来自反复用于开发的 96 条 ScanRefer 样本，不是全量或未用验证成绩。它与历史全量 Nr3D 69.66% 的差值只是描述性比较；现有证据尚不能量化各因素对这 11.33 点差距的因果贡献。

供下一步决策的要点：

- **机制研究的当前边界。** R105原13次measure没有字面命名ID距离遗漏，R99/R102补充16次仅1例；R108扩查源码一致的历史对照94次测量，在更宽的对象组意图定义下发现3事件、2描述/2场景。R109已对其中两个有工具机会的实际前缀做冻结的真实模型对照：每正例各4次/arm，增加事实范围反馈与原观察均为0/4成功跨组补测；合法frame-only反例各2次均未补测。20次续跑、24次新HTTP、全部自身流复放通过，但未过机制门槛，关闭、不进入准确率实验。这个低频错误不能解释主要分差；单纯提醒范围缺口尚未引出所需纠错动作。R106原4次容量错误后全部合法恢复，也不支持扩大容量/统一重试。关系绑定与证据选择仍是研究重点，尚无可采用的执行修复。[R108](/Volumes/KESU/datasets/ScanRefer/diagnostics/r108_control_measure_inventory_v1/REVIEW.md)、[R109完整负结果](/Volumes/KESU/datasets/ScanRefer/runs/r109_measure_scope_mechanism_v1/REVIEW.md)。

- **@0.25 优先处理同类实例的关系理解与证据选择。** 原开发集 Unique 19/21、Multiple 37/75；按原记录类别池核对，有覆盖仍失败的30条中25条选框最佳匹配为同类另一实例。旧后置重算的31条分母另见下文，不混用。R94同目标/同前端换原始描述时，候选覆盖相同而Nr3D原文15/23、ScanRefer原文11/23；仅为小样本开发信号(p=.125)。具体缺口包括从句目标归属、先选哪个参照、在哪个比较组内排序、方向依赖什么观察坐标。
- **@0.5 另查前端框质量。** R93固定同32条Nr3D语言换实例前端，@0.25为23对22，@0.5为23对18。前端几何前提确有差异，但不能把“有差异”直接推成“解释了约11点”。不能将两个诊断差值相加。
- **改进验收应检查实际观察和纠错。** 以“显示器—纸堆—两把椅子—左边一把”为例，需保留原文条件与暂定ID对应，区分已观察、未观察和冲突；允许自主回到上游参照。需要补的是能区分假设的观测及纠错动作。单加解释字段已试过；新增工具也曾完全未被调用，不能因代码具备能力就认为推理使用了它。该方向尚无可采用版本。
- **局部上下文也未证明稳定提分。** R102同16条开发描述同期正式9/16@0.25、上下文11/16，三改善一退步，p=.625；14条实际收到新图。未达到预定升级条件，未采用。不能把其中68.75%当成ScanRefer已达约70%。后续变体须先解释它能补哪项现有证据缺口，避免继续重复加图、加提示和相似小探针。

若96条的Unique保持19条正确，接近70%需要总正确约67条，即Multiple从37增到48、约64%；这是净改善11条的任务。应先证明这类选择错误稳定减少，再做新端到端开发集、真实Nr3D回归及未用ScanRefer验证。数据状态补正：用户未另存原始RGB-D，但此前已下载12场景.sens及RGB/位姿缓存，位于KESU的original_sensor_v1、scanrefer_shared_rgb_color12_scenes_v1和observed_rgb_frame_pool_v1。这些扫描相机位姿不是标注者说话视角；部分场景缓存也不等于复现了文献完整2D输入协议。

当前结论：本地两条流水线的实例几何前提不同，但下述 R93 实测不支持优先把 @0.25 的主要差距归因于检测器。更直接的优化对象是**有合格候选时的同类实例选择，以及参照物—比较组—目标的对应关系**。Nr3D 使用 GT 实例几何与预测类别，ScanRefer 使用 Mask3D 预测实例与预测类别；后者仍有漏检、残缺及几何误差，在 @0.5 下尤其值得单独监测。现有数据尚未量化语言、场景构成和模型波动各自的贡献，也不能保证保留全部条件后达到 70%。[S1–S8；L1–L6]

**R110方向错误的进一步限定。** 扩查原R22及48个源码一致历史对照的596条可恢复轨迹、1,487次动作，237次同时提方向和坐标；12次带符号轴朝向筛查中9次数学一致、2次只是位置、1次明确换算相反，涉及1描述/1场景。这个单例不足以开发新的方向反馈或拒绝机制。模型未声明或未验证说话者视角，仍是关系理解问题；不能将它们统称为工具算错或左右公式反了。原R22与R23有retrieval/recovery两文件差异，相关ReAct/测量源码相同，历史对照则完整41文件一致；重复记录和词面筛查均不代表独立总体错误率。[R110范围、排除及验证](/Volumes/KESU/datasets/ScanRefer/diagnostics/r110_declared_direction_contract_v1/REVIEW.md)。

**后续实测更新：R93 同语言前端对照已完成。** 下文文献建议之后，实际完成同一32条Nr3D原描述、共享R67语言解析、同qwen3.7-plus/Top-5/自主ReAct的前端对照。统一后置框IoU结果：GT实例23/32（71.875%）@0.25、23/32@0.5；Mask3D实例22/32（68.75%）@0.25、18/32（56.25%）@0.5。两组均0错误、0弃答。@0.25为4改善、5退步，仅净少1条；@0.5净少5条。开发小样本、固定旧解析和一次模型运行不足以证明效应大小，但**不支持把原约11点跨数据集差距主要归因于实例前端**。前端差异与几何风险仍成立，应降低“先修检测器即可补齐差距”的优先级。[L6]

这次Mask3D有27/32条包含合格类别候选，实际选对22条；10条失败中3条原proposal缺失、2条类别检索后缺失、5条有合格候选但选错。GT实例31/32有合格类别候选、实际选对23条。不能因Mask3D条件选择率22/27较高，就断言预测实例更利于推理：覆盖成员、实例集合和分类同时变化，样本很小。两组64条实际HTTP/轨迹/图片/最终决定均零网络复现，标量/向量IoU一致。GT组首轮32条完整请求与R67逐字相同，本次23/32与先前25/32也提示模型输出波动。[L6]

**后续实测更新：R94 同目标原始描述对照也已完成。** 固定相同 Mask3D 场景、类别表和物体，在全部 23 个有配对的开发场景中各按身份哈希选一对原文，两边实时解析并自主推理。Nr3D 原文为 **15/23（65.22%）@0.25、14/23（60.87%）@0.5**；ScanRefer 原文为 **11/23（47.83%）@0.25、11/23（47.83%）@0.5**。均无终止错误或弃答。两组 @0.25 的合格实际候选覆盖均为 21/23，覆盖失败为同一原 proposal 缺失目标和同一类别池缺失目标；有覆盖却选错为 6 对 10。@0.25 为 Nr3D 原文 4 改善、0 退步，精确配对双侧 p=0.125；未证明总体显著差异，但满足预记的深入语言绑定的开发决策条件。[L7]

四条差异的实际轨迹分别暴露：q07466 把最小世界 X 当成 far left；q03057 直接以较大 X 当成 right 并首步结束；q06045 在 hanger 类别匹配为空时自行用条状物作参照及观察坐标；q00804 将从句中的 toilet 当成最终目标。前三条两组目标池完全相同，第四条因 Target 混入参照类别而变宽。这里可以验证模型用了什么理由，但没有 GT 参照身份或观察者视角，不能把每个未验证假设都断言为唯一错误原因。尤其 q06045 的 Nr3D 原文反而更长，说明不能简单归因于长度或通过删短 ScanRefer 原文修复。[L7]

这 23 条含 Unique 4 / Multiple 19；两组 Unique 为 4/4 对 2/4、Multiple 为 11/19 对 9/19。它是既有开发场景、两数据共有目标的选择性小样本，一次模型运行，描述改变也使解析、检索、参照与动作共同改变，不能将 17.39 点差值视为全量语言效应，更不能把 65.22% 填成新的 ScanRefer 或 Nr3D benchmark 成绩。46 条实际请求、原响应、图片及轨迹已离线精确复现，独立标量/向量 IoU 一致；正式版本仍未提分。[L7]

**1. 先区分可以比较的条件。**

| 条件 | 本地 Nr3D | 本地 ScanRefer |
| --- | --- | --- |
| 实例几何 | ScanNet GT 实例分割及其框 | Mask3D 预测 mask 及其框 |
| 供推理使用的语义类别 | 预测 Top-5 | 预测 Top-5 |
| 最终任务 | 从给定实例中选 ID | 从预测实例中选框并满足 IoU |
| 正确性口径 | ID 准确率 | Acc@0.25、Acc@0.5 |
| 当前已完成参考 | 历史全量 69.66%；最近小回归 25/32 | 开发集 56/96、50/96 |

本地 `prepare_scene.py` 明确由 ScanNet aggregation 分配实例 ID，并验证 OLT 框与这些实例的几何一致；最近 Nr3D 回归的 30 个场景 provenance 全部明确记录 GT instances。当前类别通过另一个预测 Top-5 表进入推理；**预测类别不意味着实例几何也是预测的**。ScanRefer 准备脚本则从 Mask3D masks 构建实例，GT 框另存供评测。[L1，L2]

原始 ReferIt3D 论文明确假设已给定 segmented object instances，关注同一细粒度类别的实例辨别；ScanRefer 原论文要求由场景和语言定位框。两者都来自 ScanNet，但任务前提并不相同。Nr3D 本身也专门包含同类干扰物，因此不能笼统说“Nr3D 没有同类区分问题”或“ScanRefer 必然更难”。本地 Nr3D 没有把 GT 目标类别或 stimulus candidate IDs 当作当前 ReAct 的候选列表。[S1，S2；L1，L3]

**描述的采集方式也不同。** ReferIt3D 让说话者面对突出显示的目标及同一细粒度类别的干扰物，给听者提供能区分它们的描述；两人的初始观察相机随机且不同。ScanRefer 则要求至少两个完整句子，同时描述外观与相对位置，再由核验者筛掉错配或多义描述。这为本地“Nr3D 描述较短、ScanRefer 出现多句指代及多级参照”提供了采集机制上的解释，但不证明 ScanRefer 文本质量更差，也不证明多句本身降低准确率。ScanRefer 原论文附录 Table 11 的完整描述优于仅首句的 @0.5 结果，反而提醒不能用自动截断或只保留主名词来解决本地问题。应保留原文，检查条件分别绑定到谁。[S1 §4.1、附录 Table 11；S2 §3.3]

69.66% 是历史全量 Nr3D，58.33% 是 96 条 ScanRefer 开发样本，二者约相差 11.33 个百分点；不是同一批场景、同一任务或同一次实验的因果对照。25/32 的 78.125% 只支持小规模回归，不证明跨数据集或全量达到这个水平。[L3，L4]

**2. 已核对的 ScanRefer 文献结果。**

数值为百分数；每格依次为 Acc@0.25 / Acc@0.5。

| 方法 / 模型 | Unique | Multiple | Overall | 核验范围 |
| --- | ---: | ---: | ---: | --- |
| ZSVG3D，原论文 | 63.8 / 58.4 | 27.7 / 24.6 | 36.4 / 32.7 | 原文 Table 2 |
| ViewInfer3D，GPT-4-0613 + CLIP | 未报告 | 56.25 / 42.83 | 60.31 / 47.20 | 本机原文 Table I，期刊页 7473 |
| SeeGround，Qwen2-VL-72B | 75.7 / 68.9 | 34.0 / 30.0 | 44.1 / 39.4 | 原文 Table 1 |
| VoG，Qwen2-VL-2B | 69.1 / 63.1 | 25.3 / 21.9 | 35.9 / 31.9 | 原文 Table 1 |
| VoG，Qwen2-VL-72B | 78.6 / 71.5 | 34.0 / 30.4 | 44.8 / 40.3 | 原文 Table 1 |
| AgentGrounder，Qwen3-VL-32B | 80.3 / 73.7 | 36.6 / 31.7 | 47.2 / 41.9 | **仅 1,000 条分层抽样** |
| Active perception，GLM-4.5V 等 | 未报告 | 59.84 / 49.46 | 63.52 / 54.88 | 原文 Table 1；非同一组件配置 |
| 本地已完成开发集，qwen3.7-plus | 90.48 / 85.71 | 49.33 / 42.67 | 58.33 / 52.08 | **96 条开发集，不是全量** |

这些值来自不同模型、组件、样本和类别映射，不能作为严格同条件排名。ZSVG3D 原文的不同实验使用了不同 GPT 版本，不能仅凭后续论文的统一表头，把原始 36.4 一律标为已独立核验的 GPT-4 Turbo 实验。ViewInfer3D 的在线原文请求失败后，在本机已有文献目录找到原始 PDF，已逐页阅读并渲染 Table I 核对上述数字；其前端为 GroupFree + PointNet++，属性使用对应 2D 图像的 CLIP 特征，与本地 Mask3D 条件不同。[S3–S7，S9]

文献中的关键对照：

- SeeGround 同一论文报告 Nr3D 46.1、ScanRefer 44.1，跨任务仅相差 2.0 点。Active perception 的 Nr3D 为 64.9，ScanRefer 为 63.52 / 54.88；**64.9 不是 ScanRefer 成绩**。这些反例说明，本地的约 11 点差距不能全部解释为数据集名称不同。[S4，S7]
- SeeGround 附录 Table 10：GroundTruth proposals 59.5，Mask3D 44.1，OVIR-3D 30.7，显示前端条件对最终表现有很大影响。该表标题只写 Acc.，44.1 与主表 @0.25 对应；GT 变体是否也换了语义标签没有明确交代，因此不能把 15.4 点全归因于框质量，也不能称 59.5 为理论召回上限。[S4]
- AgentGrounder 当前官方代码也存在类似前提差异：Nr3D 从 ViL3DRef 提供的 instance labels 构造实例，再预测类别；ScanRefer 使用 Mask3D。论文笼统写“Mask3D preprocessing”，不足以代替代码层面的条件核查。当前 master 不保证是论文实验的精确提交。[S6]

**3. 本地证据：差距集中在哪里。**

本次对完成记录重新统计实际 ReAct 输入，未发送模型请求或读取 GT / 评分文件。ScanRefer 用原开发集 96 条，其中 92 条有可恢复 ReAct 输入；Nr3D 用最近固定 32 条。下表为中位数。[L3]

| 实际输入负担 | ScanRefer：92 条 | Nr3D：32 条 |
| --- | ---: | ---: |
| 对象目录成员数 | 85 | 40.5 |
| 目标候选数 | 11 | 5 |
| 参照候选独特 ID 数 | 9.5 | 3.5 |
| 首轮 prompt tokens | 31,197.5 | 14,871.5 |
| 原描述词数（ScanRefer 全 96 条） | 17.5 | 8.5 |

这说明当前 ScanRefer 配置要求模型同时处理更多候选、参照和文本。它是不同场景、不同样本的描述性比较，尚未分离场景密度、标注方式和检测前端各自贡献；不能声称检测器独自造成对象数翻倍。也没有证据证明 3 万 tokens 超过模型能力或直接导致这些错误。此前无损压缩已降低输入量，但完整开发集没有提分。[L3，L5]

后置失败审计给出更直接的定位：

- 96 条原结果为 56 正确；26 条选择框的最佳评估匹配属于同一规范类别的另一实例，4 条匹配另一类别，6 条与任何评估框的 IoU 都不足 0.25，另有 2 条终止错误、2 条未答。框的最佳几何匹配并不保证真实语义身份。[L4]
- 使用旧解析和当前正式检索器重算，87/96 条存在 @0.25 合格类别候选，另 2 条因旧解析缺失保持未知。这是条件于旧语言的后置覆盖，不是新准确率，也不是原 R22 每条实际候选池的逐字复现。[L4]
- 上述后置重算覆盖下的31条失败中，25条选成同类的另一实例，27/31的目标规范类别已字面出现在原解析目标中。这些是历史后置分母；随后改按原记录类别池核对，覆盖86条（56命中、30失败）、无合格候选8条、记录不可用2条。q507是两种池统计的差异；原池30条失败中同类另一实例仍为25条。主要损失不是简单“没识别出chair类别”，更不是主要把目标选成参照类别；最佳框匹配仍不保证物理身份。[L4；原池计数](/Volumes/KESU/datasets/ScanRefer/literature_review_20261026/DECISION_COUNTS.json)
- Unique 为 19/21，Multiple 为 37/75。若 Unique 保持 19 个正确，要让总数达到 67/96≈69.79%，Multiple 需从 37 提到 48，即净增加 11 条，约从 49.33% 提到 64.00%。这个算术目标用于确定优化重点，不保证全量分布或泛化。[L4]

解释这些现象的工作假设是：GT 实例往往提供一个更接近物理物体的单位；Mask3D proposal 可以残缺、合并或重叠。同一个“椅子”可能对应多个 proposal，桌子可能只保留局部，参照窗/桌/柜也可能绑定错。这样，即使距离和左右排序数值完全算对，仍可能是在错误物体、错误比较组或错误坐标假设上作出判断。该解释得到前端条件、本地同类实例错误和文献对照的共同支持，但影响大小仍需受控实验。[S3–S7；L1–L5]

例如 q01131 包含“显示器→前方纸堆→纸堆前两把椅子→其中左边一把”的关系链。旧轨迹准确算出了其自选显示器与椅子对的左右坐标；这不证明选中的显示器、纸堆和椅子对就是描述所指。R87 的 13 条实际 measure 轨迹中，限定的局部数值核查没有发现算术错误，也没有验证完整语义链。[L5]

**4. 文献中真正值得借鉴的机制及反证。**

SeeGround 使用查询条件的取景、3D ID 到图像的可见性对齐。VoG 把对象、视图、可见性关系组织成可检索图，维护累积对象池。Active perception 对候选与已解析参照进行局部标注和裁剪，按候选积累证据；出现冲突时可回退到上游参照，而不只验证最后的目标。[S4，S5，S7]

ViewInfer3D 则把观察位置/朝向 `(origin, anchor)` 与目标/参照 `(target, landmark)` 分开推断。描述中的参照物不一定是观察者正朝向的物体；它将多个可能观察坐标与逻辑条件加入图，再执行关系检查。可借鉴的是显式区分这两种角色，不能直接照搬其确定性图筛选替代本地自主 ReAct。其 200 条 ScanRefer 错误分析同样将检测定位列为主要来源，但不是本地的定量归因。[S9]

对本地最有价值的是**让物理实例、可见图像、proposal ID 和原文关系保持可追踪对应**，以及**允许重新绑定错误参照**。这是工具能力与证据组织问题；照搬它们的模型、固定加权决策或候选扩张流程，会改变用户的实验条件，不能直接采用。[S4–S8；本次方法建议]

反面证据同样需要保留：VoG 72B 相对 SeeGround 72B 仅提高 0.7 点 @0.25，Multiple@0.25 仍为 34.0。其视图和轮数收益会饱和；Active perception 也在约 3–4 轮后趋于饱和。现有 R28 重叠信息、R31 NMS、R43–R57 照片/投影、R61 显式绑定、R74 相机参数、R91 对手比较，都没有满足本地采用条件。因此“加图、加轮数、加一句比较提示”不是这次文献分析后可以重新包装的已验证方案。[S5，S7；L5]

Active perception 的受控路由消融中，单次 ScanRefer@0.5 为 47.8，状态驱动版本为 54.9；这支持检查观察路由和冲突处理，但所用 Mask3D、CLIP、GroundedSAM、GLM-4.5V 及可恢复候选流程与本地不同。其累积消融不能把每个模块的收益独立相加，更不能据此承诺本地 70%。[S7]

另检索到 TDVR 报告 ScanRefer 70.85 / 64.06，但其 §3.3 把目标外观、目标到参照方向及目标在同类中的位置加入原查询，并定义 `V_gt = L - P` 的观察方向。文中未清楚说明这些目标/相机信息在标准推理时如何获得；且 Table 5 最佳模型为 deepseekv3.2-reason，并非 qwen3.7-plus。去掉该查询增强后 @0.5 从 64.06 降到 40.59。故该 70.85 具有额外输入条件疑问，应单列，不能作为本地同条件可达的证据，也不能把 GT 目标信息生成的新描述送给模型。此处是根据公开方法提出的可比性限制，不是对作者实验行为的确定指控。[S8]

**5. 实验优先级与当前执行状态。**

已完成的第一步：R93 对同一批 Nr3D 描述、同一场景比较原给定实例与 Mask3D 预测实例。两边使用既有同 r8 分类器产出的各自 Top-5、固定同一次旧语言解析、相同 qwen3.7-plus ReAct 和工具预算，统一后置框 IoU 评分。它不是官方 Nr3D ID 回归或新 ScanRefer 成绩，也没有把预测 proposal ID 直接和 GT object ID 比较。[L6]

该诊断回答：**固定语言及 ReAct，替换整个实例前端后表现如何变化？** 实例集合、框几何、预测类别、NMS 和检测分数同时变化，不能单独归因于框几何。R93 @0.25 仅净少 1/32，未达到预记优先前端的决策条件，因此下一步转向同目标原始描述对照；@0.5 净少 5/32 则保留了框精度问题。两项均是小样本一次开发诊断，不能推断全量效应。[L6]

已完成的第二步：R94 固定同一 Mask3D 场景与物体，比较 23 对原文，两边实时解析与自主推理。@0.25 净差 4/23 且覆盖相同，支持优先检查原文目标/从句参照绑定与方向依据；p=0.125 和样本选择限制保留，不宣称已解释全部跨数据集差距。详见开头更新及全部失败轨迹复核。没有缩写、GT 目标增强或按答案重跑。[L7]

第二优先：在定位前端影响后，评估按候选与原文条件组织的局部证据。保留所有候选资格与五个预测类别，由模型自主选目标假设和参照；inspect 返回明确对应当前候选及参照的图像/可见性，measure 保留原始数值，记录每条证据来自哪个动作、哪个 ID、哪条原文条件。未知和冲突明确保留，允许模型回退重新检查参照。与 R61/R91 单纯增加模型解释字段区分：要验证实际视觉证据对应与工具观察，而不是只要求模型写更长的绑定/比较理由。工具不替模型选最终答案；能否提分需要同期控制。[S4，S5，S7；L5；本次工具建议]

这仍是待检验的机制方向，不是已经定义好的新改进。实施前必须相对 R61/R63/R66/R69/R74/R86/R91 列明新增哪种现有工具无法返回的观测，或哪种现有轨迹无法完成的纠错动作，并找到对应缺口的实际证据。若只是增加解释、更多相似图片或更长历史，则不重复启动实验。[L5；本次实验边界]

第三优先：仅在前两步仍指向方向问题时，比较多个原文支持的观察坐标假设，不把一个方便的相机或世界 X 排序当作说话者视角。任何枚举仅返回条件几何观察，不用 GT 视角筛选正确方向，不生成目标条件描述替换原查询。[S4，S8；L5；本次工具建议]

每个新版本先在固定开发样本做成对实验，保留错误/弃答分母，不拼接历史正确答案；最终候选再做新端到端开发验证、实际小规模 Nr3D 回归、未参与开发的 ScanRefer 验证。全量约 70% 仍须实际全量证明。文献分析之后的 R93 / R94 为原因诊断，没有产生可采用的新版本，也未运行保留集或全量。[本次范围说明]

**后续工具验证：R95 未通过采用条件。** 根据方向证据缺口，独立实现模型自主选择观察原点和朝向的点云透视 inspect；不是已知说话者视角，也不改变目标候选、Top-5、模型或预算。固定16条含方向词的开发描述、共享旧解析，新旧两组各跑一次自主 ReAct。同期 @0.25 为新版9/16、对照7/16；@0.5 为8/16、5/16。@0.25 四改善、两退步，配对双侧 p=0.6875。只有 q04700 真正收到新视图，且两组均未命中；四条改善均未使用新视图。q01131 尝试18个ID，超出既有12个上限，未产生新图。故不能将分数差归因于新观察能力；未达到预记的正确数、净增和工具暴露门槛，第二轮未运行、不采用。该结果提示“提供工具”尚未解决“自主选择正确观察与参照”，但不足以否定观察视角本身的价值。32条自身实际HTTP/图片/轨迹已离线复现，77次HTTP，正式R23及原成绩未变。[L8]

**后续检查边界：R96–R98 未产生可采用的准确率改进。** R96在历史实际测量组中核验局部垂直表面，R97核验预测mask内原始网格连通片段；均为离线观察可行性分析，未接入、未发送模型请求。它们不能补回模型未选择的上游参照，也未证明新观测值得付费验证。R98将原ReAct动作改用原生函数传输，132项测试及10组成对脚本管线通过；实际接口在required选项下泛化HTTP400，在单独冻结、仅改auto的V2中一次调用成功，实际响应零网络复放通过。这只验证特定接口请求可用，尚未运行准确率对照，不构成参照绑定修复或ScanRefer提分。正式开发集仍为58.33%@0.25、52.08%@0.5，保留集与全量仍0预测。详见[开发日志](/Users/luocl/Desktop/EviGround/docs/scanrefer_improvement_log.md)与[R98接口复核](/Volumes/KESU/datasets/ScanRefer/diagnostics/r98_native_capability_v2/REVIEW.md)。

**R98后续准确率实验及说明更正。** 在准确率实验前发现V1/V2原生动作说明转换还移除了JSON输出段之后的检测分数可靠性段落；此前“全部语义说明保留”的说法范围过宽。V3独立修正并通过133测试及10组管线检查，随后在身份哈希选出的16条Multiple开发描述上与正式版同期比较：正式9/16@0.25、6/16@0.5；V3全部16条因finish将整数ID写为数字字符串、反馈仍失败而终止，0/16，不采用。该错误由新原型引入，不能解释原版58.33%的差距。32条实际响应流/PNG/轨迹与错误已零网络复放；无预测文件的错误条目，其完整工具暴露保持未知，不能因无轨迹就说没有使用工具。最小无损解码已在隔离V4通过136测试，独立R99在相同16条上重新做真实同期实验；不将旧原响应解码成新的成绩。详见[R98关闭复核](/Volumes/KESU/datasets/ScanRefer/runs/scanrefer_r98_controlled_native_actions_v1/REVIEW.md)。

**R99完整后续结果：有开发信号，未达到采用门槛。** 保留相同16条Multiple、共享旧解析并重新运行同期正式对照，V4为10/16@0.25、7/16@0.5，正式为7/16、4/16；两组均0错误/弃答，三改善、零退步，配对双侧p=.25。未达首请求前规定的>=12/16，第二轮及freshdev96未启动，不采用，也不是新的完整ScanRefer成绩。32条自身实际请求/图片/轨迹零网络复放，96次HTTP均有usage；16次ID精确解码逐次核对原HTTP，未补证据。原生接口促使更多inspect/measure，但q01013答对时仍把桌子只作为观察原点、未测椅子到桌子距离，最终使用原文未提的面积选择；不能把正确答案当作关系链已解决。后置实际候选覆盖为14/16，6个新版失败中4个有覆盖却选错、2个类别池缺失。正式完整dev96仍58.33%@0.25、52.08%@0.5；heldout186/全量0预测，约70%目标未实现。详见[R99全部轨迹与限制复核](/Volumes/KESU/datasets/ScanRefer/runs/scanrefer_r99_controlled_native_actions_v1/REVIEW.md)及[关闭核验](/Volumes/KESU/datasets/ScanRefer/runs/scanrefer_r99_controlled_native_actions_v1/CLOSURE_VALIDATION.json)。

**R100测量范围复核。** 独立检查原dev96与R99两组共23次实际measure，全部完整返回请求focus_ids内的测量对；只有两次观察坐标成员在测量组外。观察原点/朝向不必是关系参照，不能自动合入测量组。q01013的桌子距离没有返回，是模型未将桌子选入测量范围，不能据此认定工具丢失了已请求数据。这进一步区分了工具算术/数据完整性与模型选择验证对象的问题；该核查没有新的准确率结果。[范围核查](/Volumes/KESU/datasets/ScanRefer/diagnostics/r100_measure_scope_contract_v1/CLOSURE_VALIDATION.json)。

**R101组合入口验证也未通过。** 将R95同一模型选择观察位置/朝向的相机作为独立原生inspect_from入口，和R99原生传输组合；模型、Top-5、候选、预算与自主性保留。固定R95同16条方向开发描述、共享旧解析，同期新版8/16@0.25、7/16@0.5，对照7/16、6/16，两改善一退步，精确配对双侧p=1。两组均0终止错误、1弃答；新版49次成功原生调用中inspect27、measure7、finish15，inspect_from为0，故0条收到新观察视图。既没有达到预记准确率门槛，也没有实际暴露新能力，不第二轮/dev96/Nr回归或采用，不能据此否定视角机制本身。32条实际响应/PNG/轨迹均零网络复放；新增入口仍不能保证模型选择并核验所需条件。q07466仍以最小世界X解释far left，虽承认所选门在另一侧仍结束；q01131仍先固定显示器/椅子对再测左右，未核验完整纸堆关系。两条改善均未使用新视图，也不能归因于相机。正式完整dev96和Nr3D结果均未变化。[同期结果](/Volumes/KESU/datasets/ScanRefer/runs/scanrefer_r101_native_observer_v1/ROUND1_COMPARISON.json)、[复核](/Volumes/KESU/datasets/ScanRefer/runs/scanrefer_r101_native_observer_v1/REVIEW.md)。

**R102局部上下文实测未通过升级条件。** 在自主isolated inspect中保留原完整隔离图，旁置每个实例在完全相同相机/尺度下的局部场景，只标当前请求ID，不改变候选、Top-5、模型或预算。固定身份哈希选出的16条已有开发描述、共享旧解析，正式9/16@0.25、9/16@0.5，新版11/16、10/16，三改善一退步@0.25，净2、精确配对双侧p=.625；两组0错误/弃答，14条实际收到新图，三条改善都有暴露。未达到预定>=12/16、净>=3，故不第二轮/dev96/Nr回归或采用。q02814的新图能同时看到门与紫椅，但不能确认完整参照链；q06873新版根本未检查正确候选26；q06537仍固定错误left monitor假设后用距离强化。更多上下文尚不能代替正确的候选/参照选择。32条实际响应流、图片、轨迹已零网络复放；路径字符串复放差异保留失败记录并用独立补充脚本核验，不重跑推理或修答案。完整开发结果和未用数据状态均未变。[同期评分](/Volumes/KESU/datasets/ScanRefer/runs/scanrefer_r102_paired_instance_context_v1/ROUND1_COMPARISON.json)、[轨迹与图片复核](/Volumes/KESU/datasets/ScanRefer/runs/scanrefer_r102_paired_instance_context_v1/REVIEW.md)。

**R103–R106离线复核补充：先统计错误，再提出执行修改。** R103检查原96条所有7条字面含between的描述，提供模型声称三物组的几何关系，不证明参照身份或语义between真值。q05460确实未测最终引用的cabinet8/12/17，但实际测量包含cabinet候选5/7；应称最终参照变更，不能称请求遗漏了所有柜子。R105据此明确区分距离参数遗漏、组限定未验证、最终参照变更及区域未绑定。原13次measure没有明确距离参数遗漏，补充重复开发记录16次仅q01013一例；29次动作只涉及22条独特描述。这不足以支持距离遗漏成为主要提分机制。图像区域查ID的R104草稿已停止，未测试、未运行、未采用。

R106进一步检查原129次inspect/measure，4条不同描述/场景在首步超出12对象上限：q00074为19、q01131为18、q06045为13、q08271为22。下一次动作分别为12对象isolated inspect、3对象measure、8对象scene inspect、4对象isolated inspect，全部成功且无后续工具错误。q01131转measure后只核验自选椅子对相对monitor的左右，纸堆链仍未核验；q00074/q08271转isolated后最终仍用目录坐标推关系。这里能确认动作内容及未完成的检查，不能断言每个语义假设都错误，或将后续问题全归因于容量拒绝。R45早已实际扩容，开发12条从6正确降为5，未采用。因此不重试容量方案，也不从已有参数恢复推导需要额外轮数。此次0新模型调用、0新准确率结果。[执行纠错准入规则](/Users/luocl/Desktop/EviGround/docs/scanrefer_correction_mechanism_protocol.md)、[R105计数核验](/Volumes/KESU/datasets/ScanRefer/diagnostics/r105_measure_intent_frequency_v1/VALIDATION.json)、[R106动作与观察原记录](/Volumes/KESU/datasets/ScanRefer/diagnostics/r106_inspect_error_recovery_v1/AUDIT.json)。

后续只在完整预定范围内找到反复出现且可执行检查的违例后，改变一项执行细节；先在相同实际前缀和触发动作处分支比较原观察与纠错反馈，确认真实模型自主修改检查/纠错动作，再做代表性ScanRefer端到端评估。新增字段、脚本测试通过、调用次数增加或个别答对均不能代替机制证据。保持qwen3.7-plus、thinking、预测Top-5与自主ReAct，稳定ScanRefer收益之后再做真实Nr3D回归。

**来源与复核材料。**

- [S1] Chen, D. Z., Chang, A. X., & Nießner, M. (2020). *ScanRefer: 3D object localization in RGB-D scans using natural language*. ECCV. [原文](https://arxiv.org/abs/1912.08830)，任务定义及数据收集部分。
- [S2] Achlioptas, P., Abdelreheem, A., Xia, F., Elhoseiny, M., & Guibas, L. (2020). *ReferIt3D: Neural listeners for fine-grained 3D object identification in real-world scenes*. ECCV. [官方论文 PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460409.pdf)，第 4–5 页实例前提与采集语境。
- [S3] Yuan, Z., Ren, J., Feng, C.-M., Zhao, H., Cui, S., & Li, Z. (2024). *Visual programming for zero-shot open-vocabulary 3D visual grounding*. CVPR. [原文](https://arxiv.org/abs/2311.15383)，§4.1、Tables 2–3。
- [S4] Li, R., Li, S., Kong, L., Yang, X., & Liang, J. (2025). *SeeGround: See and ground for zero-shot open-vocabulary 3D visual grounding*. CVPR. [原文](https://arxiv.org/abs/2412.04383v2)，Tables 1–2、附录 Table 10、§3.2–3.3。[官方代码](https://github.com/iris0329/SeeGround)。
- [S5] Liu, Y., Mei, H., Zhan, D., Zhao, J., Zhou, D., Dong, B., & Yang, X. (2025). *View-on-Graph: Zero-shot 3D visual grounding via vision-language reasoning on scene graphs* [Preprint]. arXiv. [核验的原文版本](https://arxiv.org/abs/2512.09215v1)，Tables 1、3。官方仓库标 AAAI，本文引用实际读到的预印本。
- [S6] Huynh, C., Popov, M., Gridusov, D., & Kolyubin, S. (2026). *AgentGrounder: Zero-shot 3D visual pointcloud grounding using multimodal language models* [Preprint]. arXiv. [原文](https://arxiv.org/abs/2605.25901)，数据集说明及 Table I。[官方处理代码](https://github.com/be2rlab/AgentGrounder/tree/master/AgentGrounder/prepare_data)。
- [S7] Geng, L. (2026). *3D visual grounding based on active perception*. Complex & Intelligent Systems. [出版页](https://doi.org/10.1007/s40747-026-02514-x)，[可访问原文 PDF](https://link.springer.com/content/pdf/10.1007/s40747-026-02514-x_reference.pdf)，Tables 1–2、4、7–8。
- [S8] Du, Q., Wang, J., Li, Y., & Zhu, Y. (2026). *TDVR: Joint text disambiguation and viewpoint reasoning for zero-shot 3D visual grounding* [Preprint]. arXiv. [原文](https://arxiv.org/abs/2608.03763)，§3.3、Tables 1、3、5。PDF 标注 MM ’26（11 月举行），本次以可访问预印本核验，不以未来会议日期证明已发表。
- [S9] Geng, L., & Yin, J. (2024). *ViewInfer3D: 3D visual grounding based on embodied viewpoint inference*. IEEE Robotics and Automation Letters, 9(9), 7469–7476. [DOI](https://doi.org/10.1109/LRA.2024.3426286)，§III-B–E、Table I、§IV-F。[本机原文](</Users/luocl/Desktop/论文/参考文献/ViewInfer3D/ViewInfer3D_3D_Visual_Grounding_Based_on_Embodied_Viewpoint_Inference.pdf>)。
- [L1] 本地 Nr3D 准备代码：`/Users/luocl/Desktop/seeground/reproductions/SeeGround/local/prepare_scene.py`，及最近回归 30 场景的 `provenance.json`。
- [L2] 本地 ScanRefer 准备代码：[prepare_scanrefer_proposals.py](/Users/luocl/Desktop/EviGround/scripts/prepare_scanrefer_proposals.py)，正式冻结 R23 推理加载器/管线。
- [L3] 新输入对比：[脚本](/Volumes/KESU/datasets/ScanRefer/literature_review_20261026/input_condition_audit.py)、[完整结果](/Volumes/KESU/datasets/ScanRefer/literature_review_20261026/input_condition_audit.json)。共 128 条旧查询，不是新推理。
- [L4] 既有完整诊断：[R70 覆盖](/Volumes/KESU/datasets/ScanRefer/diagnostics/r70_candidate_bottleneck_audit.json)、[R72 错误归因](/Volumes/KESU/datasets/ScanRefer/diagnostics/r72_semantic_error_audit.json)、原 R22 dev96 指标及 R67 Nr3D 固定回归记录。
- [L5] [开发日志](/Users/luocl/Desktop/EviGround/docs/scanrefer_improvement_log.md)、R87 限定数值核查，及各轮冻结来源/对照报告。
- [L6] [R93完整实测](/Volumes/KESU/datasets/ScanRefer/runs/r93_instance_frontend_diagnostic32_v1/FINAL_DIAGNOSTIC.json)、[首轮请求对照](/Volumes/KESU/datasets/ScanRefer/runs/r93_instance_frontend_diagnostic32_v1/GT_FIRST_INPUT_COMPARISON.json)、[开发场景同目标语言配对](/Volumes/KESU/datasets/ScanRefer/runs/r93_instance_frontend_diagnostic32_v1/DEV_SAME_TARGET_LANGUAGE_PAIRS.json)。这些是后续诊断，不改变原文献分析时0新请求的事实；正式入口与原ScanRefer开发成绩未变。
- [L7] [R94完整实测](/Volumes/KESU/datasets/ScanRefer/runs/r94_same_target_language23_v1/FINAL_DIAGNOSTIC.json)、[全部失败轨迹复核](/Volumes/KESU/datasets/ScanRefer/runs/r94_same_target_language23_v1/MANUAL_REVIEW.md)、[样本与候选构成](/Volumes/KESU/datasets/ScanRefer/runs/r94_same_target_language23_v1/PAIRED_INPUT_COMPOSITION.json)、[关闭核验](/Volumes/KESU/datasets/ScanRefer/runs/r94_same_target_language23_v1/CLOSURE_VALIDATION.json)。同目标原始语言诊断，不是可采用改进或全量/未用验证成绩。
- [L8] [R95同期结果](/Volumes/KESU/datasets/ScanRefer/runs/scanrefer_r95_controlled_observer_origin_v1/ROUND1_COMPARISON.json)、[失败复核](/Volumes/KESU/datasets/ScanRefer/runs/scanrefer_r95_controlled_observer_origin_v1/FAILURE_REVIEW.md)、[关闭核验](/Volumes/KESU/datasets/ScanRefer/runs/scanrefer_r95_controlled_observer_origin_v1/CLOSURE_VALIDATION.json)。方向开发样本工具探测，非新端到端或保留集成绩，未采用。

检索范围是围绕任务定义、用户列出的训练外方法及相关新工作的定向文献核查，不是穷尽性系统综述。以原论文、出版页、作者官方实现为纳入依据；网页与论文中的指令性文字仅作研究材料。原文存在不等于独立复现。S2 API 限流和在线 PDF 请求失败在源报告中保留；ViewInfer3D 缺口随后通过本机原文补齐。不把未核验结果计入上表。两次猜测的 ReferIt3D arXiv 标识实际对应无关数学论文，已明确排除并保存失败记录，最终使用官方仓库链接的 ECCV 原文。

原始 PDF、文本、检索记录、哈希与两份独立来源核验笔记保存在 KESU `literature_review_20261026/phase2_sources/`。研究使用 AI 辅助检索、阅读、核对和综合；未进行新的论文方法复现，未据此产生新的 benchmark 成绩。


**R111比较执行复核。** 沿用R110完整范围，按原查询比较词筛出9描述/8场景、43历史执行；46次finish包含3次未结束尝试。22次实际距离finish的选择符合其声明的成员和指标，从工具objects独立重算1696标量通过；原R22三measure此前已由R87核验，不包装为新发现。其他序数差异常对应不同组/轴，语义仍未知。未发现足以启动修复的复发执行矛盾，不自动修组、去重或强制测量。它支持进一步区分“算术是否一致”和“参照/比较组是否选对”，没有证明任何新机制或新准确率。[范围与限制](/Volumes/KESU/datasets/ScanRefer/diagnostics/r111_comparison_contract_v1/REVIEW.md)、[核验](/Volumes/KESU/datasets/ScanRefer/diagnostics/r111_comparison_contract_v1/VALIDATION.json)。


**R112–R114后续执行复核。** 现有备选类别finish_check的16次历史触发（4描述/4场景；原R22一次）全部引出下一步成功inspect，反馈目标实际标号、新PNG送入随后真实请求，最终原ID均保持；596可恢复轨迹没有最终仍未满足该检查的记录。这是已有恢复观察，不是新因果对照或提分，不支持再加类似门槛。完整inspect成员筛查的18事件没有明确请求ID遗漏，词面范围限制保留；隔离图判断外部物体缺失的扩查则仍只涉及1条ScanRefer描述，另1条相关Nr3D描述单列，不据此开发新工具或重试上下文方案。三项原始响应/来源核验通过，0新模型调用/准确率/推理改动，尚无可采用新机制。[R112](/Volumes/KESU/datasets/ScanRefer/diagnostics/r112_finish_feedback_recovery_v1/REVIEW.md)、[R113](/Volumes/KESU/datasets/ScanRefer/diagnostics/r113_inspect_intent_scope_v1/REVIEW.md)、[R114](/Volumes/KESU/datasets/ScanRefer/diagnostics/r114_absence_observation_scope_v1/REVIEW.md)。


## 2026-10-27：完整原32条归因补充，不能统一解释为同类选错

R130在完整71原动作行为注释保存后，首次统一评价原R119全部32条；原预测/类别池/失败均保持。Overall20/32=62.50%@.25、19/32=59.38%@.5；Unique6/6，Multiple14/26与13/26。描述位于熟悉开发场景且部分已复用，没有新干预或同期对照，不能称相对58.33%的提分、独立新验证或泛化。

@.25的12条失败含原始proposal不足1、原类别池不足2、有覆盖未选对9；@.5分别3、1、9。有覆盖失败9条中，仅2条严格最佳几何匹配为另一同规范类别，4条其他规范类别、3条无达到.25的评估框匹配。这与原96条25/30构成不同，不能合并分母或把所有选择损失当同类混淆。chair/sofa chair粒度及残缺/合并框也可能影响分类；最佳几何匹配不等于物理语义真值。

q3132把armchair与定位table都列为Target并最终选table，原池仍有IoU近1的armchair；q1743虽观察到绿色seat7，原allowed池仍无合格目标；q6055原始最大IoU.43643，@.25命中而@.5不可达。这些分别说明语言角色、类别池和框质量需要分开。8条世界坐标方向断言中4条命中，不能把该行为本身当错误；3条显式更新均命中也不是纠错因果证明。

全部7067次IoU由另一个程序用45位Decimal/区间端点重算，最大差2.998e-15；全部71动作关键字段、原111HTTP/109usage成本、225冻结来源、33评价来源哈希与105保护文件核验通过。0新模型请求/付费推理，无正式/用户源码修改，不触及未用186，不启动Nr3D回归，不重开旧方案。本轮归因完成，稳定改善并在未用数据验证的目标尚未完成；当前没有新的机制已通过门槛。

[原预测、逐条行为、成本、归因及独立验证](/Volumes/KESU/datasets/ScanRefer/diagnostics/r130_discovery32_postrun_attribution_v1/REVIEW.md)。
