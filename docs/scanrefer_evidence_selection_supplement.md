# ScanRefer 文献补核：可见物体与可区分证据

核查日期：2026-10-26。补充原有定向文献分析，不是新的实验结果或穷尽性综述。保留 qwen3.7-plus、thinking、预测类别 Top-5、自主 ReAct 和原工具预算；本次没有模型推理、GPU/SSH 或推理代码修改。

## 两篇补充原文

| 原文 | 作者报告的 ScanRefer Overall @0.25 / @0.5 | 对本地有用的信息 | 迁移限制 |
| --- | --- | --- | --- |
| SPAZER，NeurIPS 2025，GPT-4o | 57.2 / 48.8 | 全局3D空间筛选后，用映射到同一实例ID的原始RGB帧检查外观；按候选检索证据 | 主实验为250条子集；类别回退、Top-k候选筛选和固定阶段与本地协议不同 |
| IVSGround，2026-09-04预印本，Qwen3-VL-32B推理 | 默认验证设置56.2 / 49.7；VLM-Grounder子集58.0 / 52.8 | 训练视图选择器寻找能够区分候选的关系证据；清楚看见物体与能判断其符合描述是不同目标 | 使用ScanRefer训练划分及GT判断生成训练监督，LoRA微调Qwen3-VL-8B选择器；采用Open-YOLO3D与成对淘汰流程，不能当作完全不训练或本地同条件方法 |

SPAZER原文§4.1、§4.4及Table1/5明确区分子集与完整Nr3D。Table3的52.4→63.8等组件消融发生在Nr3D子集，不能填成ScanRefer收益。Table1已从PDF页面图像核对，主ScanRefer结果是57.2/48.8。

IVSGround的Tables3–4全部是250条Nr3D子集。相同Qwen2.5-VL-72B推理器下，SeeGround54.0、SPAZER56.0、IVSGround57.6；这控制了推理模型容量，但不同完整流程仍有其他变化，不能据此把全部差值归给单一选择器，更不能转写成ScanRefer消融。Table4(b)的同流程选择器对照55.2→56.8更直接，但仍是Nr3D子集。Table1默认验证设置未在已读正文中给出明确评测条数，本地不将其认证为已独立复现的全量结果。

## 对现有诊断的实际影响

两篇补充原文支持继续区分以下问题：

1. 目标及参照是否可见、ID是否对齐。
2. 图或测量是否包含能排除同类对手的那条条件。
3. 模型是否使用了该证据，并在冲突后改变参照、比较组或观察。

例如两个候选都是白色椅子，拍清两把椅子本身不能判定哪把属于“显示器前、纸堆旁那一组”。必须检查同一组关系。这个例子说明研究目标，不是自动绑定参照的算法，也不是已经完成的自主纠错。

本地此前已经尝试RGB、局部上下文、角色字段、方向入口和事实范围反馈，没有稳定收益；不能因为找到新论文就重做同一变体。尤其不能照搬候选回退/淘汰、训练新模型或增加阶段后仍宣称原协议不变。按用户更新后的目标，下一项研究不必先找到程序违例，但须明确哪种新的证据能力或假设更新能补足原文判断条件，并分别验证真实机制和未用样本准确率；当前预算和停止条件见[研究协议](/Users/luocl/Desktop/EviGround/docs/scanrefer_evidence_research_protocol.md)。

现有R120角色清单修复是可复核的代码正确性原型；15条属性丢失描述中13条旧模型仍主动请求检查对应属性，因此暂不能把字段恢复当作关系辨别改善。本补核没有改变这一准入判断，也没有产生新的准确率。

## 为什么优先Multiple

当前96条开发集：Unique19/21、Multiple37/75，总计56/96。保持Multiple不变，即使修好全部Unique，也只有58/96=60.42%。若Unique仍正确19条，要接近70%至少需总正确67条，即Multiple48/75=64%，比当前净增11条。这里仅做开发集内的算术规划；分组未必对应推理候选数量，不能外推成全量预测或泛化结论。

这一规模说明，继续恢复几条字段、消除少数终止错误或只改善框阈值都不足以解释达到70%的路径。主要研究对象应是有合格候选时如何稳定选对实例，同时单独监测proposal与类别覆盖，防止把检测损失算成推理损失。

## 原始材料

- SPAZER：[NeurIPS原文](https://proceedings.neurips.cc/paper_files/paper/2025/file/f1ea808562a3cd5a2a0ea94f3dee675a-Paper-Conference.pdf)，§3、§4、Tables1/3/5。
- IVSGround：[固定v1原文](https://arxiv.org/html/2609.04741v1)，§3.3、§5.1及Tables1/3/4。当前按预印本引用。
- 原文下载来源、字节数和SHA256：[RETRIEVAL.json](/Volumes/KESU/datasets/ScanRefer/literature_review_20261026/phase3_sources/RETRIEVAL.json)。
- 本地错误分母：[R72后置审计](/Volumes/KESU/datasets/ScanRefer/diagnostics/r72_semantic_error_audit.json)；候选覆盖的重算边界见[原简版分析](/Users/luocl/Desktop/EviGround/docs/scanrefer_diagnosis_brief.md)。

本补核使用AI辅助检索、阅读及综合；作者报告结果未在本地复现。
