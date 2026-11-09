# ScanRefer 对比数据与运行命令

核查日期：2026-10-24。下表为各论文自行报告的结果，单位均为 %；不代表完全相同的评测数据或实验条件。`—` 表示所核查原文未报告该项。

2026-10-26 文献与本地输入复核：见 [Nr3D 与 ScanRefer 差距分析](/Users/luocl/Desktop/EviGround/docs/scanrefer_nr3d_literature_analysis.md)。已补核 ViewInfer3D 本机原文；本地 58.33% 仍只是反复使用的 dev96，不能与论文全量结果直接排名。

2026-10-25 更新：`scanrefer_ready.sh` 已切换到正确使用 GT 框 IoU 评分的 EviGround 入口，冻结源码为 `r23_scores_recovery_v1`，默认仍为 qwen3.7-plus、预测类别 Top-5、自主 ReAct、mask NMS 0.8。新全量输出目录为 `runs/scanrefer_r23_top5_thinking_single_pass`；原 r18 源码和输出保持独立，`run_scanrefer_r18.sh` 只用于历史复现。当前 70% 目标尚未达到，尚未启动新全量推理。

## 论文结果

| 方法 | 模型 | Unique@0.25 | Unique@0.5 | Multiple@0.25 | Multiple@0.5 | Overall@0.25 | Overall@0.5 | 来源 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ZSVG3D | GPT-4 系列（不同实验版本须区分） | 63.8 | 58.4 | 27.7 | 24.6 | 36.4 | 32.7 | 原文表 2；不据后续论文统一表头认定全部为 Turbo |
| ViewInfer3D | GPT-4-0613 + CLIP | — | — | 56.25 | 42.83 | 60.31 | 47.20 | 原文表 I，期刊页 7473 |
| SeeGround | Qwen2-VL-7B | — | — | — | — | — | — | 所核查原文只给此模型的 Nr3D 结果，不能照搬到 ScanRefer |
| SeeGround | Qwen2-VL-72B | 75.7 | 68.9 | 34.0 | 30.0 | 44.1 | 39.4 | 原文表 1 |
| VoG / View-on-Graph | Qwen2-VL-2B | 69.1 | 63.1 | 25.3 | 21.9 | 35.9 | 31.9 | 原文表 1 |
| VoG / View-on-Graph | Qwen2-VL-72B | 78.6 | 71.5 | 34.0 | 30.4 | 44.8 | 40.3 | 原文表 1 |
| AgentGrounder | Qwen3-VL-32B-Instruct | 80.3 | 73.7 | 36.6 | 31.7 | 47.2 | 41.9 | 原文表 I；实验节说明仅抽取 1,000 条分层样本 |
| 3D visual grounding based on active perception | GLM-4.5V + Mask3D + CLIP 等 | — | — | 59.84 | 49.46 | 63.52 | 54.88 | 原文表 1，PDF 第 10 页 |
| 本地 r22 v2，仅固定开发 96 条 | qwen3.7-plus | 90.48 | 85.71 | 49.33 | 42.67 | 58.33 | 52.08 | 样本结果，不能当作全量；含 2 错误、2 弃答 |
| r31 框一致性去重，仅固定开发 96 条 | qwen3.7-plus | 85.71 | 80.95 | 48.00 | 44.00 | 56.25 | 52.08 | 0 错误、2 弃答；未超过 r22，未采用 |
| 本地方法全量 Top-5 单轮，开启思考 | qwen3.7-plus | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 全量未启动；使用以下新入口 |

补充：VoG 表 1 还报告了其复现的 SeeGround / Qwen2-VL-2B：59.9、55.0、20.7、18.4、30.2、27.2。这是 2B 的数据，不能填进 7B 那一行。

本地原论文路径：

- ZSVG3D：`/Users/luocl/Desktop/论文/参考文献/ZSVG3D/Yuan_Visual_Programming_for_Zero-shot_Open-Vocabulary_3D_Visual_Grounding_CVPR_2024_paper.pdf`
- ViewInfer3D：`/Users/luocl/Desktop/论文/参考文献/ViewInfer3D/ViewInfer3D_3D_Visual_Grounding_Based_on_Embodied_Viewpoint_Inference.pdf`
- SeeGround：`/Users/luocl/Desktop/论文/参考文献/SeeGround/2412.04383v2.pdf`
- VoG：`/Users/luocl/Desktop/论文/参考文献/View-on-Graph（VOG）/22807-AAAI26.LiuY-CV.pdf`
- AgentGrounder：`/Users/luocl/Desktop/论文/参考文献/AgentGrounder/2605.25901v1.pdf`
- Active perception：`/Users/luocl/Desktop/论文/参考文献/3D visual grounding based on active perception/s40747-026-02514-x_reference.pdf`

## 评测口径与数据来源

ScanRefer 报告预测框与目标 GT 框的 3D IoU，按 0.25、0.5 两个阈值统计 Unique / Multiple / Overall。预测实例 ID 与 GT object_id 属于不同编号空间，不能用 ID 相等作为 ScanRefer 准确率。

本地 Nr3D 资料使用 GT 实例几何和预测类别。若直接复用这些实例，得到的是 **GT-proposal ScanRefer**，应单独标注；不能与 SeeGround / VoG 的 Mask3D 主表当成同一协议比较。

2026-10-24 更新：已从 VLM-Grounder 作者的公开仓库下载原始格式 `ScanRefer_filtered_val.json` 副本，含 9,508 条描述、141 个场景，保存到 `/Volumes/KESU/datasets/ScanRefer/`。同目录已有 ScanNet 标签映射，下载来源、固定提交和 SHA-256 记录在 `DOWNLOAD_SOURCE.json`。这是研究仓库公开副本，并非通过 ScanRefer 官方申请邮件取得的下载。

已补齐 141 个 ScanRefer 验证场景的四类原始文件（网格、轴对齐信息、superpoints、aggregation），保存在 KESU 的 `raw_scans/`。`evaluation_gt/` 保存独立评估框；9,508 条查询的目标编号均存在，3,603 个作者公开框与本地计算框逐一核对，最大坐标差约 0.0000013 米。记录见 `raw_inputs.json` 和 `PREPARATION_STATUS.json`。

Mask3D 官方验证集权重、固定提交的源码、原 r8 的 PointNeXt/CLIP/六个 RGB 分类头、词表和锁定选择均已保存到 KESU 的 `models/` 并校验 SHA-256。全部 141 个场景的 Mask3D 与 r8 GPU 预处理已完成，共 14,694 个预测实例；新生成的实例点云和 Top-5 分类导出已回传 KESU 并通过本机全量数据校验，记录见 `GPU_PREPROCESSING_VERIFIED.json`。

已有 `olt.zip` 包含作者 ScanRefer 的 GT / 预测框和标签，但不包含用于本方法观察、测量和重新分类的逐实例点云、掩码与 Top-5 分数。仅有该 ZIP 不能当作当前 r8 分类输入。另一个作者结果 ZIP 有 9,498 条结果；它不是官方原始标注文件，不用它冒充完整官方验证集。

已准备的数据与对应约束：

1. `ScanRefer_filtered_val.json` 与 `scannetv2-labels.combined.tsv` 已下载到 KESU，清单转换可直接运行。官方原始分发入口仍为 ScanRefer 使用条款申请表；当前使用的公开副本来源见 `DOWNLOAD_SOURCE.json`。
2. 对应 ScanRefer 验证场景的 Mask3D 实例点云。`scene.npz` 中 `ids`、`scene_instances.json` 与分类器的 `bbox_id` 已逐项核对为同一套预测实例；**不能混入 GT 实例几何作为推理输入**。重叠掩码采用下文所述的固定几何保留规则。
3. 对这些预测实例运行原分类器后导出的 r8 Top-5 数据。全部 `scene.npz` 的哈希已与分类导出记录核对一致，14,694 个实例的框、点数、Top-5 排名及分数均已核对；不是旧 Nr3D GT 实例的分类导出。
4. 每个场景的 `objects_gt.json`：仅供协调进程评估，按官方 object_id 保存 `[cx, cy, cz, sx, sy, sz]`。预测框、GT 框和渲染点云必须在同一轴对齐坐标系下。

## 预处理与验证进度

- `scripts/prepare_scanrefer_raw.py`：复用已有缓存并补齐缺失的官方文件，构造独立评估框；已完成全部 141 个场景。
- `scripts/prepare_scanrefer_proposals.py`：复用冻结 r18 附带的 Mask3D 推理函数，使用 ScanNet200 validation checkpoint。重叠掩码通过复制共有顶点保留完整几何，不通过 GT 匹配筛选实例；仅剔除少于三个点或包围框有非正尺寸的退化掩码。
- `scripts/classify_scanrefer_r8.py`：原 r8 B/M 的九次采样/旋转、六个冻结 RGB 头、0.4/0.6 融合，导出原 r18 可读取的 Top-5 格式。没有重新训练或用 ScanRefer 标注选择参数。
- CPU 数值核对已覆盖原 404 + 3,413 个物体的缓存特征，Top-5 排名全部一致；融合分数最大差约 0.0000012。GPU 编码器进一步核对了历史 3,413 个缓存物体中的首、中、末三个物体，每个物体九次编码的最大绝对差均为 0；真实 Mask3D 实例上的完整 r8 冒烟测试及全部 141 场景的 GPU 预处理也已完成。
- `scripts/rebuild_scanrefer_scenes.py`：从服务器实际生成的预测掩码在本机重建点云，全部 141 个 `scene.npz` 均与服务器原始文件的 SHA-256 完全一致。节省了约 1 GB 点云下载，未以 GT 实例替代预测实例。记录为 `LOCAL_SCENE_RECONSTRUCTION.json`。
- 14 项相关测试通过；GT/预测 ID 空间分离、固定评测分母、空预测/错误计失败、重叠掩码几何、API 配置缺失时在批次启动前退出，以及私密配置文件加载均有检查。

本次使用自行运行的 Mask3D ScanNet200 前端，不声称与 SeeGround 作者发布的 proposal 集合完全相同；论文表中的数据仍为作者各自的结果。

## 当前 GPU 环境与执行状态

2026-10-24 通过 SSH 技能连接 `scanrefer-westb-16520`，实测为 RTX 5090 32GB。复用了历史 Mask3D / classifier-heads CUDA 环境及模型缓存；没有重新训练。工作目录为 `/root/autodl-tmp/scanrefer_20261024`，564 个原始文件均已核对 SHA256。全部 141 场景的 Mask3D 耗时 1,818.57 秒、r8 耗时 340.78 秒，GPU 预处理合计 2,159.35 秒，约 36 分钟；不含环境准备、传输及本机校验时间。

r8 已在真实 Mask3D 预测实例上通过九次 PointNeXt 编码、四视角 CLIP、六个 RGB 头及融合自检。运行时使用独立 `inference-source` 副本，恢复历史部署中的路径修复及分类推理跳过重建模块的导入配置，并沿用历史 Python 依赖路径；模型权重和前向计算不变，原冻结源码不修改。兼容配置、242 个模型源码哈希及 CUDA 算子哈希已回传 KESU 的 `gpu_runtime/R8_RUNTIME_COMPATIBILITY.json`；同目录保留 GPU 运行状态、日志、冒烟测试和编码器对照记录。

全部 GPU 输出已回传并通过本机数据校验。后续推理在 Mac 上调用 Qwen API，不再依赖 GPU 服务器。已保留的 r22 完整固定 96 条为 58.33%@0.25、52.08%@0.5。2026-10-25 的后续 r29 紧凑输入消融独立完成相同 96 条，得到 57.29%、52.08%，0 请求错误、2 弃答计入分母；输入 token 中位数减少约 25.75%，但未提高准确率，未切入正式入口。用户已明确保持 qwen3.7-plus，后续继续改进输入与工具。完整结果见改进日志；独立保留集和新全量尚未开始模型推理。

2026-10-25 的后续工具实验仍保持同一模型、完整 Top-5 与自主 ReAct：r37 可选视觉问答、r38 隔离图紧凑取景加对侧图，各自独立完成同一固定 12 条开发探针，均为 7/12 @0.25、6/12 @0.5，低于原同批 8/12、7/12。两轮均为 0 请求错误，实际新输入与源码哈希审计通过，但不采用、不扩大全量；12 条不是完整数据集成绩。正式入口继续 r23，ScanRefer 接近 70% 的目标尚未达到。Nr3D 输入回归与 r23 一致，69.66% 仍是历史全量成绩。记录见 `docs/scanrefer_improvement_log.md` 与各轮 `FINAL_VALIDATION.json`。

随后 r39 收紧 measure 返回范围的固定 13 条探针为 6/13、6/13（原同批 5/13、5/13），未达到预记至少净增 2 条的推进条件，不采用。r40 仅离线核对统一分位框，虽 @0.25 多覆盖 1 条，@0.5 丢失 3 条，未修改推理或预测框。

r41 将既有解析器的带原文引句关系、属性角色/否定、观察者与未解析子句补充给 ReAct，原模型、候选、Top-5、图片、动作与预算不变。独立固定 12 条多句描述探针为 8/12、8/12，较原同批 6/12、6/12 有 2 改善、0 退步。随后同一冻结源码在新目录 `runs/scanrefer_r41_language_dev96_v1` 独立完成整个 dev96，未导入探针答案：56/96 = 58.33%@0.25、52/96 = 54.17%@0.5，2 请求错误和 1 弃答计失败；@0.25 为 6 改善、6 退步，未超过原 56/96 主指标。因此不采用 r41，正式入口仍 r23。93 条有效输入和全部 96 条独立 IoU 复算通过，不能将小探针 8/12 或开发集成绩填作全量结果，70% 目标仍未达到。

## 本机固定路径与命令

本组命令使用冻结 r23 源码，保留 Top-5、思考开启、8192 token floor、工具预算 3、8 workers / 8 API 并发。相对已完整测试的 r22 v2，r23 仅增加 knapsack/backpack 精确同义词与服务端 JSON 生成中断时的同请求单次恢复，不含未奏效的方向提示、atlas、相机接口或 finish 校验探针。启动器读取原冻结目录的 `.env` 后，再读取本机已有的 `~/.config/seeground/qwen-token-plan-20261022.env`；可用 `SEEGROUND_ENV_FILE` 显式指定其他私密配置文件。API 密钥不写入代码或 KESU 数据包。输出也放在 KESU。

标注和清单已准备完成；以下命令无需占位符，可以重新生成相同清单：

```bash
cd /Users/luocl/Desktop/EviGround
/Users/luocl/Desktop/seeground/reproductions/SeeGround/.venv/bin/python scripts/scanrefer.py prepare \
  --annotations /Volumes/KESU/datasets/ScanRefer/ScanRefer_filtered_val.json \
  --labels /Volumes/KESU/datasets/ScanRefer/scannetv2-labels.combined.tsv \
  --output /Volumes/KESU/datasets/ScanRefer/scanrefer_val_manifest.json
```

最终入口已固定真实目录。启动器在创建批次前先调用冻结客户端的 `_settings()`，检查 API 地址、模型和密钥；检查失败会立即退出，不再为每条查询写出 null。`preflight` 仍不发送 API 请求，不等于服务端连通性测试。`run` 会产生 Qwen API 费用；GPU 服务器已不再需要。

```bash
bash /Users/luocl/Desktop/EviGround/scripts/scanrefer_ready.sh preflight
bash /Users/luocl/Desktop/EviGround/scripts/scanrefer_ready.sh run
bash /Users/luocl/Desktop/EviGround/scripts/scanrefer_ready.sh report
```

已验证的固定目录：

- 预测实例：`/Volumes/KESU/datasets/ScanRefer/scanrefer_mask3d_prepared_scenes`
- r8 导出：`/Volumes/KESU/datasets/ScanRefer/scanrefer_mask3d_r8_verified_export`
- 推理结果：`/Volumes/KESU/datasets/ScanRefer/runs/scanrefer_r23_top5_thinking_single_pass`

前两个目录均已完整生成并核对。启动器会检查所需文件、去重 policy 和冻结源码，不再要求填写 `/absolute/path/to/...`。保持 KESU 挂载；启动器通过 caffeinate 保持 Mac 唤醒，仍需要网络。再次使用同一命令会恢复新输出目录中全部已保存评估，不仅复用答对的样本；不导入 r18、r19、r22 或开发探针的答案。

2026-10-24 修复启动配置：原冻结 `.env` 只有地址和模型，没有密钥。第一次实际启动中 57 条查询均在请求发出前因配置缺失而失败，另 17 个目录在中断启动时生成，全部没有 `prediction.json`。该次输出已整体改名保留为 `runs/scanrefer_r18_top5_thinking_single_pass_startup_failed_20261024_221741`，诊断见其中的 `STARTUP_FAILURE_DIAGNOSTIC.json`；新正式运行不复用这些失败缓存。此次修复不改冻结 r18 算法或预测实例。

修复后已用第一条原始查询完成真实 API 试跑，输出位于 `runs/scanrefer_r18_api_fix_smoke1_20261024`：4 次调用均成功，返回 `predicted_id: 2` 及完整预测框，用时约 181 秒。该例 IoU 为 0，仅验证运行链路，不代表准确率；不复制进正式实验。记录见 `RUNTIME_API_VERIFIED.json`。正式输出目录已重新通过全量预检，9,508 条全部待运行、零缓存；直接使用相同的 `run` 命令即可启动。

`scanrefer_metrics.json` 中 `unique / multiple / overall` 各有 `acc25`、`acc50`，以 0–1 小数保存，乘 100 填表。只有 `complete=true` 且 `pending=0` 才是全量结果；错误、null、待完成样本都保留在固定总分母。旧 r18 的 `correct`、`accuracy`、`category_candidate_recall` 和 `target_category_rank_groups` 依赖 ID 对比，不用于预测框 ScanRefer 的最终成绩。

官方分组定义：ScanRefer `lib/dataset.py` 的 `_get_raw2label`、`_get_unique_multiple_lookup` 和 `data/scannet/model_util_scannet.py` 的 18 类定义。当前清单为 Unique 1,845 / Multiple 7,663；与使用不同类别映射的作者统计不直接混用。

2026-10-25 的 r42 原始网格表面工具独立 color12 为 8/12 @0.25、7/12 @0.5，与原同批持平，1 改善、1 退步。7 条实际使用 9 张 mesh 图，像素/相机/完整五类/源码/输入与独立 IoU 审计通过，但未达到请求前记录的推进门槛，因此不采用，正式入口与完整 dev96 参考不变。

r43 保持同一模型/五类/自主 ReAct，新增原始传感 RGB 裁剪的固定 color12 仍为 8/12、7/12，逐例正确性与原同批相同。9 条实际调用、12 张图和 51 个原始源裁剪的独立像素/深度/位姿审计通过，但没有准确率收益，因此不采用。原始传感数据与新输入缓存均保存在 KESU，本次无 GPU 服务器处理；正式入口仍 r23，完整 dev96 参考仍为 58.33%@0.25、52.08%@0.5。


r44 共享原始 RGB 场景视角的固定 color12 为 8/12 @0.25、7/12 @0.5，与原同批逐例正确性相同。5 条实际使用 10 张新图，原始 sensor JPEG/深度/位姿、42 个预测实例标号端点、完整输入与独立 IoU 审计通过；0 错误、0 弃答，但没有准确率提升，因此不采用、不扩大 dev96，正式入口仍 r23。该 12 条探针的 66.67% 不能作为完整 ScanRefer 成绩，70% 目标仍未达到。


r45 修复大组 inspect 被12对象上限拒绝，并保留每6个一张图、模型自主动作与预算3。独立固定容量12为5/12 @0.25、3/12 @0.5，低于原同批6/12、4/12，0请求错误、1弃答；2次实际大组检查成功，全部25张工具组图与跨步骤文件身份通过审计，但准确率未改善，因此不采用。

r46 独立组合对照已完整结束：共同检验预测mask/bbox一致去重、可读ID标号、既有带原文引句条件与完整inspect分组。固定24条开发探针为13/24 = 54.17%@0.25、10/24 = 41.67%@0.5，原同批15/24、11/24；0请求错误、0弃答，2改善、4退步，约831秒。未达到首个请求前规定的18/24、11/24门槛，因此拒绝采用、不扩大dev96，正式入口仍r23。24条实际输入、48张初始图、30次inspect的34张组图、24场景policy与独立IoU复算通过；本轮实际无超过12个对象的inspect。11条失败中10条有合格候选但选错，1条原始框缺失，说明这些组合修复仍不足以解决候选比较问题。四份最终报告保存在 `/Volumes/KESU/datasets/ScanRefer/runs/scanrefer_r46_interactions24_v1`；正式完整dev96参考仍58.33%@0.25、52.08%@0.5，70%尚未达到。这些开发结果不能填作完整ScanRefer成绩。

r47 从r23独立检验此前被渲染过滤的原始未分配扫描几何：初始图和scene inspect左侧原图完整保留，右侧同机位只增加原始id=-1的RGB/XYZ背景，不赋类别或可选ID。模型、完整Top-5、动作schema和预算3不变。固定24条完整结果为16/24 = 66.67%@0.25、12/24 = 50.00%@0.5，原同批15/24、11/24；3改善、2退步，0错误、0弃答，约561秒。未达预记17/24主指标门槛，不采用、不扩大dev96。46张初始双面板、16张scene双面板及全部原图/空背景图、24场景原始预测mask补集、完整输入与独立IoU审计通过；124原测试和5专项测试通过。四份最终报告在 `runs/scanrefer_r47_context24_v1`，正式入口仍r23，保留集和全量未运行，70%尚未达到。

r48只增加可错的原文主语角色提示，固定24完整结果为12/24 = 50.00%@0.25、9/24 = 37.50%@0.5，低于原同批15/24、11/24；1改善、4退步，0错误、0弃答，约461秒。目标/参照混淆的q00804被修复，但泛用提示未改善整体表现，因此不采用、不扩展dev96。124原测试、6专项测试、真实ScanRefer/Nr3D完整请求回归以及实际候选/Top-5/模型/源码与独立IoU审计通过；四份报告在`runs/scanrefer_r48_subject24_v1`。正式入口仍r23，保留集与全量未启动，70%尚未达到。

r49进一步只在当前解析含多个Target时提供不含ID列表的主语假设，单Target输入保持不变。新固定12包含全部3条旧多Target描述及9条按哈希选择的单Target对照，完整结果7/12 @0.25、5/12 @0.5，与原同批持平，2改善、2退步，0请求错误、1无候选弃答；未达预记8/12推进门槛，因此不采用、不扩大dev96。11条实际ReAct/完整五类/候选/模型/源码与独立IoU核验通过，两条实际启用提示。四报告在`runs/scanrefer_r49_ambiguous12_v1`；正式入口和完整dev96参考均不改变。

补充复现性核验：在r49已完成样本的7条无提示对照中，重建的完整首轮客户端消息、system、图像、schema与参数均和原基线逐项相同，仍有4条最终ID不同。核验没有调用新模型、没有读取GT作推理输入；记录于`diagnostics/r49_initial_input_repeatability_audit.json`。这说明现有独立轨迹有实际运行波动，不能把小探针的所有变化视为方法的因果效果；该记录不提供新的全量准确率。

r50采用固定同一份qwen语言解析、每组两次全新自主ReAct的受控开发比较。原r23两次均7/12 @0.25、6/12 @0.5；首轮追加原始RGB的两次为8/12、8/12及7/12、6/12，累计15/24比同期14/24只多1次主指标正确决定，未达首次请求前的推进门槛，因此不采用、不扩大dev96。24次决定只有12条独特查询，不能填入端到端ScanRefer主表。48条独立IoU零偏差、实际图像与原sensor来源核验通过、冻结源码未变；四组完整四报告与`CONTROLLED_COMPARISON.json`在`runs/scanrefer_r50_controlled_initial_rgb_v1`。正式入口仍r23，58.33%仍是完整开发参考，70%目标未达到，Nr3D历史69.66%不当作新复跑。

r51在完全相同的原照片/选帧上增加预测mask的实测深度支持投影，保留原图、Top-5和自主ReAct。同期点标号两次7/12、8/12，新投影两次均7/12（本轮@0.25与@0.5相同），累计14/24低于15/24，因此拒绝，不扩大dev96。48条独立IoU零偏差、实际图片逐字节匹配请求前的独立像素验证，42包源码及5个冻结脚本未变；四组完整报告与`CONTROLLED_COMPARISON.json`在`runs/scanrefer_r51_controlled_support_v1`。这些重复开发决定不是新的端到端或全量准确率，正式入口和58.33%@0.25参考均不改变；70%尚未实现。

另做2次同模型、无查询/类别提示的原图读图诊断，发现101的“灰色面板”描述与完整ReAct的“tan pillow”冲突，但未识别完整物体或找到正确目标，不计入任何准确率。报告为`diagnostics/r51_query_free_photo_read_v1/DIAGNOSTIC_SUMMARY.json`；它仅支持继续检查上下文中的物体解释，不证明只由分类先验导致错误，也不改变正式运行命令。


r52在自主isolated inspect后追加同一qwen3.7-plus的无查询/无分类先验RGB观察，124原测试+4RGB+6观察测试与真实请求回归通过。相同冻结解析的12条开发描述各两次完整对照：控制15/24@0.25、13/24@0.5，新组15/24、14/24，主指标无净增，未达预记至少18/24且多3次，拒绝采用。24次决定只有12条独特描述，不是独立24样本或新端到端成绩。48条独立IoU、48张实际工具图及100个原sensor源裁剪核验通过；新组增加18次盲读调用但未证明稳定收益。完整结果在 `runs/scanrefer_r52_controlled_rgb_observer_v1/CONTROLLED_COMPARISON.json`。正式入口/完整dev96参考不变，Nr3D本轮仅输入回归，70%目标未达到。
r53只补回源表已有完整五类融合score、类型和非概率说明，不改模型、候选、图片、动作或预算。124原测试+3专项测试、ScanRefer实际请求分数核对和Nr3D完整请求回归通过。固定24条描述、两组各两次全新ReAct已全部结束：控制两次为13/24、16/24 @0.25及10/24、12/24 @0.5；新组两次均13/24 @0.25，@0.5为9/24、11/24。累计控制29/48、22/48，新组26/48、20/48，未达首次请求前EXPERIMENT的推进条件，因此拒绝采用、不扩大dev96；每组48决定只有24条独特描述，不是新端到端准确率。96条独立IoU、实际首轮原字段/图片/schema/完整五类/候选与20290处分数、4058条类型记录核对通过；四组四报告及CLOSURE_VALIDATION齐全。成本包含失败响应usage，新组1次超时usage未知。完整输出在 `runs/scanrefer_r53_controlled_scores_v1`；正式r23命令和完整dev96的58.33%@0.25、52.08%@0.5参考不变，Nr3D本轮只有输入回归，70%尚未达到。此前本段开始记录补于首轮运行期间，不冒充预记来源。

r54检验在自主isolated inspect中追加由原网格三角面和实测深度共同支持的原sensor RGB表面裁剪，原点云图、原照片、Top-5与动作预算均保留。134项测试、真实ScanRefer/Nr3D输入回归、1276个显示裁剪的原sensor重建与20416个独立射线像素抽查通过。固定color12、相同旧语言解析的双次对照已完成：原RGB和新组每次均8/12 @0.25、7/12 @0.5，各累计16/24、14/24，净改善0；未达到首次请求前18/24且净增至少3次的门槛，拒绝、不扩大dev96。每组24决定只有12条独特描述，不是新的端到端成绩。48条独立IoU零偏差、74张实际图像重建、92个原sensor裁剪和完整输入/五类/候选/模型/预算/GT/源码审计通过；四组四报告及CLOSURE_VALIDATION保存在`runs/scanrefer_r54_controlled_surface_rgb_v1`。正式r23命令和完整dev96参考58.33%@0.25、52.08%@0.5保持，Nr3D只有输入回归，保留集/全量均0个prediction，70%目标尚未达到。

r55从正式r23独立检验measure的可选双参考between几何，参考由ReAct自主绑定，不自动排名或选目标，原Top-5/三动作/预算3/模型/图片保持。124原测试+6专项测试、真实ScanRefer三步输入与pipeline GT隔离、Nr3D完整三步请求回归通过。固定7条between+7条非between不同场景、同一旧语言解析的双次对照已结束：两组每次分别7/14、6/14，两阈值均累计13/28，净收益0，未达预记20/28且净增3次门槛，拒绝、不扩大dev96。每组28评估只有14条独特开发描述，不是新端到端成绩。4次实际新测量/11行独立复算零偏差；56条独立评分、55条成功完整输入与原测量字段核对通过。控制首轮有1条连续两次HTTP400 JSON生成失败，保留在分母；该条没有完整成功输入trace，不冒称56条均已验证。四组四报告及CLOSURE_VALIDATION齐全，结果在`runs/scanrefer_r55_controlled_between_v1`。正式r23命令、完整dev96的58.33%@0.25/52.08%@0.5与70%未达状态保持；Nr3D本轮仅输入回归，保留集/全量仍0个prediction。



R56（2026-10-25）已完成并拒绝：保留R51同一原始sensor帧/可见ID/标号/原照片，将旧支持顶点单元与原预测mask完全包含三角面的实测深度支持取并集。固定color12与同模型旧语言解析，两组各两次全新自主ReAct。原顶点组8/12、7/12 @0.25（合计15/24），连续表面组7/12、7/12（14/24）；@0.5合计14/24、13/24，未达预记18/24且净增3次门槛，不扩大dev96、不改正式入口。48条预测无错误/弃答，48完整轨迹109次请求本地回放、92张实际RGB图及独立IoU审计通过。375原始缓存sensor帧来源逐项验证，23张显示图完整重建、3040确定性原生像素独立射线抽验通过；43包源码及6冻结脚本未变。全部材料在`runs/scanrefer_r56_controlled_shared_surface_v1`与`diagnostics/r56_actual_input_replay/REPLAY_VALIDATION.json`。小探针的24决定只有12条独特描述，不是端到端成绩。正式完整dev96仍58.33%@0.25、52.08%@0.5；Nr3D本轮仅输入回归，历史69.66%非新复跑；保留集/全量仍0 prediction，约70%目标未达。


R57（2026-10-25）已完成并拒绝：相同R56观测支持、照片/帧/标号，面内填色改保留原生材料色的中性轮廓，并改准确的照片说明。固定color12与同一旧qwen语言解析，两组各两次全新自主ReAct。填色对照8/12、6/12 @0.25（14/24），轮廓8/12、7/12（15/24）；@0.5合计13/24、14/24，净增1次未达预记18/24且净增3次门槛，不扩大dev96、不改正式入口。48条无终止错误/弃答，独立IoU零偏差；48完整轨迹111个成功记录请求、92张实际RGB图的源码/图片回放通过。前置request检查的动态helper覆盖限制与后置回放隔离修正已如实单列`diagnostics/r57_request_validation_addendum.json`；真实各组独立进程/PYTHONPATH及图片来源正确，43包源码与5冻结脚本未变。输出在`runs/scanrefer_r57_controlled_native_color_v1`。24决定只有12独特开发描述，不是端到端成绩；正式完整dev96仍58.33%@0.25、52.08%@0.5，Nr3D仅输入回归、历史69.66%非新复跑，保留集/全量0 prediction，约70%目标未达。


R58（2026-10-25）已完成并拒绝：从正式r23独立在模型主动measure后增加观测mask顶点距离，旧system/schema/初始输入/图片/Top-5/候选与预算3不变。124原测试+8距离测试、真实ScanRefer/Nr3D三步回归通过；42包文件/7脚本首请求前冻结。相同冻结语言解析、固定color12两次同期对照：R23两次均7/12 @0.25、6/12 @0.5（累计14/24、12/24）；R58两次均8/12 @0.25，@0.5为7/12、8/12（累计16/24、15/24），未达至少18/24且净增3次门槛，不扩大dev96、不改正式入口。两次主指标净增均是未调用新测量的同一枕头描述，不能证明工具收益；新组4次实际测量的122对观察/67独特对全点独立复算零偏差，48条独立评分/完整轨迹128次记录请求回放通过。AppleDouble非目录枚举的后置审计修正单列diagnostics/r58_post_audit_addendum.json，冻结文件未改。19个主报告及CLOSURE_VALIDATION保存在runs/scanrefer_r58_controlled_observed_points_v1，全部进程结束。每组24决定仅12独特开发描述，不是新端到端成绩；完整dev96仍58.33%@0.25、52.08%@0.5，Nr3D只有输入回归、历史69.66%非新复跑，保留集/全量0 prediction，约70%目标未达。


## R59 测量平面图同期对照（2026-10-25，未采用）

- 固定measure13、两次独立重复：R23与R59每次均6/13 @0.25和6/13 @0.5；累计均12/26。每组只有13条独特条件开发描述，不替代完整端到端成绩。
- 52条独立IoU、139次成功记录请求的实际消息/选项哈希和图片/完整轨迹回放、19张实际测量图几何均通过。0终止错误/0弃答；新组2次失败尝试已恢复但usage未知，完整成本不可补零。未达首请求前推进门槛，不扩大dev96、不切换入口。
- 正式完整dev96参考仍58.33%@0.25、52.08%@0.5；Nr3D 69.66%是历史全量，近期输入回归不冒充新准确率。目标未达到，保留集/全量仍0预测。
- R60参数行动历史仅零请求原型：125项测试及真实ScanRefer/Nr3D三步输入检查通过，尚未冻结/实跑/采用。未经同期模型结果验证，不能作为提分或70%版本。
- 正式可用命令保持本文件原有r23入口。R59结果位于`/Volumes/KESU/datasets/ScanRefer/runs/scanrefer_r59_controlled_measured_maps_v1/CONTROLLED_COMPARISON.json`。


## R60 行动历史同期对照（2026-10-25，未采用）

- 同期两次控制为7/12、7/12 @0.25和6/12、5/12 @0.5；新组均8/12、7/12。累计新组16/24、14/24，控制14/24、11/24，主指标净增2次但未达首请求前推进门槛；不扩大dev96，不切换正式入口。
- 48条独立IoU、126次实际请求的预调用消息/选项哈希、完整图片/轨迹和历史/原始观察前缀全部通过。0终止错误、0弃答、0失败API尝试；所有源码/6脚本保持冻结。12条独特开发描述的重复探针不能替代完整端到端或全量成绩。
- 唯一跨重复稳定改善是q03809；该例两组首工具动作在暴露干预前已不同，不能将个例改善唯一归因于行动历史。完整记录见`/Volumes/KESU/datasets/ScanRefer/runs/scanrefer_r60_controlled_parameter_history_v1/CONTROLLED_COMPARISON.json`。
- 正式命令仍为原r23入口，完整dev96参考58.33%@0.25、52.08%@0.5。Nr3D历史全量69.66%，近期输入回归不是新准确率；保留集/全量仍0预测，目标尚未达到。
- 原始RGB数值与所选frame朝向的后续检查仅零请求诊断，没有新推理包或提分版本。不能把颜色分位数当语义颜色，也不能把坐标对齐的外部相机当真实说话者相机或正确参照绑定。
