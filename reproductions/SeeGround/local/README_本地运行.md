# SeeGround 本地 API 试运行（Mac）

原工作环境已建立独立 `.venv`（不随仓库上传），无需 CUDA、PyTorch3D 或本地大模型。原论文代码保持原样，新增 `local/` 提供 CPU 点云预览和图片＋物体表到 API 的最小定位入口。

## 当前验证状态

- 已读取本地 ScanNet scene0000_00：81,369 个点，生成真实点云预览。
- 已生成合成样例：4 个带编号的方盒，分别代指两把椅子与两台电脑。用于检查软件链路，不是逼真的场景，不是 ScanNet 数据，不产生论文准确率。
- 离线请求构造检查通过。
- 离线测试覆盖渲染遮挡、候选校验、Qwen 请求与输出解析、三维框 IoU、几何证据和视觉复核流程。
- 尚未调用真实 API；需要填入配置。真实点云目前缺少物体实例标注与查找表，所以仅作无编号预览。

## 新电脑安装

```bash
cd paper/reproductions/SeeGround
python3 -m venv .venv
.venv/bin/python -m pip install -r local/requirements-lock.txt
cp .env.example .env
chmod 600 .env
```

使用 Python 3.11 或更高版本。安装后再按下面步骤配置与测试。

## 1. 填写配置

在仓库根目录的 `.env` 文件填写：

```dotenv
SEEGROUND_BASE_URL=你的服务商提供的兼容接口基础地址
SEEGROUND_MODEL=你的账号可调用的准确模型ID
SEEGROUND_API_KEY=你的密钥
```

Base URL 通常以 `/v1` 结尾，应直接按服务商提供的地址填写，不要填文档网页，也不要重复附加 `/chat/completions`。当前入口只支持 OpenAI Chat Completions 兼容接口，模型须支持 Base64 图片输入。若服务商要求额外参数，需要对应适配。

`.env` 已加入 Git 忽略规则，权限为 600。不要把密钥发到聊天中。当前程序只向配置的服务发送指定图片、查询与物体表。

CPU 流程的 `run.py` 与 `real_scene.py` 共用 `local/qwen_api.py`，使用 OpenAI Chat Completions 兼容格式。服务器上只需配置一次 `.env`，后续模块可以复用同一个客户端。

## 2. 运行离线样例

```bash
cd paper/reproductions/SeeGround
.venv/bin/python local/run.py demo
.venv/bin/python local/run.py ground \
  --image local_outputs/demo/scene.png \
  --objects local_outputs/demo/objects.json \
  --query 'Find the laptop nearest to the red chair.' \
  --dry-run
```

`--dry-run` 不联网，不计 API 费用。

## 3. 发起一次真实 API 测试

填好 `.env` 后，去掉 `--dry-run`：

```bash
.venv/bin/python local/run.py ground \
  --image local_outputs/demo/scene.png \
  --objects local_outputs/demo/objects.json \
  --query 'Find the laptop nearest to the red chair.'
```

这会产生一次 API 调用，可能收费。结果写入 `local_outputs/prediction.json`，包含模型名、预测编号、三维框、原回答及服务端提供的 token 使用量。合成样例设计的目标编号是 2，但不保证实际模型回答正确。测试文件中的模拟成功不能作为真实 API 成功的证据。

## 4. 查看真实 ScanNet 点云

```bash
.venv/bin/python local/run.py preview \
  --ply /path/to/scannet/scans/scene0000_00/scene0000_00_vh_clean_2.ply
```

输出 `local_outputs/scannet_preview.png`。目前是固定观察视角，不执行查询解析或论文视角自适应策略；无物体实例标注时不伪造编号。程序使用 CPU 针孔投影与逐像素最近深度，可见表面标号仅在传入点级实例 ID 的合成样例中演示。

## 5. 与原论文复现的边界

当前完成的是本地可运行的探索入口，不是完整基准复现：

- 使用 NumPy/Pillow CPU 渲染，未复刻作者 PyTorch3D 相机校准与光栅化参数。
- 使用简化测试提示，未运行第一次查询解析和查询自适应视角选择。
- 第三方 API 模型可能与论文 Qwen2-VL-72B 不同。
- 尚未准备真实场景的实例分割、三维物体表、Nr3D 查询及官方评测口径。

下一步：先确认 API 图像请求成功，再补齐真实 scene0000_00 的实例标注与查找表，将真实场景接入；之后再接查询解析与相机策略，最后才评测。

安装依赖的锁定清单：`local/requirements-lock.txt`。检查命令：`.venv/bin/python local/test_local.py`。

## 已跑通：真实场景两阶段试运行（2026-09-08）

最新进展覆盖了上文部分待办：已准备 `scene0011_00` 的真实点云、实例分割、坐标对齐矩阵及作者 Nr3D 查找表，并接入“API 解析查询 → 匹配目标/参照物 → CPU 选视角并标号 → API 选目标”。

- 点数：237,360；作者查找表候选数：27。
- 逐实例重建包围框与作者 GT 查找表比较，最大坐标差约 `4.53e-7 m`。
- GT 类别表仅用于核对，模型输入使用作者提供的预测类别表。没有将目标真值编号传给模型。
- 手写探索查询：`Find the large table surrounded by chairs.`
- `qwen3-vl-flash` 解析出 table / chairs，最终预测 ID 2；图中对应中央长桌，程序查出了它的包围框。
- 此查询并非 Nr3D 原始标注，不报告基准准确率。Nr3D 官方 Google Drive 描述下载及在线浏览器目前在本机超时。
- CPU 投影、类别词匹配、提示词等仍为本地适配，不能等同作者原始实验。稀疏点渲染也与论文图片质量不同。

### 本机重跑

在本目录安装依赖并配置 `.env` 后：

```bash
# 原始文件放在 local_outputs/downloads，准备时不调用 API
.venv/bin/python local/prepare_scene.py --scene scene0011_00

# 运行一次新查询会调用 API 两次，可能收费；run-name 必须使用新名称
.venv/bin/python local/real_scene.py \
  --query 'Find the large table surrounded by chairs.' \
  --run-name trial_02
```

文件来源：

- 作者查找表：[seeground_object_lookup_table.zip](https://github.com/user-attachments/files/18056532/seeground_object_lookup_table.zip)，本机保存为 `local_outputs/downloads/olt.zip`。
- ScanNet 场景目录：`http://kaldir.vc.in.tum.de/scannet/v2/scans/scene0011_00/`。按数据许可使用以下四个文件：`scene0011_00_vh_clean_2.ply`、`scene0011_00.txt`、`scene0011_00.aggregation.json`、`scene0011_00_vh_clean_2.0.010000.segs.json`。

产物目录 `local_outputs/scene0011_00/`：`objects.json`、`objects_gt.json`、`scene.npz`、`overview.png`、`provenance.json`。每次查询独立子目录内保存 `run.json`（解析、相机、标号和适配说明）、`view.png`、`prediction.json`。

这些数据、图像与 API 输出仅留本地，不上传 Git。新机器需自行准备数据和配置。


## Nr3D 官方描述：10 条真实试跑

已使用官方 Nr3D CSV 中 scene0011_00 的前 10 条 `correct_guess=True` 描述运行本地 API 流程，10 条均完成，按目标编号相等计分为 7/10。选题在推理前固定；没有把正确答案传给模型。单场景小样本、Qwen3-VL API、CPU 渲染及简化类别匹配均不同于论文完整评测，不能用 70% 与论文表格直接比较。

在 SeeGround 目录运行：

```sh
.venv/bin/python local/batch_nr3d.py
```

默认读取 `local_outputs/downloads/nr3d_official.csv` 和已准备的 `local_outputs/scene0011_00`。官方描述下载入口：https://drive.google.com/file/d/1qswKclq4BlnHSGMSgzLmUu8iqdUXD8ZC/view 。数据需自行下载，不随代码上传。下载工具若无法连接，可使用本机已配置的系统代理。

结果位于 `local_outputs/nr3d_trial_10/report.html`，包括原文、正确编号、预测编号、模型解释及每题图片；`manifest.json` 保存数据哈希和固定选题，`results.json` 保存统计。统计中包含 `acc25`、`acc50`、`mean_iou`，逐题结果还保存模型、延迟和 token 使用量。重复运行复用已完成预测，只重试失败题；更改模型或代码后，应使用 `--output local_outputs/nr3d_trial_new` 创建新实验，避免混用缓存。失败请求计入总数。

初步错误线索：第 3、6 题的正确目标 15 在作者预测类别表中为 water cooler，而 GT 类别为 trash can；第 4 题正确目标 24 未出现在当次图片的可见编号列表中。这些是可检查的影响因素，尚不能单独确定因果。GT 类别只用于事后分析，没有替换推理输入。

离线校验：`.venv/bin/python -m unittest discover -s local -p 'test*.py'`。

## 6. 服务器运行顺序

```bash
git clone -b qwen-vl-baseline git@github.com:channingLuocl/paper.git
cd paper/reproductions/SeeGround
python3 -m venv .venv
.venv/bin/python -m pip install -r local/requirements-lock.txt
cp .env.example .env
chmod 600 .env
.venv/bin/python -m unittest discover -s local -p 'test*.py'
```

先使用 `--dry-run` 检查本地链路，再填写服务器上的 `.env` 并运行真实 API。不要把 `.env`、`local_outputs/` 或 API 响应提交到 Git。

## 7. 几何证据与类别视觉复核（2026-10-05）

三个模式默认仍为 `baseline`，可用于开发集对照；不声称是 TDVR 复现或新增论文贡献。

| 模式 | 改动 | 每题 API 调用次数 |
| --- | --- | --- |
| `baseline` | 原类别词匹配和单视角流程 | 2 |
| `geometry` | 完整参照物解析、类别别名匹配、逐目标/参照物中心距离和 AABB 间隔 | 2 |
| `visual_review` | geometry 基础上逐页复核所有实例的双视角局部图，再增加 3 张上下文图 | 2 + ceil(实例数 / 9) |

类别别名包括 television/TV/screen、trashcan/trash can、fridge/refrigerator、front doors/door 等。
`screen` 与 `monitor` 被暂时当作 TV 的检索别名，不是可靠的语义等价；类别与参照物实例仍需核验。
找不到参照物时保留缺失状态，不再用目标物冒充参照物。程序提供每一对实例的距离，不自动把多个参照物合并成一个，也不推断唯一参照物。
中心距离和 AABB 间隔是不同近似量，均不是真实表面距离；左右不等于固定世界坐标轴方向。

视觉复核不读取 `objects_gt.json`，不使用固定编号修正类别。仅从预测实例点云生成按 ID 编排、隐藏预测类别标签的局部图，Qwen 返回可能的目标、参照物和无法辨认的编号。
未辨认的实例保留为候选；复核结果是模型判断，不保证召回所有误标实例。所有物体仍出现在最终文字表中，没有硬限制最终答案必须位于复核候选内。
此版本对每题全部实例复核，是固定预算对照方案，尚未实现自适应触发或动作选择。补充相机只用于看清证据，不是恢复描述者站位；遮挡、渲染稀疏和未分割的参照物仍可能导致失败。

服务器更新后分别运行，使用新的输出目录：

```bash
cd ~/paper
git switch qwen-vl-baseline
git pull --ff-only
cd reproductions/SeeGround
.venv/bin/python -m unittest discover -s local -p 'test*.py'

.venv/bin/python local/batch_nr3d.py --scene scene0011_00 --count 10 \
  --mode geometry --output local_outputs/nr3d_qwen_geometry_01

.venv/bin/python local/batch_nr3d.py --scene scene0011_00 --count 10 \
  --mode visual_review --output local_outputs/nr3d_qwen_visual_review_01
```

27 个实例时视觉复核模式每题共 5 次 API 调用，成本高于单视角模式。此轮不自动发起收费实验。
10 条单场景样本用于开发调试，不能作为最终泛化结果；后续固定其他场景/样本进行验证。

`run.json` 保存每次调用的原回答、token 使用量、`finish_reason`、解析、几何证据、视图可见编号与未标出的目标编号。
`ground_response.json` 在解析最终编号之前保存回答，便于诊断无效输出。
`prediction.json` 保存 `pipeline_latency_s`（包括解析、复核、渲染及最终请求）和各次调用 token 用量；`latency_s` 只表示最终选择请求耗时。
`finish_reason=length` 说明回答截断；若仍包含合法编号，保留预测并记录截断状态，不把它误当作完整解释。

批量实验 manifest 绑定模式、模型、接口地址、代码与输入数据哈希；配置变更会拒绝复用原目录（包括旧版本 manifest），不会记录 API key。
准确率和 IoU 阈值准确率将失败样本计入分母；只有失败样本而没有任何有效 IoU 时输出 null，避免声称完成了评测。
Nr3D 此处使用 GT 实例几何和预测类别，主要指标是编号准确率；此处 IoU 辅助统计不等于 ScanRefer 检测框基准成绩。

## 8. 第二轮：距离排序与独立类别检查

第一轮在 scene0011_00 的 10 条开发样本中，geometry 为 6/10，全实例 visual_review 为 3/10。
这是用户服务器日志的单次结果，不是完整基准或统计显著性结论。全实例复核混淆目标/参照角色并影响相机，旧 `visual_review` 保留供审计，不建议继续用于新实验。

新模式保持单张上下文图：

| 模式 | 行为 | 每题调用次数 |
| --- | --- | --- |
| `ranked_geometry` | 几何模式基础上验证原句的距离极值词，并由程序按每个参照物分别排序 | 2 |
| `category_review` | ranked_geometry 加逐实例、单类别 yes/no/uncertain 视觉检查 | 2 + 实际复核实例数 |

距离排序只支持原句明确的 nearest/closest/farthest/furthest，解析中的 phrase 必须是原句子串并含对应极值词。
没有支持文本的 nearest/farthest 会变为 other，同时保留原 API 回答和拒绝原因。否定或排除条件先不执行排序。
by/beside/next to 不升级成 nearest。排名保持每个参照物独立，保留距离相等的多个最优编号，不合并参照物、不硬选答案。
这些检查仅验证极值词证据，不保证解析的目标和参照物绑定正确。距离仍为中心距离，类别误标依然会影响排序。
`large black screen` 等表达在检索时去掉常见属性词，原句的外观条件仍提供给最终模型。
左右尚未转换成描述者坐标系，明确标记 unresolved；不能把新的固定相机当作说话者朝向。

类别复核默认 `--review-limit 6`：优先检查预测类别匹配的实例，再按与已匹配参照物的最小中心距离补充其他实例。
没有匹配参照物时按物体表顺序补充；这只是候选召回启发式，不保证找到远处的误标目标，亦非经过验证的冲突触发策略。
实例提议不使用 GT 类别、GT 目标编号或固定编号。只复核单一目标类别；多目标类别或未解析类别时跳过。
每次调用仅含一个实例的双视角图及类别问题，隐藏预测标签和原查询，不允许输出 TargetIDs/AnchorIDs。
Qwen 的 yes/no/uncertain 只做软证据；它不改变目标、参照物集合、相机、可见性标记，不硬剔除任何实例。
额外距离证据仅覆盖被复核实例；未检查的实例不视为否定。调用失败、错误编号、截断或非法格式都转为 uncertain 并记录原因。

服务器拉取代码并跑测试后，先运行排名对照，再运行类别对照；不会自动发起收费调用：

```bash
.venv/bin/python local/batch_nr3d.py --scene scene0011_00 --count 10 \
  --mode ranked_geometry --output local_outputs/nr3d_qwen_ranked_01

.venv/bin/python local/batch_nr3d.py --scene scene0011_00 --count 10 \
  --mode category_review --review-limit 6 \
  --output local_outputs/nr3d_qwen_category_01
```

类别模式每题最多 8 次 API 调用，可能比旧 visual_review 更贵；这里检验独立类别问题的可靠性，不宣称已降低成本。
`run.json` 记录候选提议策略、每个检查的 verdict/reason、原 API 回答、拒绝的关系和每个参照物下的排名。
manifest 绑定 review-limit，防止预算不同的实验复用同一缓存。旧实验目录不要覆盖。

## 9. 诊断驱动模式 diagnostic

新增流程：完整条件解析 → 共享观察者/参照物假设 → 程序化约束诊断 → 按缺失证据选择 RGB 复核 → 重算假设与稳定候选 → 最终预测。
这是借鉴论文思路的实验实现，不是任一论文的完整复现，也不预先声称提高准确率。

```bash
cd ~/paper
git switch qwen-vl-baseline
git pull --ff-only
cd reproductions/SeeGround
.venv/bin/python -m unittest discover -s local -p 'test*.py'
.venv/bin/python local/batch_nr3d.py --scene scene0011_00 --count 10 \
  --mode diagnostic --review-limit 6 \
  --output local_outputs/nr3d_qwen_diagnostic_01
```

观察者与检查相机分开。仅在描述明确提供 origin 和 facing 时计算左右/前后：水平 forward 从站位指向朝向参照物，right 为 forward × Z-up。
明确“房间中间”时使用点云 AABB 中心近似；仅“面对柜子”没有默认站位。“左窗户”不自动解释成世界轴负方向，也不自动解释成最左窗户。
多个站位/朝向实例保留为不同共享假设；同一假设的多个条件取 AND，不把不同假设当作投票。
同类别参照绑定在同一假设内共享，不为每个目标单独挑有利朝向。256 个组合以上停止稳定性结论并记录预算不足。

`nearest/farthest` 使用中心距离排序并保留并列；`near` 只记录距离，当前没有经验证的阈值，不升级为 nearest。
corner、地面支撑、序数、左右极值、否定以及未支持的条件均记录为缺失，不在证据不足时声称已解决。
解析器仍可能错误绑定或遗漏语义；引文校验和常见遗漏检查不能替代正确的语言理解。

RGB 为可选输入，目前没有自动下载 ScanNet `.sens`、恢复相机位姿或投影实例。
没有 `rgb_crops.json` 时仍可运行，通常每题 2 次 API（解析、最终选择）；新模式不会用点云裁剪替代真实 RGB 复核。
有裁剪清单时每题最多额外 `review-limit` 次复核，每次最多使用该实例的前两张真实图片。
裁剪应由相机位姿/深度与实例投影绑定，包含可辨认目标及少量上下文，不能使用 GT 类别或 Nr3D 目标答案挑图。
在 `local_outputs/scene0011_00/rgb_crops.json` 放入以下格式，图片路径相对于场景目录：

```json
{
  "scene": "scene0011_00",
  "crops": [
    {
      "id": 15,
      "image": "rgb/instance_15_frame_0100.jpg",
      "source_frame": "scene0011_00/color/0100.jpg",
      "binding": "projected_instance"
    }
  ]
}
```

上例编号仅说明格式，不是指定应修正的对象。导入会检查场景、已知 ID、图片与路径；`projected_instance` 是上游提供的来源声明，程序不能独立证明投影正确。
清单与图片哈希进入 manifest，裁剪更改必须另开实验目录。RGB 复核只判断实例类别及该类别的外观条件，yes/no/uncertain 都是模型证据；uncertain 不等于 no。
复核针对待检验候选/参照物，不按未知裁剪重复调用。没有相应 RGB 或预算不足时保留 missing。
候选提议是预测类别实例加按参照距离补充的有限实例，不能保证召回所有误标物体；resolved 只在这个候选范围内成立。

输出字段：

| 字段 | 含义 |
| --- | --- |
| `diagnosis.hypotheses` | 各共享观察者/参照物解释、约束测量及可行候选 |
| `diagnosis.stable_candidate_ids` | 所有假设可行集合的交集，不是投票结果 |
| `diagnosis.status` | resolved：范围内所有条件受支持且唯一；missing：缺少条件/证据；ambiguous：多个完整候选/解释；conflict：受支持约束无法同时满足或类别被否定 |
| `diagnosis.actions/checks/missing` | 触发原因、RGB 原判断、仍缺失的证据 |
| `decision_status/accepted` | 最终模型选择与诊断一致且未截断的 resolved 才 accepted；不一致为 conflict |
| `diagnosis_policy` | 最终提示采用的策略，保存在 run.json/prediction.json：resolved 使用 supported_unique_candidate；其他状态使用 audit_only |

`audit_only` 明确要求最终模型从完整物体表、原句和图像重新判断，不得用诊断候选集合排除对象、优先选择或打破平局。
已测量事实仍可辅助核对明确条件，但部分条件满足不能证明答案。resolved 的唯一候选只作为范围内强支持，原句/图像冲突仍允许模型提出异议。
这是提示策略调整，不会强行替换预测编号，也不保证消除模型波动或提升准确率；需使用新输出目录验证。

所有状态仍输出最佳猜测 `predicted_id`，标准 accuracy/IoU 的分母不变。
批量报告额外统计状态数量、accepted_count 和 accepted_accuracy；后者只能结合覆盖率阅读，不能拿筛选后的正确率替代总体准确率。
当前 10 条开发样本只用于检验诊断是否有效；后续需要固定独立场景做泛化验证。

### 首轮服务器反馈与修复

首轮 diagnostic 为 5/10，10 条均 missing、accepted_count=0，平均流水线耗时约 6.13 秒。
这没有证明提高准确率。服务器没有提供真实 RGB，missing 包含类别证据不足；部分方向条件还因解析缺少 scope 未能执行。
原查询之后追加的提示例句被解析器混入 Unhandled，例如电视旁垃圾桶描述被加入朝向冰箱/角落条件。
已将 diagnostic 解析改为独立 system schema 和 user JSON 查询，移除具象例句，明确每条关系的 scope，并校验 Unhandled 必须来自原句。
这些校验仍不保证实际模型正确解析；未解析的条件保留缺失。
扩展候选现在仅用于诊断/RGB 复核，不再改变单张检查图的相机和标号。给定相同解析，新模式的检查图与 ranked_geometry 一致。
group-facing 的复数类别引文检查也已修正。更新后的流程尚未跑收费 API，不能据离线测试声称恢复 6/10 或更高成绩。
重跑请使用新目录 `--output local_outputs/nr3d_qwen_diagnostic_02`；旧结果保留供比较。

第二轮服务器运行 10 条全部未完成；错误为解析 JSON 格式不正确。
第一条原回答含 Observer/Attributes/Unhandled/Relations，却缺少 Target/Anchor，未进入渲染和最终定位。
已补齐解析提示的完整顶层 JSON 模板，明确后续字段说明不能省略 Target/Anchor；未通过校验的输出仍拒绝进入定位，不猜补类别。
批处理现在在终端打印 ERROR（失败阶段、异常类型、具体原因），run.json/results.json 保存失败阶段和流水线耗时。
离线测试不验证模型遵守模板：服务器应先用 `--count 1 --output local_outputs/nr3d_qwen_diagnostic_smoke_03` 检查，再在新目录运行整批。

## 10. 人工类别与预测类别对照

单条/批量入口均新增 `--object-table pred|gt`，默认 pred。
pred 读取 `objects.json`，gt 读取 `objects_gt.json` 的完整场景表（编号、类别、框）；两者都不把本条查询的 target_id 或 stimulus_id 传给模型。
GT 场景语义类别可作为 oracle 条件输入，不能与使用预测类别的论文结果直接比较。
此前文档中“GT 类别仅用于评测”的说明描述旧版/default pred，现在 gt 明确允许场景类别作为输入。
推理与最终模型提示都读取所选表，run.json/prediction.json/manifest/HTML 报告明确记录来源；两种表不允许复用同一缓存。

diagnostic 的 gt 条件把标注类别当作已知，只检索该类别，不再为纠正类别误标补充邻近实例；pred 保持候选扩展及类别复核。
人工类别不包含外观判断和参照实例答案：颜色/尺寸等属性、near、缺少站位和未支持条件仍可能导致 missing。
两组按同样场景、同样 CSV 描述、同样模型与代码运行；解析和最终 API 每次重新调用，模型随机性仍可能影响差异。
这组检验 oracle 场景语义输入的作用，包含检索、相机和诊断的下游变化，不是固定所有中间证据的单因素消融。

在服务器拉取并测试后，先跑人工类别，再跑本版预测类别作对照（各题默认 2 次调用，若已有 RGB 可能额外复核）：

```bash
cd ~/paper
git pull --ff-only
cd reproductions/SeeGround
.venv/bin/python -m unittest discover -s local -p 'test*.py'

.venv/bin/python local/batch_nr3d.py --scene scene0011_00 --count 10 \
  --mode diagnostic --object-table gt --review-limit 6 \
  --output local_outputs/nr3d_qwen_diagnostic_gt_01

.venv/bin/python local/batch_nr3d.py --scene scene0011_00 --count 10 \
  --mode diagnostic --object-table pred --review-limit 6 \
  --output local_outputs/nr3d_qwen_diagnostic_pred_04
```

本机仅运行模拟 API 测试；真实准确率须以上述服务器运行结果为准。GT 类别带来的提升不属于新增推理算法贡献。

## 11. 固定多场景 baseline / diagnostic 对照

已有 scene0011_00 的 10 条开发样本：GT baseline 6/10，GT diagnostic 8/10。
这只是单次小样本结果，不能据此宣称稳定提升。接下来固定推理代码，用未调试场景检验效果。

新增 `local/paired_nr3d.py`，复用原批处理的推理、评分和缓存逻辑。
默认排除 scene0011_00，在官方 CSV 的 correct_guess=True、dataset=nr3d 行中抽样。
只选择作者提供的 GT OLT ZIP 中存在、有效描述不少于请求题数的场景。
固定 seed=2026，随机抽取 5 个场景、每场景 20 题，保存精确 CSV 行号、原描述和评测答案。
这是未调试场景的探索验证，不是按官方 train/test split 组织的完整测试集成绩。
清单中的 target_id/stimulus_id 仅用于对齐与评分，推理只收到原描述与所选场景表。

先生成清单；此命令不下载数据、不调用 API：

```bash
cd ~/paper
git pull --ff-only
cd reproductions/SeeGround
.venv/bin/python -m unittest discover -s local -p 'test*.py'

.venv/bin/python local/paired_nr3d.py plan \
  --scene-count 5 --per-scene 20 --seed 2026 \
  --plan local_outputs/nr3d_heldout_plan_01.json
```

程序列出抽中的场景、有效描述数和缺少的准备文件。不会自动下载或准备场景。
每个新场景仍需在 local_outputs/downloads 中备齐此前用过的四种原始文件：
`<scene>_vh_clean_2.ply`、`<scene>.txt`、`<scene>.aggregation.json`、`<scene>_vh_clean_2.0.010000.segs.json`。
继续复用已有 olt.zip；原始文件齐全后按清单提示执行 `local/prepare_scene.py --scene <scene>`。
本阶段不需要 `.sens`、RGB 视频或 GPU 渲染依赖。
可用 `--scenes sceneXXXX_00 sceneYYYY_00` 明确指定未调试场景；仍受排除列表、OLT 和题数检查约束。
清单已存在且内容不同会拒绝覆盖；更换抽样参数时另取文件名。

全部场景准备好之后，再执行收费 API 实验：

```bash
.venv/bin/python local/paired_nr3d.py run \
  --plan local_outputs/nr3d_heldout_plan_01.json \
  --object-table gt --review-limit 6 \
  --output local_outputs/nr3d_paired_gt_01
```

运行前核对 CSV 哈希、逐行内容、全部场景文件、GT 目标对齐及物体表格式。
缺少任一场景时整体停止，尚未调用 API；不自动跳过缺失场景来改变固定题目集合。
两种模式使用完全相同的场景、题目、物体表和模型配置；各自重新解析/调用模型，
所以这是整体流程对照，不是固定中间解析的几何模块消融。API 随机性仍影响结果。
100 题 × 2 种模式通常共 400 次调用；如已有 RGB 清单，diagnostic 可能额外复核。
原命令重跑时复用成功预测、重试失败项；不重复计入同一题。代码、模型、数据或预算变化时须新建输出目录。

输出目录包含：

- `comparison.json`：两组总体指标、每场景指标、改善/退步数量、诊断状态、已完成成对题数。
- `comparison.csv`：逐题原描述、正确编号、双方预测、变化类型与请求错误。
- `<scene>/baseline/`、`<scene>/diagnostic/`：原有逐题 run.json、图像、预测和 HTML 报告。

两组失败请求均计入各自准确率分母；改善/退步中可能包含请求失败，应结合 error 列和 completed_pairs 阅读。
每完成一个场景更新比较报告，planned_total 标明固定清单总题数；中途报告不能当作全部场景的成绩。
逐题变化有 improved、regressed、both_correct、both_wrong 四种，不能只展示改善的例子。
本地测试仅使用模拟 API；实际新场景准确率尚未测得。

### 距离补全修正与第二轮结果

同一 100 题第二轮 GT baseline 60/100，diagnostic 64/100，改善 16 题、退步 12 题；
第一轮分别为 59/100、67/100。第二轮未证实距离补全有收益，两轮都重新调用 API，差异不能全部归因于代码。
两轮诊断状态数量都为 resolved=8、missing=77、ambiguous=1、conflict=14；数量相同不保证对应题目相同。

已撤回将 `of the two ... further from` 转成整个类别最远实例的规则。
这类描述比较的是特定子集，当前没有该子集绑定能力，继续作为未支持条件。
关系补全只接受整句为简单的“目标类别 + closest/nearest/farthest/furthest + 参照类别”结构，
且解析已有唯一目标和参照类别；嵌套参照、额外方向、否定、比较级和不匹配类别都不猜补。
恢复的关系记录在 `parsed_query.recovered_distance_relations`，原始 API 回答仍保存在 calls 中。
补全随后接受正常的距离校验；参照实例不存在、方向观察坐标不完整时仍记录缺失。
候选集合并不等同于 baseline：两种模式的类别匹配器和解析提示不同。
诊断候选只是证据，最终模型仍读取完整物体表；8 个 accepted 正确不能替代全部 100 题准确率。
此前排查脚本读取 run.json 顶层 parsed 导致空列表；正确字段为顶层 `parsed_query`。
