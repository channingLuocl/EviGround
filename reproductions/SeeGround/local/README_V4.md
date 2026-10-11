# Relative-size v4

v3 基线：`ba1dd6e`（分支 `codex/relative-size-evidence-v3`）。v4 在独立工作树和分支 `codex/relative-size-evidence-v4` 实现。环境复用已安装的 v3 `.venv`；不修改 v3 源码、环境包或历史输出。

## 行为

- 原文证据清单包含对象角色、同指、比较成员数量、限定附着与依赖；外层 ReAct 保持 text_only，自行选择五维操作顺序。JEV 和直接 chat 两个入口均收到清单及结构化视觉观察。
- 显式关系子集使用逐成员的同一关系证据绑定，未知成员不被当作否。绑定完成后冻结；成员改变时旧尺寸证据失效，并能触发一次共享回溯。普通后置修饰仍可保留附着歧义，独立 `and above ...` 与已验证的 `in the room` 假设不限制比较集合。数量检查不使用 GT stimulus 候选列表。
- 尺寸同时保存原框、点云定向指标、原始/稳健三轴边界及来源；框和同实例点云不视为独立认证。水平指标不会因高度未知同时失效。完整性另列：稳定水平薄面仅提示 surface-only 风险，保留“真薄物体/局部扫描”两种解释；可见侧面和底部才允许视觉完整性有支持，缺失几何不会由换视角补齐。
- 体积和占地排序冲突不再永久否决取证：保留两种排序，允许结构解释与暂定选择，不据替代指标硬删候选。桌子/桌面的水平包围面积在体积不可用时可支持暂定答案；若可靠体积与面积同序，继续用体积。桌高改为广覆盖水平上表面相对局部 **已标记 floor 实例** 的高度，地面/平面证据缺失保持未知。书架同时 longer/shorter 分别为水平长轴/竖直跨度；主体高度另列诊断，不把分位数跨度冒充主体分割。
- size 内部按需渲染统一蓝色、唯一 ID 的候选框，同一相机与尺度，附尺寸、单位、边界质量与完整性。指标分歧、测量质量/薄面风险自动构成 size 任务的内部检查理由；模型也可用 `inspect=true`、`focus_ids`、`azimuth_deg`、`elevation_deg` 选择有针对性的视图。结构化观察返回外层，图片不进入外层 text_only 输入。无 GT 答案颜色或其他目标描述。
- near、明确目标在参照物上的支撑、against/attached/hanging，以及房间角落可执行几何＋内部视觉取证。near 是近距离支持区/未知区/明显远离区，不能替换成 nearest；支撑须有上下表面接近和水平重合；角落须有两面接近、相交、正交的已供给墙实例。AABB 自身不能证明接触，缺参照/可见性保留未知；桌角、反向“with a plant on top”等不冒充目标支撑/房间角落。
- 初始和后续删除均记录引用、证据、绑定和删除前状态。候选为空、已完成绑定发生具体变化、最终复核有依据的矛盾可共享一次回溯。恢复相关删除候选，撤销相关过滤/尺寸依赖并重开 ReAct；不因模型更喜欢另一 ID 改答案。撤销的初始/尺寸谓词须补新证据，不能重复旧计算自动认证。
- 仅有 predicted_id 时最终复核所选对象、主要被删/存活替代与最多两个绑定参照。回溯后重新选择也复核，无第二次回溯。保留 singleton_fallback；`size_proxy_tentative`、`dynamic_threshold_tentative` 均不改变 accepted。复核 consistent 不把原有 missing 升级为完整认证。

## 固定资源边界与消融

默认启用 v4，动态阈值默认关闭。环境变量值均为 `0` 或 `1`，并进入批次 manifest。输出目录记录源码 hash，不可复用不同设置的实验目录。

| 变量 | 用途 |
| --- | --- |
| `SEEGROUND_V4=0` | v3 行为兼容回归 |
| `SEEGROUND_V4_INVENTORY=0` | 不提供证据清单 |
| `SEEGROUND_V4_BACKTRACK=0` | 关闭一次回溯及其预留 |
| `SEEGROUND_V4_FINAL_REVIEW=0` | 关闭最终复核 |
| `SEEGROUND_V4_SIZE=0` | 保留 v3 尺寸规则与测量质量 |
| `SEEGROUND_V4_RELATIONS=0` | 关闭新增关系取证 |
| `SEEGROUND_V4_DYNAMIC_THRESHOLD=1` | 单独运行有限阈值实验 |

`--review-limit 3` 为正常证据操作上限；发生一次有效回溯后，总上限为 **4**，不是无限补预算。最多 `max(2×limit+3, limit+3)+3` 次外层尝试；关闭回溯不加 3。每例内部 size 视觉调用最多 2，新增关系视觉调用最多 2，最终复核最多 2，每调用 1 个共同视图；原外观分组调用仍受原有外层预算约束。内部调用、渲染和 API 延迟记入轨迹与汇总。

动态放宽只在模型停止、预算允许、已有绑定/可靠测量且没有指标歧义时尝试一次：相对差距下限 5%，绝对下限保留体积 0.0005 m³、面积 0.005 m²、长度 0.03 m。高度/桌面高度排除在本轮放宽之外，确保 29522 不会由总跨度错误排序得到确定答案。重复调用本身不是放宽理由。

## 离线验证

旧 603 项测试验证 v3 行为；新增 v4 测试在自身 setUp 中显式打开 v4，覆盖绑定、质量、薄面风险、视图、内部调用、回溯和再次复核。旧测试中固定的模型回复不包含 v4 新增视觉回复，且原尺寸断言属于 v3 规则，因此兼容套件须使用明确的 v3 消融设置：

```sh
cd /Volumes/KESU/论文/.worktrees/relative-size-evidence-v4/reproductions/SeeGround
PYTHONDONTWRITEBYTECODE=1 SEEGROUND_V4=0 PYTHONPATH=local .venv/bin/python -m unittest discover -s local -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=local .venv/bin/python -m unittest test_v4 -v
```

保存解析结果的几何回放、五个案例共同视图（无需新 API 调用）：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=local .venv/bin/python local/validate_v4_offline.py \
  --saved-root /Volumes/KESU/论文/.worktrees/relative-size-evidence-v3/reproductions/SeeGround/local_outputs/relative_size_20261011_201732_v3_spacy_full300 \
  --scene-root /Volumes/KESU/论文/.worktrees/relative-size-evidence-v2/reproductions/SeeGround/local_outputs \
  --output local_outputs/v4_offline_validation
```

检查集：五个尺寸案例、七个错误兜底、五个语法案例，另选既有 v3 样本中未列入这 17 条的 12 条。它们不构成新数据集的独立验证。`offline_report.json` 保存全部条件、测量、v3/v4 保留候选和评估字段；评估目标在几何计算/渲染完成后才附加。

已验证：636 项通过（603 项兼容＋33 项 v4）；29 条离线回放零网络调用，目标保留 22→27。五个语法案例继续保留各自正确候选；10431 暂定尺寸代理指向 10；30825 暂定面积代理指向 17；11862 的 19 号同时为稳定边界和 surface-only 风险，不能认证最小完整体积；29522 缺已标记局部地面，保持未知；37559 保留主体/测量缺口。**这些是候选保留和取证行为，不能作为新 ReAct/VLM 准确率或已纠正兜底数量。**

## 新推理命令

先加载原有 API 环境配置，不在文档保存密钥。保持 `SEEGROUND_SYNTAX_BACKEND=spacy`，将 v4 候选表设置明确记录为 GT。

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=local .venv/bin/python local/run_module_acceptance.py \
  --manifest local/validation/v4_regression_17.json \
  --scene-root /Volumes/KESU/论文/.worktrees/relative-size-evidence-v2/reproductions/SeeGround/local_outputs \
  --output local_outputs/v4_regression_17 --review-limit 3 --workers 10

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=local .venv/bin/python local/run_module_acceptance.py \
  --manifest local/validation/v4_additional_12.json \
  --scene-root /Volumes/KESU/论文/.worktrees/relative-size-evidence-v2/reproductions/SeeGround/local_outputs \
  --output local_outputs/v4_additional_12 --review-limit 3 --workers 10

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=local .venv/bin/python local/run_size_validation.py \
  --manifest local/validation/nr3d_size_300.json \
  --scene-root /Volumes/KESU/论文/.worktrees/relative-size-evidence-v2/reproductions/SeeGround/local_outputs \
  --output local_outputs/v4_full300 --review-limit 3 --workers 10
```

上述输出目录每组消融必须换新名称。完整对比须同时看整体/已回答准确率、null、accepted、复核触发、回溯数量、复核错改对/对改错、额外 API 调用及延迟。源码实现期间未发起付费的新 300 条推理。

此工作树 `local_outputs/downloads/nr3d_official.csv` 已链接复用 v3 官方 CSV 文件；manifests 保留官方 CSV hash。后续 ScanRefer 仅复用实例 ID、坐标系、框来源及取证接口；本轮没有 ScanRefer 完整实验、预测框/GT 框混合报告或任务训练。
