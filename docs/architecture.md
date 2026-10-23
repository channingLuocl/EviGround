# EviGround 代码布局

方法名称为 **EviGround**，论文暂定标题为 *EviGround: ReAct-Based Visual–Geometric Reasoning for 3D Visual Grounding*。Python 包、安装名称和命令入口统一为 `eviground`，配置使用 `EVIGROUND_*`；旧配置通过 `config.py` 兼容读取。

默认运行入口使用 2026-10-23 冻结的 Qwen r18 代码。目录迁移只调整模块导入、资源路径、子进程入口和源码指纹，不修改候选筛选、提示词、几何计算或模型决策。

| 目录 | 职责 |
| --- | --- |
| `src/eviground/api/` | API 请求、并发控制和响应结构 |
| `src/eviground/grounding/` | 候选推理、主动观察和 r18 决策 |
| `src/eviground/query/` | 查询解析、语义图和改写 |
| `src/eviground/geometry/` | 位置、距离、尺寸和空间关系 |
| `src/eviground/rendering/` | 场景绘制、视角和视觉证据 |
| `src/eviground/categories/` | 预测类别、Top-1/Top-5 输入及匹配 |
| `src/eviground/evaluation/` | 评估、进度、结果存储和续跑 |
| `src/eviground/classifiers/` | 分类器及其特征模块 |
| `src/eviground/cli/` | 命令行入口 |
| `src/eviground/experiments/` | 分类训练、诊断、缓存与历史实验工具 |
| `tests/unit/` | 默认离线回归测试 |
| `data/` | 场景标签常量和官方查询清单 |
| `third_party/` | 原始 SeeGround 和第三方模型代码 |
| `docs/` | 当前指南、来源和历史文档 |
| `outputs/` | 已冻结输入、结果与审计；生成的新结果单独忽略 |

`outputs/r18_qwen_original_resume_20261022/runtime_original/` 保留原始源码，作为冻结审计的依据。新代码从 `src/eviground/` 导入，不使用冻结目录作为 Python 搜索路径。

`docs/module_layout.json` 列出原始 152 个模块与新包路径的对应关系。旧的开发代码及测试以历史实验保留，避免将后来尚未采纳的实验改动混入本次 69.66% 主线。

冻结结果的源码哈希记录原始布局。新实验在新输出目录生成新的源码指纹，不能把迁移后的代码写入旧冻结批次。
