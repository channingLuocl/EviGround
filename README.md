# EviGround

**ReAct-Based Visual–Geometric Reasoning for 3D Visual Grounding**

EviGround 根据语言描述定位 3D 场景中的物体。输入是点云、实例几何和外部分类器预测；大模型自主选择 `inspect`（观察）、`measure`（测量）和 `finish`（作出选择）。

## 安装

需要 Python >= 3.10。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

在 `.env` 填写自己的模型 API 密钥、模型名称和接口地址。默认使用 OpenAI 兼容协议。

ScanRefer 原始点云预处理及完整离线测试还需要 `pip install -e '.[scanrefer]'`。Mask3D 和 r8 分类器的 GPU 环境、权重及冻结外部源码需另行准备，不包含在该可选依赖中。

## 运行

```bash
eviground \
  --predictions /absolute/path/to/classifier_predictions.zip \
  --scene-root /absolute/path/to/prepared_scenes \
  --category-top-k 5 \
  --workers 8 --api-concurrency 8 \
  --output outputs/top5_single_pass
```

- `--predictions` 必须显式指定，支持已校验的分类导出 ZIP 或目录。
- `--category-top-k 1` 只使用第一类别；`5` 保留五个候选类别，保留排名和分差，结合视觉与空间证据判断。
- 默认评估 `data/queries/nr3d_val.json` 中的 7805 条描述。用 `--limit 10 --prepare-only` 检查输入，不调用模型。
- 同一输出目录、相同代码和配置可继续未完成任务；已完成的正确、错误和 null 都原样保留。新实验必须使用新输出目录。
- 单轮结果保存在 `results.json`；错误和 null 计入总分母。没有跨运行正确结果合并、累计命中或 GT 类别推理入口。

## 目录

```text
src/eviground/   当前方法、类别适配、渲染与评估
data/queries/   固定评估描述；答案只供评估进程使用
docs/           使用说明与方法说明
tests/unit/     当前流程的离线回归测试
scripts/        命令启动脚本
```

详细说明见 [运行指南](docs/usage.md) 和 [方法结构](docs/architecture.md)。外部模型、分类器训练代码和旧实验结果不随本仓库分发。依赖及来源说明见 [NOTICE](NOTICE)；许可证见 [LICENSE](LICENSE)。

## ScanRefer 与消融实验

- ScanRefer 清单转换与评分：`scripts/scanrefer.py`；主推理仍使用 `eviground --manifest ...`，预测实例按目标 GT 框的 3D IoU 评分，报告 Acc@0.25 / Acc@0.5。
- ScanRefer 原始输入、预测实例、分类导出、策略与审计脚本位于 `scripts/prepare_scanrefer_*.py`、`scripts/classify_scanrefer_r8.py` 和 `scripts/audit_scanrefer.py`。
- Nr3D 子集抽样、E0–E5 批处理与审计汇总见 [消融实验指南](docs/ablation.md)。当前抽样器和六条件批处理器仅支持 Nr3D；ScanRefer 子集消融尚未由这些脚本实现或启动。
- 可选动态 DAG Monitor 的开关、依赖回退、独立计划修正预算及同子集 E0 对比命令见该指南的“动态 DAG Monitor 与原 ReAct 对比”；默认推理仍使用原 ReAct。
- `scanrefer_ready.sh`、`scanrefer_gt_ready.sh`、`run_scanrefer_r18.sh` 和 `run_eviground_local.sh` 保留历史本机路径及冻结源码入口，不能视为新机器上的通用启动命令。新实验应显式传入清单、预测导出、场景和独立输出目录。

`experiments/`、`output/`、`outputs/`、`tmp/`、`local_outputs/` 与 `docs/figures/` 保存本地实验快照、模型输出或论文素材，不纳入当前源码提交；已有文件保留在本机。

## 测试

```bash
PYTHON_DOTENV_DISABLED=1 PYTHONPATH=src python -m unittest discover -s tests/unit
```

清理后的版本需要独立运行以报告新指标；既往实验数字不是本版本的重新评测结果。
