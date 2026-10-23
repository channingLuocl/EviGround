# EviGround

**ReAct-Based Visual–Geometric Reasoning for 3D Visual Grounding**

EviGround 根据语言描述定位 3D 场景中的物体。输入是点云、实例几何和外部分类器预测；大模型自主选择 `inspect`（观察）、`measure`（测量）和 `finish`（作出选择）。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

在 `.env` 填写自己的模型 API 密钥、模型名称和接口地址。默认使用 OpenAI 兼容协议。

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

## 测试

```bash
PYTHON_DOTENV_DISABLED=1 PYTHONPATH=src python -m unittest discover -s tests/unit
```

清理后的版本需要独立运行以报告新指标；既往实验数字不是本版本的重新评测结果。
