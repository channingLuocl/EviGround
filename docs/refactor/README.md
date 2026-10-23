# 目录重构验证（2026-10-23）

本次将冻结的 Qwen r18 主线整理为标准 Python `src/` 布局，并集中放置文档、测试、数据清单与第三方代码。原始完整单轮基线为 **5,437 / 7,805 = 69.66%**。

## 迁移范围

- 原始 152 个模块的对应路径见 [`../module_layout.json`](../module_layout.json)。
- 包内导入改为 `seeground.*`；资源路径集中在 `src/seeground/paths.py`。
- 主任务通过包模块启动子进程；源码指纹使用包内相对路径，避免不同目录的同名文件互相覆盖。
- 默认入口保留原 r18 运行策略：预测 Top-5、Qwen、思考模式、本地/API 并发各 8；结果续跑仍保留所有已完成结果，不累计历史正确样本。
- 冻结目录保持原始内容。旧验证路径仅保留相对符号链接，供原冻结校验器复核。
- 重构前实验工作区代码保留在 `experiments/legacy_local/`；历史测试保留在 `tests/legacy/`。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| 默认离线测试 | 287 项，0 失败，0 错误，1 项按原条件跳过 |
| 核心算法静态比较 | 核心模块的 AST 与冻结源码一致，包括提示词和常量；仅排除资源/源码指纹函数 |
| 全部原始模块比较 | 751 个顶层定义一致；另外 51 个定义的差异为资源路径、入口、源码哈希或回放模拟路径 |
| Wheel 构建、临时环境安装 | 通过；未修改原运行环境 |
| 仓库外执行安装后的命令 | `seeground --help` 与 worker 模块入口通过 |
| 安装后真实场景准备 | `--limit 1 --prepare-only` 通过，没有调用模型 |
| 原冻结快照复核 | 7,805 条、5,437 条正确、152 份原始源码、140 份类别输入全部通过 |

测试结果见 [`unit_tests.json`](unit_tests.json)，源码比较索引见 [`algorithm_equivalence.json`](algorithm_equivalence.json)。

迁移前测试状态见 [`baseline_tests.json`](baseline_tests.json)：35 组历史测试中，23 组适用于冻结主线并通过；另外 12 组针对尚未采纳的实验功能，迁移前已有失败或导入错误。默认测试集保留前 23 组，并增加 5 项目录迁移检查；本次没有修改算法来适配历史实验测试。

本次没有重新执行付费全量推理；69.66% 是原冻结结果的独立复核，不代表重新运行大模型后保证得到相同结果。

## 复核命令

安装项目及依赖后，在仓库根目录运行：

```bash
PYTHON_DOTENV_DISABLED=1 python -m unittest discover -s tests/unit -t .
bash scripts/verify_results.sh
```

普通安装的 wheel 从仓库外使用时，设置 `SEEGROUND_ROOT=/path/to/repository` 以定位查询、类别输入和第三方资源。README 中的 `pip install -r requirements.txt` 使用可编辑安装，可自动定位仓库。
