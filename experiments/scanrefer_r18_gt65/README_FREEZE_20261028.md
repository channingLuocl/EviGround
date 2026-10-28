# ScanRefer 当前代码冻结：20261028-dev

本目录保存 2026-10-28 正在手动运行的 ScanRefer 实验代码。327 个文件与源文件逐字节相同，总计 3,516,689 字节；入口与 visual109_runtime 三个运行模块已逐项核对当前 protocol.json 中的 SHA-256。逐文件清单见 freeze_20261028/CODE_MANIFEST.json。

源目录：/Volumes/KESU/experiments/scanrefer_r18_gt65

源实验 Git 提交：f946b4b（当前运行新增脚本此前尚未提交，现已一并收录）。本分支基于 paper 仓库 master 802e50fd47f6a838f765f5f1e98401e61accbc15，新增独立 experiments/scanrefer_r18_gt65 目录，未替换仓库原有 src。

冻结的是代码与配置依据，不是数据包或可移植容器。未上传 API 密钥、私有环境文件、9508 条标注、场景几何、权重、预测资产、逐条结果、图片及原始 API 输出。输入与历史结果仍位于本机源实验目录；脚本的绝对路径与既有环境依赖被如实保留。不要直接在本快照执行 run 并误以为输入已经随 Git 提供。

主要入口：retry_all_false_gpt61.sh、retry_all_false_gpt61.py；运行覆盖：visual109_runtime/ 与 visual109_policy.py；依赖：source_r18/local/、retry_null.py、retry_state.py、retry_null_progress.py、scanrefer_metrics.py、verify.py。

当前请求配置为第三方兼容网关的 gpt-6.1-sol，temperature=0，保持原有 thinking、schema、预算及重试设置；RL 关闭。配置详情见 freeze_20261028/RUN_PROTOCOL_SUMMARY.json。该摘要删去了凭据指纹和已知失败样本 ID 列表；原协议文件的哈希保留供本机核验。模型名称是请求标识，不据此推断网关内部实现或价格。

当前任务只提交快照和交接计划，没有停止、重启、重新配置或改写正在运行的实验。下一位模型的执行入口是 GOAL_HANDOFF_67_AND_ABLATIONS.md。

本机离线测试（会读取源实验既有基线，不调用模型）：在源实验目录设置 PYTHONDONTWRITEBYTECODE=1，用既有 .venv Python 执行 unittest test_all_false_gpt61 test_visual109 test_null_visual_recall。单独克隆本仓库后，依赖场景和历史输出的测试需要恢复原有输入；不要把缺数据报错当成代码回归。
