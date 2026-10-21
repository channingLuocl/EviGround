# R18 GPT-6.1 relay 冻结快照（2026-10-22）

本目录记录当前 R18 代码和预测类别输入，分支为 `codex/r18-gpt61-formatfix-freeze-20261022`。

- `reproductions/SeeGround/local/` 已同步 R18 工作副本；`freeze_manifest.json` 保存源码逐文件 SHA256。
- 预测类别输入使用 `outputs/r8_grounding_input_20261020/20261020_r8_object_predictions/`，同时保留原始 ZIP；两者的哈希也在清单中。运行设置为 Top-5 候选类别，禁止读取 GT 类别。
- `dev96_manifest.json` 是固定的 96 条开发选择；快照时 R18 GPT-6.1 relay 已完成 4/96，`r18_formatfix_dev96.log.snapshot` 只是当时刻日志。dev96 和完整 7805 条运行均未在本快照中宣称最终指标。
- 场景点云、模型权重、凭证、虚拟环境和不断变化的 `local_outputs` 不写入 Git。

源码检查：

```bash
cd reproductions/SeeGround
PYTHONPATH=local .venv/bin/python -m unittest local.test_r18_react local.test_gpt61_action_format
```

结果为 27 项通过；GPT relay 图文/配置定向检查另有 3 项通过。恢复或复现实验时，以 `freeze_manifest.json` 的哈希和 `dev96_manifest.json` 为准。
