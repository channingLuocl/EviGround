# EviGround

**ReAct-Based Visual–Geometric Reasoning for 3D Visual Grounding**

通过多类别假设、主动视觉观察和几何测量获取证据，完成三维视觉定位。

基于预测物体类别的 3D 视觉定位。默认使用 **Qwen r18 + Top-5 + 开启思考**；冻结的 Nr3D 完整单轮结果为 **69.66%（5,437 / 7,805）**，不累计历史正确结果。

```text
src/eviground/   核心代码与命令行
docs/           使用说明与历史文档
tests/          离线测试
data/           查询清单与标签常量
scripts/        启动和验证脚本
third_party/    原项目及第三方代码
outputs/        冻结输入、结果与审计
```

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

在 `.env` 填入 `EVIGROUND_API_KEY`，并准备场景目录；密钥和本地场景数据不提交到 Git。现有 `SEEGROUND_*` 配置仍可读取，同名配置以 `EVIGROUND_*` 为准。

## 运行

```bash
eviground --scene-root /path/to/prepared_scenes \
  --output outputs/runs/r18_top5 --prepare-only

eviground --scene-root /path/to/prepared_scenes \
  --output outputs/runs/r18_top5
```

默认 Top-5、本地并发 8、API 并发 8。同一配置使用同一输出目录续跑，保留已完成的正确、错误和失败结果；修改配置后使用新目录。

详见 [运行指南](docs/usage.md)、[代码布局](docs/architecture.md) 和 [冻结结果](outputs/r18_qwen_top5_full7805_freeze_20261023/README.md)。

原始 SeeGround 组件及其说明保留在 `third_party/` 与 `docs/upstream/`；历史冻结记录保留原始名称和哈希。
