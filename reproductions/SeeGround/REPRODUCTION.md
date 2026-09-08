# 来源与复现范围

本目录是第三方论文的复现实验，不是原创 SeeGround 方法。

- 原仓库：https://github.com/iris0329/SeeGround
- 上游快照：`928d827b18345454c6083cf49c9a7348cc04f6c1`
- 论文：SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding（CVPR 2025）。
- 原作者与论文引用见原 README；原 Apache-2.0 LICENSE 保留。

## 本分支改动（2026-09-08）

新增 `local/` 下的 Mac/CPU 试运行入口：NumPy/Pillow 点云渲染、合成场景、OpenAI 兼容视觉 API 调用、返回 ID 查表、离线及模拟接口测试。新增 `.env.example`，扩展 `.gitignore`。原作者算法脚本未修改。

本地文档中的验证记录来自原开发机器；并不表示刚克隆的仓库自带数据、环境或有效 API 配置。

## 未上传内容

API 密钥与 `.env`、虚拟环境、运行结果、ScanNet 点云、原仓库的 `weights/pnext_cls.pth`。如需原始预处理流程，请从上游取得对应权重与合法授权的数据。保留上游跟踪的划分列表、坐标对齐元数据和代码所需的类别常量。

## 复现边界

CPU 渲染和简化提示不是论文完整实现；更换 API 模型后不能直接宣称复现原 Qwen2-VL-72B 成绩。当前待完成：配置真实 API、补齐真实实例查找表、接入查询解析与视角选择、核对评测口径。
