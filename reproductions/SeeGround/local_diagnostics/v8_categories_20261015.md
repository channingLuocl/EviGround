# v8 类别预测实现验证

日期：2026-10-15。

- Git 主分支 `master` 已快进至 v7.4 提交 `9bce5658194499424d666278dd9930a251c0f6b7`，并推送到 `origin/master`，远端读取确认一致。
- 新开发分支：`codex/seeground-v8`。
- v7.4 冻结结果 SHA-256 仍为 `9fd6b1cc86a9ff3001531b1f36d232a90dcac2e9b47f7ac046e23e38971e38a8`，与归档时一致。
- 离线测试 21 项通过：`test_object_classification_v8`、`test_category_review`、`test_frozen_run`、`test_v7_4`。
- 新启动脚本通过 `bash -n`，差异通过 `git diff --check`。
- 在真实场景 `scene0500_00` 实例 12、`scene0678_00` 实例 9 上实际渲染四视图；前者图片已人工式视觉检查，布局、实例与局部上下文标记正常。
- 子集准备命令实际执行：`local/run_local_v8_categories.sh --limit 100 --prepare-only`。
- 生成固定种子 42 的 100 个唯一实例，覆盖 66 个场景；默认输出目录 `local_outputs/v8_category_subset`。
- 该目录只有选择、配置、输入哈希与空汇总，没有 `response.json`；没有发起付费模型推理，没有测得新分类准确率。

覆盖的关键行为：GT／作者标签／查询不进入预测函数与实际请求；真实合成点云渲染；
非法模型类别／响应拒绝；四字段控制台输出；答错也缓存；子集扩大保留既有实例；
模型或输入改变拒绝复用；API 限流错误停止安排新实例；错误重试保留旧尝试；冻结目录拒绝写入。

这一版是多视角实例分类基线，未修改 v7.4 定位规则，也未自动启动全量定位。
