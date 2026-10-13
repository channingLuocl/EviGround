# SeeGround v7.1 计划

记录日期：2026-10-13。当前正在执行 v7 的 17 条 null 重跑，等重跑结束后再修改推理代码。

## 执行顺序

1. 等 v7 的 68 条首次运行结果和当前 17 条 null 重跑都完成。
2. 冻结并汇总 68 条结果，分别记录首次结果、null 重跑结果和最终合并结果。
3. 修改代码为 v7.1，先验证新增语义和现有规则，再确定需要重跑的开发样例。
4. 使用 v7.1 运行 `nr3d_size_300.json` 的 300 条验证集。

## v7.1 首要修改

### 1. 补齐宽度比较语义

`wider` 当前没有进入 `SIZE_WORDS`，因此被解析为 appearance 条件，没有生成尺寸约束；即使 GT 框存在，后续也不会读取宽度并比较候选。

需要补齐：

- `wider` / `widest` → 宽度或横向 extent，选择较大值；
- `narrower` / `narrowest` → 同一指标，选择较小值；
- 保留 `longer`、`shorter` 与 `taller`、`shorter` 的语义区分；
- 使用物体朝向或 oriented bounding box 确定横向宽度，不固定使用世界坐标 X；
- 在 rail、board、book、picture 等薄或旋转物体上检查指标是否可用，指标不可靠时保留 unknown 而不是臆测。

验收样例：`scene0146_00: the wider of the two rails.` 应产生两个候选的宽度约束；若 GT 几何可用，应输出唯一候选，不再因为 `size_constraints=[]` 正常弃答。

实现状态（2026-10-13）：已完成。`wider/widest/narrower/narrowest` 在 v7 标志下默认启用，指标为标注框 `horizontal_short_extent_m`；`--retry-false` 已实现并通过 25 个相关单元测试。全量 744 个用例中，104 个失败/错误在改动前即存在（依赖特定环境变量的旧用例），与本次修改无关。

### 2. 保持并回归现有 v7 规则

- 参考物和嵌套关系先绑定，再执行尺寸比较；
- 单候选兜底保留，但检查候选删除依据；
- 复核中明确的候选矛盾必须传入实际否决字段；
- 书、图片等薄物体检查面积与体积代理的冲突；
- strict schema、接口错误分类和有限恢复保持可追踪。

### 3. 增加 `--retry-false` 手动重跑参数

用户要求：模型有随机性及幻觉，除 null 外，选错编号的样本也需要支持再次运行。

- `--retry-false` 选择已有评估中 `correct=false` 的全部样本，包括错误编号、正常弃答和接口/响应失败；跳过已有 `correct=true` 的样本。
- 与 `--retry-null`、`--retry-errors` 互斥，要求明确指定已有输出目录，并遵守现有代码、输入和设置一致性检查。跨版本运行使用新目录。
- 对选中的样本重新执行推理，提供新的恢复预算，保留旧预测、评估及日志；汇总使用最新一次结果，即使最新结果仍错误或变为 null。
- `correct` 和标注编号仅用于批次选择及离线评估，不向模型或推理逻辑提供正确答案。
- 保存重跑清单及次数，分别报告首次结果和重跑后的结果，避免将多次针对 false 重跑的累计成绩当作单轮成绩。

验收：混合正确编号、错误编号、正常 null、接口失败的缓存时，仅重跑后三类；旧尝试可追溯，最新汇总更新正确，参数冲突和跨版本目录复用会被拒绝。

## 300 条运行约定

300 条清单为 `local/validation/nr3d_size_300.json`，共 92 个场景。启动入口 `local/run_local_v7_1.sh` 已创建，默认 workers=2、api_concurrency=2，输出目录默认 `local_outputs/v7_1_size_300_<时间戳>`。

```bash
local/run_local_v7_1.sh
```

失败重跑（需已有 run）：

```bash
SEEGROUND_RUN_DIR=<已有 run 目录> local/run_local_v7_1.sh --retry-null   # 只重跑 null
SEEGROUND_RUN_DIR=<已有 run 目录> local/run_local_v7_1.sh --retry-false  # 重跑所有 correct=false
```

先确认场景准备齐全：`.venv/bin/python local/prepare_size_validation.py --manifest local/validation/nr3d_size_300.json`。旧 v7 结果保留作为基线。

## v7.1 验收指标

- `wider` 等宽度词确实生成可执行的 `size_constraints`；
- 68 条上比较首次结果、null 重跑合并结果和 v7.1 回归结果；
- 300 条同时报告 accuracy、answer coverage、answered accuracy、accepted accuracy、null/error 数、API 调用次数和耗时；
- 单独统计接口失败、正常弃答、关系绑定失败、尺寸指标冲突和标注/扫描疑似争议，不能只看总 accuracy。
