# NR3D 全量验证

## v7.4：全量描述的场景推理修复（当前入口）

```bash
cd /Users/luocl/Desktop/seeground/reproductions/SeeGround
local/run_local_v7_4_full.sh
```

使用 GT 类别，输出在 KESU 上的 `local_outputs/v7_4_nr3d_val_full_gt`。
重复执行同一命令即可断点续跑；停止时只按一次 Ctrl+C，等待活动样本保存。
历史答对的记录自动导入，不再调用模型。错误、null 不跨版本导入；本目录已完成
的错误结果默认也按断点保留。若要主动重跑本目录错误，运行：

```bash
local/run_local_v7_4_full.sh --retry-false
```

旧 v7.3 的 33 条新推理中有 29 条 null：26 条主要受规则覆盖/候选裁剪限制，
3 条是视觉报告达到 2400 token 上限后截断，均非 HTTP 或网络失败。
例如“stack 最底层”和“两床之间”已在视觉报告中识别出候选，但原规则选择门槛
没有接收这些证据，仍输出 null。最初 300 条偏大小描述，不能代表全量空间语言的覆盖率。

v7.4 在存在规则无法处理的描述、观察框架缺失或规则筛空候选时，直接用原句、
完整实例表、两张带编号场景图及计算出的邻近距离进行一次多模态选择。
其余样本继续原证据流程；若最后没有可区分答案，再调用场景推理。
这些输出明确标记为 `v74_multimodal_tentative`，不是所有语言条件都已证明。
模型只接收场景类别/框，不接收该查询的正确目标编号。无充分证据仍允许 null；
不会用随机编号填充。保留原视觉核验时，报告缩短，并将输出上限提高到 4800。

8 条预选旧 null 的小样本回归：第一轮 4 正确、1 错误、3 因引用格式无效而 null；
修复引用字段的严格 schema 后，复用已正确的 4 条，再跑剩余 4 条，新增 2 条正确、
2 条错误，累计 6/8、0 null，无 API 错误。两次尝试均保留在旧目录。
这是针对失败样本的累计回归结果，不能作为 v7.4 单次全量准确率。

已答对永久保留的缓存策略见下文；它仍是 `cumulative_ever_correct`，
不是论文标准的单次准确率。资源和正确记录的准备检查不发起 API 请求：

```bash
local/run_local_v7_4_full.sh --prepare-only
```

## v7.3：修复后开始和续跑

v7.3 修复 planner 与证据执行器的可选参数不一致：appearance 的多余 inspect
以及 size/position/distance 的多余 context 不再让有效请求被拒绝；
viewpoint 不执行渲染，校验后清理多余视觉字段。
执行器仍验证对象编号、视角和参数类型；证据不足时仍会保留 null。

```bash
cd /Users/luocl/Desktop/seeground/reproductions/SeeGround
local/run_local_v7_3_full.sh
```

此命令使用 GT 类别，固定输出为 KESU 上的 `local_outputs/v7_3_nr3d_val_full_gt`。
再次运行同一命令会续跑，进度包含实时 True 数量和两位小数正确率。
只按一次 Ctrl+C，并等待正在运行的样本保存后退出。

采用“曾经正确就永久保留”的累计策略：启动时自动扫描 KESU 上历史批次，
同一 CSV、样本编号、场景、描述和目标编号曾答对，就保存到
`local_outputs/.correct_results_gt.json`。预测类别另存到 `.correct_results_pred.json`。
代码版本和模型变更不影响正确答案复用；新答对的样本立即原子保存。
同一批次已完成的其他结果仍按断点规则保留，除非显式要求重跑错误。
现有目录中原 300 条评估保留；新的输出目录只自动导入历史正确结果，
错误和 null 不因“曾经正确”策略被导入。

若历史正确答案替换了当前目录中的错误或 null，旧评估归档到该样本的
`superseded_by_correct_cache`，图片和轨迹保留。即使 `--retry-false` 或
`--retry-errors`，已答对的样本也不会再执行。
以后修改推理代码仍用新的输出目录，启动时会自动继承正确缓存：

```bash
SEEGROUND_RUN_DIR="/Volumes/KESU/论文/paper/reproductions/SeeGround/local_outputs/你的新版本目录" \
  local/run_local_v7_3_full.sh
```

该成绩是历次运行累计正确率，不是单一版本或论文标准的一次评测准确率。
汇总文件注明 `evaluation_policy=cumulative_ever_correct`，每条结果保留来源版本，
`groups.implementation_version` 分组只能解释各答案来自哪个版本。
请保留旧目录中的图片与详细轨迹。

仅验证资源、导入旧结果，不发起推理：

```bash
local/run_local_v7_3_full.sh --prepare-only
```

以下为历史 v7.1 运行说明。修复后请使用上述 v7.3 入口。

全量是 SeeGround 作者发布结果中的 **7,805 条描述、130 个场景**，不是全部训练数据。
清单由原始 NR3D CSV 的 ScanNet 验证场景及 `mentions_target_class=True` 生成，
保留重复描述，不额外筛选 `correct_guess` 或大小词。
场景、目标编号、原始描述及重复次数已与作者发布的结果逐项对齐。

- [论文 §4.1](https://arxiv.org/html/2412.04383v2#S4.SS1)
- [作者发布结果](https://github.com/user-attachments/files/18056533/seeground_results.zip)
- [作者使用的 Vil3DRef 标注预处理](https://github.com/cshizhe/vil3dref/blob/main/preprocess/prepare_nr3d_annos.py)

这里只对齐查询范围；推理继续使用本地 v7.1、qwen3.7-plus、GT 物体类别和 CPU 渲染，
不同于论文的 Qwen2-VL-72B 及其原始实现，成绩应注明这个设置。

## 资源

`local_outputs` 指向 `/Volumes/KESU/论文/paper/reproductions/SeeGround/local_outputs`。
原始资源在其 `downloads` 下，处理后场景在各个 `sceneXXXX_XX` 目录。
后台准备日志为 `local_outputs/prepare_nr3d_val_full.log`。
若准备过程被中断，重跑以下命令，已下载和已处理资源会跳过（单个未下载完的文件重新下载）：

```bash
cd /Users/luocl/Desktop/seeground/reproductions/SeeGround
PYTHONPATH=local .venv/bin/python -u local/prepare_size_validation.py \
  --manifest local/validation/nr3d_val_full.json
```

## 开始和续跑

300 条 GT 基线已完成（200/300 = 66.67%），全量场景资源已准备好。
此入口显式使用 GT 类别，并关闭 v7.2，保持 v7.1 基线的推理设置。
开始全量或中断后续跑：

```bash
cd /Users/luocl/Desktop/seeground/reproductions/SeeGround
local/run_local_v7_1_full.sh
```

固定输出目录为 `local_outputs/v7_1_nr3d_val_full`，中断后重复同一命令即可。
每条样本的 `evaluation.json` 原子保存，续跑根据单条结果恢复；
`results.json` 大约每 30 秒刷新一次，正常结束或按 Ctrl+C 停止后再次汇总。
按一次 Ctrl+C 会取消排队样本，等待正在运行的最多 4 条保存后退出。
进度、正确数量和正确率包括已复用和已缓存样本，null 及错误计入分母：

```text
[---------------.........] 188/7805 2% | 6751s | ETA 4022s | True 137/188 = 72.87%
```

默认复用 `local_outputs/v7_1_size_300_20261014_112437`。
启动时要求该源批次全部完成，并核对 CSV、推理代码、模型参数和场景哈希。
v7.2 新增的两处代码仅在预测类别模式或 v7.2 模式下生效；
GT v7.1 复用检查只接受这两处已核对的精确哈希变更，其他推理代码变更仍会被拒绝。
核对记录保存为全量目录中的 `reuse_verification.json`。
正确、错误和 null 全部原样计入，避免只挑选正确结果。
评估文件快照写入全量目录；图片和详细轨迹继续引用原批次，所以请保留原目录。
单独核对资源并导入缓存、不发起模型请求，可运行 `local/run_local_v7_1_full.sh --prepare-only`。

默认续跑跳过已有评估（包括 null 和失败），避免重复计费。
若明确要重跑 null，原尝试会保留：

```bash
local/run_local_v7_1_full.sh --retry-null
```

`--retry-null` 只重跑已有 null，不补跑尚未执行的样本。
之后重复普通全量命令继续未运行部分。
更改模型或推理代码时需要新建输出目录，不会把不同实验设置静默混合。
