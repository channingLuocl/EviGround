# v8.1：独立物体类别预测

修订标识 `object-category-v8.1-independent-r1`，分支 `codex/seeground-v8`。

## 输入与输出

正式推理接口只接受彩色点云坐标、RGB、实例 ID 和公共类别词表：

彩色点云＋实例划分 → 多视角点云渲染 → VLM 独立识别 → 物体类型。

不读取作者现成类别来作候选、复核、覆盖或回退；GT 与查询也不进入推理。
渲染只用点云，不自动读取外部网格。实例划分已经提供，不包含场景实例分割。
公共类别词表为 ScanNet200 排除 wall/floor/ceiling 的 197 类，它不是实例答案。

当前识别器仍是 VLM，不是 PointNeXt；没有加载或训练 PointNeXt 权重。
这次修订纠正输入依赖，尚未证明独立分类准确率提高。此前融合运行的 61% 使用了作者标签，
不能作为独立分类成绩；其同轮独立 VLM 为 40%，且该实验使用实测网格，不能直接当作当前
点云模式的实测成绩。若下一步采用直接编码三维点云的网络，需要匹配的预训练权重与文本
对齐组件，不能用现成答案代替网络推理。

## 仅提供点云的推理入口

NPZ 文件需有三个数组：xyz 为 N×3、Z 向上、单位米；rgb 为 N×3、范围 0..255；
ids 为 N 个整数实例 ID，负值视为不分类的背景。无需 objects.json、GT、manifest 或网格。

下面以现有 scene0131_00 点云中 ID 30 的实例为例：

```bash
cd /Users/luocl/Desktop/seeground/reproductions/SeeGround
SEEGROUND_WORKERS=10 SEEGROUND_API_CONCURRENCY=8 \
local/run_local_v8_categories.sh \
  --cloud local_outputs/scene0131_00/scene.npz --bbox-id 30 \
  --output local_outputs/v8_1_independent_scene0131
```

省略 --bbox-id 则处理输入点云中的所有非负实例 ID；可用 --bbox-id 30 33 指定多个实例。
此入口代码为 local/predict_pointcloud_v8.py，不会加载任何类别答案表。
默认本地 10 并发、API 8 并发，读取现有 .env 模型配置。

输出示例仅说明格式，不代表真实模型结果：

```json
{"bbox_id": 30, "predicted_object": "computer tower", "status": "completed"}
```

没有 GT，因此该入口不计算正确率。结果保存在 predictions.json 及每个实例的 prediction.json；
同时保留视图、请求与响应。输入／代码／模型变化时拒绝复用旧目录。
重复同一命令复用完成结果，--retry-errors 仅重试接口／程序错误，旧尝试归档到 attempts/。
按一次 Ctrl+C 停止新实例并保存正在执行的实例。错误输出 null，不用作者标签填补。

## 固定子集评测入口

```bash
cd /Users/luocl/Desktop/seeground/reproductions/SeeGround
SEEGROUND_WORKERS=10 SEEGROUND_API_CONCURRENCY=8 \
local/run_local_v8_categories.sh --limit 100
```

默认新目录 local_outputs/v8_1_independent_subset。
该入口为了与历史结果保持相同的样本，仅读取 objects.json 的实例 ID／包围框建立清单；
它的 target 字段与 objects_gt.json 的 GT 类别只在评分端使用，均不传给分类器。
源码中已移除作者标签融合、双类别复核与标签回退路径，不能用 --method fusion 开启。

逐实例输出仍为 bbox_id、target、predicted_object、correct，进度条显示累计准确率。
results.json 保存独立分类 accuracy 和仅供比较的 author_accuracy。
评分维持原有词汇归一，不合并 chair/office chair、table/desk 等上下位类别。
接口错误／无法识别都留在分母内，不按答案正误重跑或选择历史成功答案。

扩大子集用 --limit 300，前缀抽样 seed=42 保持一致；--limit 0 是全量实例类别预测，
不是 7,805 条描述定位。这些已查看过错例的实例用于开发，不能宣称独立测试。
本次没有自动启动分类 API 请求；不建议仅为了新版本名重复大量调用，需先明确识别方法改进。

两个入口均支持 --prepare-only，仅检查输入、保存配置，不调用模型。
注意使用不同输出目录区分单场景推理与固定子集评测。

## 历史保留与测试

61% 融合实验的全部 3,029 个源文件已复制并逐文件核验 SHA256，连同融合实现保存至：
/Volumes/KESU/论文/paper/reproductions/SeeGround/archives/v8.1_fusion_100_61_20261016。
原目录 local_outputs/v8_1_fusion_subset 不修改；本机核心 JSON 备份为
/Users/luocl/Desktop/seeground/archives/v8.1_fusion_100_61_20261016_critical。
此前 42.33% 的 300 实例纯视觉实验及冻结 v7.4 记录也保留原位置。

```bash
PYTHON_DOTENV_DISABLED=1 SEEGROUND_API_PROVIDER=openai SEEGROUND_V4=0 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=local OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m unittest \
  test_pointcloud_input_v8 test_object_classification_v8 \
  test_category_review test_frozen_run test_v7_4
```

测试涵盖没有任何标签表时的真实点云渲染与模拟分类、改变 GT／作者标签后预测不变、
接口失败无标签回退、缓存／重试／模型变化保护和已有冻结保护。
