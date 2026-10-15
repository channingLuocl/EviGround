# v8：物体类型预测子集

v7.4 代码已保存在 Git 主分支 `master`，提交 `9bce565`，并已推送至 `origin/master`。
v7.4 的冻结全量运行结果与归档保留原位置。v8 开发分支为 `codex/seeground-v8`。

## 本轮实现

新增独立实例类别预测，不改变已有定位器。输入是彩色场景点云与已有实例划分，
每个实例渲染三个独立视角、一个局部上下文视角，拼成四视图，交给 `.env` 中配置的
视觉模型识别。上下文图只标记待识别实例；GT 类别、作者预测类别和 Nr3D 查询均不进入模型输入。
使用作者相同的 ScanNet200 公共类别词表，排除 wall/floor/ceiling，共 197 类。

模型输出最有依据的类别、其他可能类别、可见性与视觉理由。无法识别时允许 null，
不通过 GT 答案替换预测。这是第一版多视角识别基线，尚未证明优于作者预测方式。

## 运行 100 个物体

```bash
cd /Users/luocl/Desktop/seeground/reproductions/SeeGround
local/run_local_v8_categories.sh --limit 100
```

默认本地 10 并发、API 8 并发，复用现有 `.env` 模型配置。每个实例的首次识别发出一次
逻辑请求，严格输出接口可能触发客户端自身的格式重试。无需重新安装模型或下载权重。

抽样单位是唯一的 `(scene, bbox_id)`，固定种子 42，不重复预测多条描述中的同一物体。
全量 130 个场景共 3,413 个 OLT 实例，因此子集结束显示 `100/100`；
`7805/7805` 是定位描述数，与类别预测的物体数不同。类别子集均匀抽取物体，不代表查询频次加权准确率。

逐条标准输出为以下四个字段。这里是**格式示例，不是真实推理结果**：

```text
{"bbox_id": 12, "target": "chair", "predicted_object": "chair", "correct": true}
```

`target` 是评分端读取的 GT 类型；`predicted_object` 是模型类型。进度条在 stderr 中显示
已完成数、耗时、ETA 和累计 `True 正确数/已完成数 = 准确率`，与现有批处理风格一致。
不同场景中的实例 ID 可能重复，完整场景 ID 记录在文件和汇总中。

默认输出：`local_outputs/v8_category_subset`。`local_outputs` 当前指向 KESU，
实际路径为 `/Volumes/KESU/论文/paper/reproductions/SeeGround/local_outputs/v8_category_subset`。

## 续跑与扩大子集

重复相同命令直接续跑：已完成预测无论答对答错都复用，不按答案挑选历史结果。
按一次 Ctrl+C 停止安排新物体，正在请求的实例会保存后退出。

把子集扩大到 300 个，前 100 个的预测会继续复用：

```bash
local/run_local_v8_categories.sh --limit 300
```

固定种子抽样是可扩展的前缀，因此不是重新抽取一批 300 个物体。
之后需要分类全量实例时可用 `--limit 0`；这是类别预测全量，不会自动运行 7,805 条定位。

只对接口／程序错误重试，保留原错误记录与原始响应：

```bash
local/run_local_v8_categories.sh --limit 100 --retry-errors
```

识别错误和无法识别的已完成答案不会被此参数重跑。遇到非成功 HTTP 状态、网络错误或超时，
停止安排新的实例，保存正在进行的结果，避免接口异常时刷出大量 null。
错误具体原因写到 stderr 与每条 `prediction.json`。

更换模型、代码、输入、随机种子时需指定新目录，避免混合实验：

```bash
local/run_local_v8_categories.sh --limit 100 --output local_outputs/v8_category_subset_experiment2
```

只生成抽样与运行配置，不调用模型：

```bash
local/run_local_v8_categories.sh --limit 100 --prepare-only
```

## 结果与评分

- `config.json`、`inputs.json`、`selection.json`：模型／代码／输入哈希和固定抽样。
- `results.json`：全部已完成实例及汇总；包含 v8 与作者预测在**相同已完成物体**上的准确率。
- `<scene>/object_<ID>/request.json`：实际文本指令、公共词表、几何输入与图片路径。
- 同目录下的四张原视图、`instance_views.png`、`response.json`、`prediction.json`、`evaluation.json`。
- 错误重试的旧结果放在实例目录下 `attempts/`。
- `<scene>/predicted_objects.json`：仅包含新预测类别的物体表；`complete` 标明是否覆盖整场景，
  不把未识别／未运行实例偷偷替换为 GT 或作者预测。此表尚未自动接入定位器。

控制台 `correct` 和汇总 `accuracy` 采用固定词汇归一：大小写、空格／连字符、公共词表规则生成的
复数及 sofa/couch、television/tv、trashcan/trash can。`office chair/chair`、`table/desk`、
`shelf/bookshelf` 保持不同。另存 `exact_correct`、`exact_accuracy` 供严格字符串比较。
API 错误与无法识别均计入当前总数，不从分母中删去；汇总分别记录 `errors` 和 `unrecognised`。

这次分类验证适合检查视觉识别质量。后续需要固定分类结果，接入同一定位器，
分别运行作者预测类别、新预测类别、GT 类别对照，测量最终定位效果。

## 离线测试

```bash
PYTHON_DOTENV_DISABLED=1 SEEGROUND_API_PROVIDER=openai SEEGROUND_V4=0 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=local OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m unittest \
  test_object_classification_v8 test_category_review test_frozen_run test_v7_4
```

测试使用合成点云和模拟接口，不产生模型费用；涵盖真实渲染、请求隔离、输出格式、
错误恢复、子集扩展、模型／输入变化拒绝复用，以及冻结目录保护。
