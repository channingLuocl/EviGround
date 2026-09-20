# 可行视角一致性探索

这个小工具不预测“描述者的真实位置”，也不调用任何语言模型。它回答一个更基础的问题：

> 在一组水平观察方向中，给定锚点和 `left` / `right` 关系，哪一个同类候选会被几何规则选中？

如果不同方向选出不同候选，查询对视角敏感；若所有方向都指向同一候选，则该关系在这组采样下稳定。

## 输入与几何约定

输入是物体表 JSON。每个物体需要 `bbox_id`、`target` 与六维 `bbox_3d`，前三项是三维中心。`--heading-deg` 表示观察者朝向世界 XY 平面的角度：0 度朝 +X，90 度朝 +Y。对于每一个朝向，工具以锚点为原点，将候选到锚点的位移投影到观察者右向量上；`left` 选投影最小者，`right` 选投影最大者。

这只是一个可复查的几何诊断，不代表相机位置、可见性、遮挡、物体朝向或自然语言的完整语义。

## 示例

在仓库根目录运行：

```sh
python3 viewpoint_exploration/explore_horizontal_viewpoints.py \
  --objects reproductions/SeeGround/local_outputs/scene0011_00/objects.json \
  --target-category window \
  --anchor-id 18 \
  --relation right \
  --step-deg 10 \
  --output /tmp/window_from_tv.json
```

输出中：

- `per_heading` 给出每个方向的胜出候选和投影；
- `candidate_support` 给出每个候选在采样方向中获胜的比例；
- `is_stable` 只有在所有采样方向给出同一候选时才为真。

## 批量最小实验

`run_horizontal_viewpoint_experiment.py` 对一份人工核验的 JSONL 清单逐条运行诊断。清单字段为：

- 必填：`query_id`、`scene_id`、`target_category`、`anchor_id`、`relation`；
- 可选：原句 `utterance`、真值 `target_id`、相对 `objects_path`；
- 可选：`allowed_headings_deg`。只有当人工已从原句确认朝向线索时才填写它；不填则枚举完整水平圆周。

`target_id` 严格只用于输出后的诊断统计，绝不参与候选选择、方向选择或稳定性判断。示例格式见
`manifest.example.jsonl`；其中 ID 和场景仅为格式示例，不能用于正式实验。

假设每个场景物体表在 `OBJECTS_DIR/<scene_id>.json`；如果现有布局不同，在清单中给每条记录填写
`objects_path` 即可。例如 `objects_path` 可以写成 `scene0011_00/objects.json`。

```sh
python3 viewpoint_exploration/run_horizontal_viewpoint_experiment.py \
  --manifest /path/to/curated_left_right.jsonl \
  --objects-dir /path/to/object_tables \
  --step-deg 10 \
  --output-dir /path/to/results/horizontal_step10
```

输出目录必须尚不存在，包含：

- `per_query.jsonl`：每条查询的候选、每个方向的赢家、稳定/冲突状态及可选的只读真值诊断；
- `summary.json`：总样本数、可纳入样本数、稳定/冲突数量和冲突比例。

`ineligible` 表示当前物体表里少于两个同类非锚点候选、找不到锚点或输入几何不完整；它不应混入
稳定性比例的分母。正式实验前先固定清单、角度间隔和人工核验规则；不得利用 `target_id` 挑选朝向。

### 本轮范围

本实现只检验水平 `left/right` 的几何稳定性。它没有解析自然语言、预测真实说话者位姿、建模相机
位置/高度、过滤遮挡与可达性，也没有调用 VLM。下一阶段才可在固定的冲突样本上增加语言线索、
可见性过滤或主动感知基线。
