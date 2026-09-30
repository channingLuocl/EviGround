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

### 从官方 Nr3D CSV 生成审核清单

Nr3D 的官方 `nr3d.csv` 不是本实验的直接输入。先用下面的脚本筛选：人工听者答对、明确提及目标类别、
含有且只含一种 left/right 词、并且对象表中存在至少两个同类候选的查询。脚本按固定随机种子抽样，
输出待审核 CSV；它不会猜测锚点，也不会将文本中的方向词直接当作最终关系。

```sh
python3 viewpoint_exploration/prepare_nr3d_review_sheet.py \
  --nr3d-csv /path/to/nr3d.csv \
  --objects-dir /path/to/seeground_object_lookup_table/nr3d/gt \
  --limit 200 \
  --seed 20260920 \
  --output /path/to/review/nr3d_left_right_review.csv
```

在审核表中逐行填写 `anchor_id` 和最终确认的 `relation`，需要时填写逗号分隔的
`allowed_headings_deg`，例如 `80,90,100`。只有将 `review_status` 改为 `approved` 的行才会被
转换。`relation_hint`、`candidate_ids` 和 `scene_object_catalog` 只辅助人工阅读；其中后者列出同场景
的 `实例 ID:类别`，可用来填写锚点。`target_id` 保留为评估字段，不能据此填写锚点。

```sh
python3 viewpoint_exploration/review_sheet_to_manifest.py \
  --review-csv /path/to/review/nr3d_left_right_review.csv \
  --output /path/to/review/nr3d_left_right_manifest.jsonl
```

再将生成的 JSONL 传给上一节的批量实验脚本。第一轮建议审核 100 条而不是直接审核 200 条；
确认对象表 ID 与 Nr3D 目标 ID 匹配后，再固定规则扩大样本。

### 本轮范围

本实现只检验水平 `left/right` 的几何稳定性。它没有解析自然语言、预测真实说话者位姿、建模相机
位置/高度、过滤遮挡与可达性，也没有调用 VLM。下一阶段才可在固定的冲突样本上增加语言线索、
可见性过滤或主动感知基线。

### 第二轮：`facing_anchor` 的 mesh 支持位姿近似

完整 360° 朝向枚举会让任何非退化的左右关系翻转，因此它只能作为“语言未约束视角时的下界”诊断，
不能证明方法有效。`run_facing_anchor_pose_experiment.py` 用于更严格的第二轮：只纳入人工确认包含
`facing X`、`looking at X` 等单一锚点朝向线索的句子。

它先使用同场景 `.txt` 中的 `axisAlignment` 将原始 mesh 对齐到 SeeGround GT 对象框坐标系，再从低位
mesh 顶点生成 2D 地面候选格点，并剔除落在 GT 物体框（膨胀一个人体半径）内的位置。在每个保留位置，
观察者朝向锚点；再计算 left/right 的赢家是否一致。该脚本不使用 `target_id`
选择位置或赢家。它也没有建模可见性、遮挡、房间连通性、真实说话者位置或相机高度，所以输出必须称为
“mesh 支持的可行位姿近似”，不能称为真实位姿恢复。

```sh
python3 viewpoint_exploration/run_facing_anchor_pose_experiment.py \
  --manifest viewpoint_exploration/review_artifacts/nr3d_facing_anchor_manifest_5.jsonl \
  --objects-dir data/seeground_object_lookup_table/nr3d/gt \
  --mesh-dir data/scannet_minimal \
  --output-dir results/facing_anchor_mesh_reviewed5
```

可用下面的脚本将某条结果画成俯视 SVG。蓝框是锚点，橙框是同类候选，圆点是可行观察位置；圆点颜色表示该位置下
`left/right` 选择的候选。

```sh
python3 viewpoint_exploration/render_pose_case_svg.py \
  --per-query results/facing_anchor_mesh_reviewed5/per_query.jsonl \
  --objects data/seeground_object_lookup_table/nr3d/gt/scene0697_00.json \
  --query-id 33271 \
  --output results/facing_anchor_mesh_reviewed5/query_33271_topdown.svg
```

### 第三轮：规范锚点视角

`facing_anchor` 的任意可站位置仍可能把同一句话解释成不同方向。第三轮将“面对锚点”近似为锚点每一个可能
正面对应的一个规范朝向：统计锚点四侧局部带内，既有 mesh 地面支持又不落在物体框内的格点数；保留得分至少为
最高分 80% 的方向。如果保留方向选出同一目标，结果为稳定；否则保留冲突，而不是用任意位置打破平局。

```sh
python3 viewpoint_exploration/run_canonical_anchor_view_experiment.py \
  --manifest viewpoint_exploration/review_artifacts/nr3d_facing_anchor_manifest_5.jsonl \
  --objects-dir data/seeground_object_lookup_table/nr3d/gt \
  --mesh-dir data/scannet_minimal \
  --output-dir results/canonical_anchor_view_reviewed5
```

这是墙体/空闲空间的第一种几何代理，尚未显式分割墙面或做射线可见性检测；因此它用于检验假设和暴露失败案例，
不能称为完整的可达性或可见性模型。

### 未见样本扩展

机制验证完成后，用 `prepare_facing_anchor_manifest.py` 从原始 Nr3D CSV 自动构建未见样本。它只保留：听者答对、
包含单一 left/right 词、至少两个同类候选、包含 facing/looking-at 线索，并且该线索后的锚点类别在场景中恰好对应
一个实例的查询。锚点不需要人工填写；`target_id` 只保留为评估字段。

```sh
python3 viewpoint_exploration/prepare_facing_anchor_manifest.py \
  --nr3d-csv data/nr3d.csv \
  --objects-dir data/seeground_object_lookup_table/nr3d/gt \
  --limit 50 \
  --seed 20260922 \
  --exclude-manifest viewpoint_exploration/review_artifacts/nr3d_facing_anchor_manifest_5.jsonl \
  --output results/manifests/nr3d_facing_anchor_auto50.jsonl
```

下载该清单涉及的 ScanNet mesh 与 `.txt` 时，使用下列可重跑下载器。它以 `.part` 临时文件原子写入，
中断后再次运行会跳过已完成文件、重试未完成文件。

```sh
python3 viewpoint_exploration/download_scannet_manifest_assets.py \
  --manifest results/manifests/nr3d_facing_anchor_auto50.jsonl \
  --output-root data/scannet_minimal
```
