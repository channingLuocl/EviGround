# 房间中心先验对照

运行 `run_room_center_experiment.py`，一次输出两种方法。只接受 facing_anchor 清单，不能用于 from-the-door 描述。

- center：对齐后的完整扫描网格 XY 边界中点，朝向锚点中心。只作为虚拟视角，不检查地面和碰撞。AABB 中心不保证位于真实房间内部。
- central_free：中心 1 米内所有满足原有地面和障碍规则的格点，各自朝向锚点。全部选同一候选才为 stable，否则为 conflict；无位置则为 ineligible。不会退回虚拟中心。

为了隔离视角先验影响，沿用最低网格点上方 8 厘米的地面估计、二维物体 bbox 障碍检查和原有横向候选排序；不含墙分割、连通性、可见性和真实三维相机投影。地面估计遗漏或悬挂物水平投影导致的拒绝仍可能存在。

```bash
python3 viewpoint_exploration/run_room_center_experiment.py \
  --manifest results/manifests/facing_reviewed_dev10.jsonl \
  --objects-dir data/seeground_object_lookup_table/nr3d/gt \
  --mesh-dir data/scannet_minimal \
  --output-dir results/room_center_dev10
```

summary.json 保存参数和两种方法的覆盖率、稳定输出正确率、全集正确输出比例。每种方法各有 per_query JSONL，包含观察位置、方向和预测。数据错误单列 error，不能解释为几何不可判定。该开发集已用于诊断，不作为未见测试集。固定参数后比较原有方案的 8/10 覆盖和 8/10 正确输出；保留所有失败记录。
