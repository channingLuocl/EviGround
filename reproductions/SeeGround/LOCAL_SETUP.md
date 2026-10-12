# 本地运行布局

代码在桌面 `~/Desktop/seeground`，Git 分支为 `master`。场景、CSV 和运行结果在外接盘：

`/Volumes/KESU/论文/paper/reproductions/SeeGround/local_outputs`

代码目录中的 `local_outputs` 是指向该资源目录的软链接，因此不会把数据提交到 Git。

更新代码：

```bash
cd ~/Desktop/seeground
git pull --ff-only origin master
```

运行 v6.1：

```bash
cd ~/Desktop/seeground/reproductions/SeeGround
local/run_local_v61.sh
```

`SEEGROUND_WORKERS` 控制本地 CPU worker，默认 32；`SEEGROUND_API_CONCURRENCY` 控制同时请求模型的数量，默认 4。结果目录默认写入 KESU 资源盘下的 `local_outputs/v6_1_local_*`。
