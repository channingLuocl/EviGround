# SeeGround 本地 API 试运行（Mac）

原工作环境已建立独立 `.venv`（不随仓库上传），无需 CUDA、PyTorch3D 或本地大模型。原论文代码保持原样，新增 `local/` 提供 CPU 点云预览和图片＋物体表到 API 的最小定位入口。

## 当前验证状态

- 已读取本地 ScanNet scene0000_00：81,369 个点，生成真实点云预览。
- 已生成合成样例：4 个带编号的方盒，分别代指两把椅子与两台电脑。用于检查软件链路，不是逼真的场景，不是 ScanNet 数据，不产生论文准确率。
- 离线请求构造检查通过。
- 4 项测试通过：遮挡点过滤、重复 ID 拒绝、离线检查不联网、模拟 API 返回 ID 后正确查出三维包围框。
- 尚未调用真实 API；需要填入配置。真实点云目前缺少物体实例标注与查找表，所以仅作无编号预览。

## 新电脑安装

```bash
cd paper/reproductions/SeeGround
python3 -m venv .venv
.venv/bin/python -m pip install -r local/requirements-lock.txt
cp .env.example .env
chmod 600 .env
```

使用 Python 3.11 或更高版本。安装后再按下面步骤配置与测试。

## 1. 填写配置

在仓库根目录的 `.env` 文件填写：

```dotenv
SEEGROUND_BASE_URL=你的服务商提供的兼容接口基础地址
SEEGROUND_MODEL=你的账号可调用的准确模型ID
SEEGROUND_API_KEY=你的密钥
```

Base URL 通常以 `/v1` 结尾，应直接按服务商提供的地址填写，不要填文档网页，也不要重复附加 `/chat/completions`。当前入口只支持 OpenAI Chat Completions 兼容接口，模型须支持 Base64 图片输入。若服务商要求额外参数，需要对应适配。

`.env` 已加入 Git 忽略规则，权限为 600。不要把密钥发到聊天中。当前程序只向配置的服务发送指定图片、查询与物体表。

## 2. 运行离线样例

```bash
cd paper/reproductions/SeeGround
.venv/bin/python local/run.py demo
.venv/bin/python local/run.py ground \
  --image local_outputs/demo/scene.png \
  --objects local_outputs/demo/objects.json \
  --query 'Find the laptop nearest to the red chair.' \
  --dry-run
```

`--dry-run` 不联网，不计 API 费用。

## 3. 发起一次真实 API 测试

填好 `.env` 后，去掉 `--dry-run`：

```bash
.venv/bin/python local/run.py ground \
  --image local_outputs/demo/scene.png \
  --objects local_outputs/demo/objects.json \
  --query 'Find the laptop nearest to the red chair.'
```

这会产生一次 API 调用，可能收费。结果写入 `local_outputs/prediction.json`，包含模型名、预测编号、三维框、原回答及服务端提供的 token 使用量。合成样例设计的目标编号是 2，但不保证实际模型回答正确。测试文件中的模拟成功不能作为真实 API 成功的证据。

## 4. 查看真实 ScanNet 点云

```bash
.venv/bin/python local/run.py preview \
  --ply /path/to/scannet/scans/scene0000_00/scene0000_00_vh_clean_2.ply
```

输出 `local_outputs/scannet_preview.png`。目前是固定观察视角，不执行查询解析或论文视角自适应策略；无物体实例标注时不伪造编号。程序使用 CPU 针孔投影与逐像素最近深度，可见表面标号仅在传入点级实例 ID 的合成样例中演示。

## 5. 与原论文复现的边界

当前完成的是本地可运行的探索入口，不是完整基准复现：

- 使用 NumPy/Pillow CPU 渲染，未复刻作者 PyTorch3D 相机校准与光栅化参数。
- 使用简化测试提示，未运行第一次查询解析和查询自适应视角选择。
- 第三方 API 模型可能与论文 Qwen2-VL-72B 不同。
- 尚未准备真实场景的实例分割、三维物体表、Nr3D 查询及官方评测口径。

下一步：先确认 API 图像请求成功，再补齐真实 scene0000_00 的实例标注与查找表，将真实场景接入；之后再接查询解析与相机策略，最后才评测。

安装依赖的锁定清单：`local/requirements-lock.txt`。检查命令：`.venv/bin/python local/test_local.py`。

## 已跑通：真实场景两阶段试运行（2026-09-08）

最新进展覆盖了上文部分待办：已准备 `scene0011_00` 的真实点云、实例分割、坐标对齐矩阵及作者 Nr3D 查找表，并接入“API 解析查询 → 匹配目标/参照物 → CPU 选视角并标号 → API 选目标”。

- 点数：237,360；作者查找表候选数：27。
- 逐实例重建包围框与作者 GT 查找表比较，最大坐标差约 `4.53e-7 m`。
- GT 类别表仅用于核对，模型输入使用作者提供的预测类别表。没有将目标真值编号传给模型。
- 手写探索查询：`Find the large table surrounded by chairs.`
- `qwen3-vl-flash` 解析出 table / chairs，最终预测 ID 2；图中对应中央长桌，程序查出了它的包围框。
- 此查询并非 Nr3D 原始标注，不报告基准准确率。Nr3D 官方 Google Drive 描述下载及在线浏览器目前在本机超时。
- CPU 投影、类别词匹配、提示词等仍为本地适配，不能等同作者原始实验。稀疏点渲染也与论文图片质量不同。

### 本机重跑

在本目录安装依赖并配置 `.env` 后：

```bash
# 原始文件放在 local_outputs/downloads，准备时不调用 API
.venv/bin/python local/prepare_scene.py --scene scene0011_00

# 运行一次新查询会调用 API 两次，可能收费；run-name 必须使用新名称
.venv/bin/python local/real_scene.py \
  --query 'Find the large table surrounded by chairs.' \
  --run-name trial_02
```

文件来源：

- 作者查找表：[seeground_object_lookup_table.zip](https://github.com/user-attachments/files/18056532/seeground_object_lookup_table.zip)，本机保存为 `local_outputs/downloads/olt.zip`。
- ScanNet 场景目录：`http://kaldir.vc.in.tum.de/scannet/v2/scans/scene0011_00/`。按数据许可使用以下四个文件：`scene0011_00_vh_clean_2.ply`、`scene0011_00.txt`、`scene0011_00.aggregation.json`、`scene0011_00_vh_clean_2.0.010000.segs.json`。

产物目录 `local_outputs/scene0011_00/`：`objects.json`、`objects_gt.json`、`scene.npz`、`overview.png`、`provenance.json`。每次查询独立子目录内保存 `run.json`（解析、相机、标号和适配说明）、`view.png`、`prediction.json`。

这些数据、图像与 API 输出仅留本地，不上传 Git。新机器需自行准备数据和配置。


## Nr3D 官方描述：10 条真实试跑

已使用官方 Nr3D CSV 中 scene0011_00 的前 10 条 `correct_guess=True` 描述运行本地 API 流程，10 条均完成，按目标编号相等计分为 7/10。选题在推理前固定；没有把正确答案传给模型。单场景小样本、Qwen3-VL API、CPU 渲染及简化类别匹配均不同于论文完整评测，不能用 70% 与论文表格直接比较。

在 SeeGround 目录运行：

```sh
.venv/bin/python local/batch_nr3d.py
```

默认读取 `local_outputs/downloads/nr3d_official.csv` 和已准备的 `local_outputs/scene0011_00`。官方描述下载入口：https://drive.google.com/file/d/1qswKclq4BlnHSGMSgzLmUu8iqdUXD8ZC/view 。数据需自行下载，不随代码上传。下载工具若无法连接，可使用本机已配置的系统代理。

结果位于 `local_outputs/nr3d_trial_10/report.html`，包括原文、正确编号、预测编号、模型解释及每题图片；`manifest.json` 保存数据哈希和固定选题，`results.json` 保存统计。重复运行复用已完成预测，只重试失败题；更改模型或代码后，应使用 `--output local_outputs/nr3d_trial_new` 创建新实验，避免混用缓存。失败请求计入总数。

初步错误线索：第 3、6 题的正确目标 15 在作者预测类别表中为 water cooler，而 GT 类别为 trash can；第 4 题正确目标 24 未出现在当次图片的可见编号列表中。这些是可检查的影响因素，尚不能单独确定因果。GT 类别只用于事后分析，没有替换推理输入。

离线校验：`.venv/bin/python -m unittest discover -s local -p 'test*.py'`，6 项通过。
