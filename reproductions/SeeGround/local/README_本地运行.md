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
