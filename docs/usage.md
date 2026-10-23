# EviGround 运行与配置

## 安装

在仓库根目录使用 Python 3.10 或更新版本：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

在 `.env` 中填写 `EVIGROUND_API_KEY`。默认地址、模型与参数沿用已完成的 Qwen Coding Plan 运行；`openai` 是兼容接口名称，实际调用模型为 `qwen3.7-plus`。如使用其他服务地址，应相应修改 `.env`。

当前环境变量统一使用 `EVIGROUND_*`。旧 `.env` 或进程环境中的 `SEEGROUND_*` 会在配置加载时兼容映射，新名称优先。仓库外运行普通 wheel 时，可设置 `EVIGROUND_ROOT=/path/to/repository`。

核心定位使用 CPU 场景渲染，不要求安装分类器训练环境。历史分类训练依赖与 GPU 工具见 `docs/history/local/` 和 `third_party/`。

## 场景和类别输入

场景需要按以下格式准备；使用自己已有的场景目录，无需拷贝到仓库：

```text
prepared_scenes/
  scene0011_00/
    scene.npz
    scene_instances.json
    objects_gt.json
```

默认查询清单是 `data/validation/nr3d_val_full.json`，包含 7,805 条。默认预测类别输入是 Git 中冻结的 `outputs/r8_grounding_input_20261020/20261020_r8_object_predictions/`，可通过 `--predictions` 指定其他导出。

`objects_gt.json` 用于评测侧比较目标与三维框，真实物体类别不作为模型的类别输入；模型使用 Top-5 预测类别。Top-5 分数沿用原来的融合分数定义，不当成概率。

## 准备、运行与续跑

```bash
# 检查所有输入、生成清单，不调用大模型
eviground --scene-root /path/to/prepared_scenes \
  --output outputs/runs/r18_top5 --prepare-only

# 完整运行
eviground --scene-root /path/to/prepared_scenes \
  --output outputs/runs/r18_top5
```

也可以使用 `python -m eviground` 或 `bash scripts/run_grounding.sh`。本地并发和 API 并发默认均为 8：

```bash
eviground --scene-root /path/to/prepared_scenes \
  --category-top-k 5 --workers 8 --api-concurrency 8 \
  --output outputs/runs/r18_top5
```

Top-1 对照使用 `--category-top-k 1` 并选择新的输出目录。

同一配置再次运行同一命令即可续跑。已完成的正确、错误和失败条目一律保留，只处理未完成条目。错误与弃答均进入完整分母，不合并历史正确缓存。代码、模型或类别配置改变后必须选择新目录；重构前的冻结目录仅用于查询和校验。

默认启用 r18、思考模式、8,192 最小输出 token 预算、温度 0 和原生 JSON Schema。新代码的源码指纹反映 `src/` 布局，与原始布局的哈希自然不同。

## 离线验证

```bash
PYTHON_DOTENV_DISABLED=1 python -m unittest discover -s tests/unit -t .
bash scripts/verify_results.sh
```

回归测试使用本地数据和模拟模型响应。冻结结果验证不需要密钥或外置硬盘。
