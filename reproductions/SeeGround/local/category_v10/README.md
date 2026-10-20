# v10：监督训练的 PointNeXt 类别适配实验

这是独立实验目录，未修改冻结的 v8.3 分类器和 v9 定位代码。第一阶段冻结 PointNeXt 及已有语义投影，使用它们提取全局及 8 个局部裁块的 512 维特征，训练新增的小模块；不是重新训练整个 PointNeXt，也不是已证明有效的论文创新。

## 设计与对照

三个训练设置使用同一份特征、两视图训练样本、类别交叉熵、弱一致性损失、优化器与开发集选模规则：

| mode | 行为 | 可训练参数 |
| --- | --- | ---: |
| global | 固定文本相似度 + 学习的全局线性分类修正 | 100,864 |
| attention | 全局查询决定所有类别共用的局部注意力，再融合局部类别分数 | 232,593 |
| conditioned | 每个类别与当前竞争类别形成不同查询，选择局部证据并学习融合权重 | 232,593 |

`conditioned` 的竞争类别来自当前全局预测的软分布，不从 GT 插入候选，不做硬 Top-5 截断。输出仍覆盖固定 197 类。训练和推理使用同一个前向函数，类别标签只进入训练损失。注意力和融合权重不是可解释部件标注，也不是经过校准的正确概率；局部区域未被观察到不代表部件不存在。

训练视图为原物体及确定性随机保留 80% 点的增强视图，开发和评估只有原物体。两视图平均 CE + 0.1 × 原视图到增强视图的 KL；原视图概率在 KL 中停止梯度。所有训练对照使用相同目标。轻度删点并不能模拟完整真实遮挡，其有效性仍需实验验证。可用 `--consistency-weight 0` 做一致性损失消融。

每个视图进行 9 次编码。第一版不使用 RGB、不解冻主干；这样可以先隔离聚合模块的作用。RGB 分支、主干微调和更强的数据增强应分别作为后续实验，不能和模块一起加入后混报收益。

## 数据：先准备训练集，禁止使用现有验证场景训练

截至 2026-10-17，本地已有的 130 个点云场景都属于官方验证集，尚无训练场景；本机 PyTorch 无 CUDA。真实训练和真实点云 GPU 前向尚未运行。

已加入官方训练清单 `data/scannet/scannetv2_train.txt`：

- 来源：https://github.com/ScanNet/ScanNet/blob/master/Tasks/Benchmark/scannetv2_train.txt
- SHA256：`96acca299b7855f02824c496b19077904d80996e7ced1bb9f0dac98f7dd4d0c8`
- 1,201 个训练场景；按物理房间 `sceneXXXX` 分组、seed 42、约 10% 内部开发：1,078 train / 123 dev。
- 当前 Nr3D 评估选择是 130 个验证场景、7,805 条描述。已经查看过的验证集不能称为全新独立测试集。

先通过 [ScanNet 官方渠道](https://github.com/ScanNet/ScanNet#scannet-data) 获取 v2 **训练扫描**。分类阶段只需要每个场景的四个文件，不需要 `.sens` 视频：

```text
<raw-root>/sceneXXXX_XX/
  sceneXXXX_XX.txt
  sceneXXXX_XX.aggregation.json
  sceneXXXX_XX_vh_clean_2.0.010000.segs.json
  sceneXXXX_XX_vh_clean_2.ply
```

转换直接读取官方实例 ID 和类别，不需要 `olt.zip` 或作者预测。原始类别经过现有固定词法规范化后映射到 197 类，不合并上下位类别。墙、地板、天花板在所有划分中排除；训练中的其他词表外类别会记录并排除；开发/评估中的词表外目标保留并计错，整体准确率不缩小分母，同时单列词表外数量。类别平均准确率明确只统计词表内类别。

这是 **ScanNet 训练场景上的监督类别适配，之后在 Nr3D 上评估**，没有使用 Nr3D 描述训练语言定位器。若以后只用 Nr3D 训练描述覆盖的场景，应先与官方训练清单取交集并固定该子集，再重新准备全部实验，不能混用两种训练范围。

## 运行顺序

在 SeeGround 根目录执行。下面的 `python` 指已配置好的环境；GPU 特征提取沿用 v8.3 的 NVIDIA CUDA 环境及算子，转换额外需要 `plyfile`（现有本地环境已安装）。不调用语言模型 API，不受 grounding 的 8 并发设置影响。

1. 转换训练扫描，放入与现有验证场景相同的场景根目录。已经存在的场景不会覆盖。

```bash
python local/category_v10/run.py convert \
  --scenes data/scannet/scannetv2_train.txt \
  --raw-root /data/scannet/scans \
  --output local_outputs
```

2. 生成 train / dev / eval 清单。先检查训练场景与**整个官方验证集**无场景/房间交叉，再按房间划分内部开发集。同一实例只产生一个原始样本，不按 Nr3D 描述重复。

```bash
python local/category_v10/run.py prepare \
  --train-scenes data/scannet/scannetv2_train.txt \
  --scene-root local_outputs \
  --output local_outputs/v10_manifests
```

3. 在 GPU 环境提取特征。按物体保存，可用同一条命令继续中断的提取。代码、输入、模型或配置改变会拒绝复用。

```bash
for split in train dev eval; do
  python local/category_v10/run.py cache \
    --manifest local_outputs/v10_manifests/$split.json \
    --output local_outputs/v10_features/$split || break
done
```

评估特征不参与训练选模。特征缓存完成后，可将缓存和文本特征带回本地，再关闭服务器；训练小模块和评分支持 CPU。

4. 用相同预算分别训练三个设置。`--threads 10` 表示 CPU 计算线程数，不是同时启动 10 个实验。默认 40 epoch、batch size 64、lr 0.003，都是初始研究设置。

```bash
for mode in global attention conditioned; do
  python local/category_v10/run.py train \
    --train-cache local_outputs/v10_features/train/features.npz \
    --dev-cache local_outputs/v10_features/dev/features.npz \
    --mode "$mode" --threads 10 --device cpu \
    --output local_outputs/v10_models/$mode || break
done
```

开发集交叉熵最低的 epoch 保存为 `head.pt`。中断会保留已完成 epoch 的历史和最优模型；本版训练不提供优化器续训，重新训练需新输出目录。

5. 在固定评估集上预测和评分。训练/开发场景或相同云文件内容进入评估时会报错。

```bash
for mode in global attention conditioned; do
  python local/category_v10/run.py predict \
    --cache local_outputs/v10_features/eval/features.npz \
    --head local_outputs/v10_models/$mode/head.pt \
    --threads 10 --device cpu \
    --output local_outputs/v10_predictions/$mode || break
done
```

每个结果目录包含：无 GT 的 `predictions.json`、单独评分的 `results.json`、输入/模型/代码来源 `config.json`、冻结标记 `FROZEN.json`。评分报告 Top1、Top5、词表内类别平均准确率、词表外目标数，并给出同一原视图冻结全局分类基线。这个基线不是 v8.3 的双采样局部融合，后者仍使用原入口单独比较。正式消融需要多个固定随机种子与相同训练预算，不能只报告最高的一次。

## 与 Nr3D 定位流程衔接的边界

本次完成到监督分类预测/评测，**尚未启动或改写完整 Nr3D 定位流程**。新输出是未经校准的 softmax 分数，不是 v8.3 余弦分数；不能直接套用 v9 的 `0.20 / 0.03` 门限，也不能伪装为 v8.3 冻结结果。下一阶段应在内部开发场景选择门限，再通过独立适配入口接入 v9，评估目标/参照物召回、Hard/View-dependent 子集和最终定位准确率。最终定位结果还受空间关系、参照物绑定和视角理解影响，不承诺达到 68.39%。

## 已完成验证

```bash
PYTHONPATH=local .venv/bin/python -m unittest category_v10.test_experiment -v
```

7 项 CPU 测试覆盖：真实梯度和损失下降、两种注意力相同参数量及不同查询、Top-5 外类别可被修正、物理房间/验证集隔离、实例去重/词表外分母、原始扫描转换、特征缓存续跑、训练保存与重载一致性、GT 不进入预测文件、损坏特征拒绝、冻结结果保护以及原子下载/磁盘空间保护。CPU 测试中的 CUDA 编码部分使用明确的测试替身；真实分类精度和论文新颖性尚未验证。

## 服务器全量入口

`pipeline.py` 将最小扫描下载、转换、特征提取、三方法三随机种子训练及分类评分串联。只有人类明确同意 ScanNet 条款后才能传 `--terms-confirmed`。默认种子为 42/43/44，每个方法 40 epoch；所有模型都先在内部开发集选模，再统一在官方验证场景上评分。下载保留至少 5 GiB 可用空间，空间不足会失败并记录状态，不会清理已有数据。原始扫描保留以便追溯。

2026-10-17 用户已明确同意条款并授权服务器下载和训练。本次服务器为 RTX 5090 D，已有算子原先不支持此 GPU，因此为 v10 单独编译 `sm_120` 算子，放在实验目录的 `cuda_ops/`，通过 `PYTHONPATH` 选择；不替换原环境内的 CUDA 扩展。

```bash
cd /root/paper/reproductions/SeeGround
PYTHONPATH=/root/autodl-tmp/seeground-v10/cuda_ops \
  .venv-v82/bin/python -u local/category_v10/pipeline.py \
  --work-root /root/autodl-tmp/seeground-v10 \
  --terms-confirmed --seeds 42 43 44
```

同一 work-root 用文件锁阻止并行写入；代码、数据清单或训练设置改变必须使用新目录。下载成功和转换完成可复用，特征按实例续跑；中断的训练模型目录需要保留并改用新目录，不能把已有 `head.pt` 误认成完整 40 epoch。已冻结的预测保持只读。

已启动的后台作业不要重复运行上面的启动命令。可以查看：

```bash
cat /root/autodl-tmp/seeground-v10/status.json
tail -f /root/autodl-tmp/seeground-v10/pipeline.log
```

最终结果为 `/root/autodl-tmp/seeground-v10/comparison.json`；数据、特征、模型和逐物体预测分别放在 `scenes/`、`features/`、`models/seed*/` 和 `predictions/seed*/`。`launch.json` 保存启动命令和 PID，`pipeline_config.json` 保存代码哈希与数据范围。SSH 断开不影响已启动的后台进程；关机则会中断。

## 本地外置硬盘下载

在本地仓库运行 `.venv/bin/python -u local/download_scannet_external.py`，默认保存到 `/Volumes/KESU/ScanNet_v10/raw/场景ID/`。需要 `curl`，无需 GPU。沿用已确认的 ScanNet 条款授权；默认四个场景并行、从清单末尾开始，以减少与服务器从前往后下载的即时重叠。它仍会下载完整训练清单，不会自动获知服务器已完成的场景。

已完成的文件会跳过；未完成文件使用 `.part` 和 HTTP Range 续传，源站不支持续传时重新下载该文件。下载成功才改为最终文件名。外置盘未挂载会拒绝运行，并保留至少 5 GiB 空间预算。`--limit 1` 可试下一个场景；`--workers 4` 控制并发，`--output` 可更换外置盘目录。同一目录有文件锁，失败以非零状态退出并打印场景，重新运行可补下。本脚本只下载原始文件，不转换、不训练、不上传。

上传时使用服务器临时目录暂存，仅将完整文件放入 `/root/autodl-tmp/seeground-v10/raw/` 的对应场景目录；合并前停止服务器下载任务，避免同名文件并行写入。
