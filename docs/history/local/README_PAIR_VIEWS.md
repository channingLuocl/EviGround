# 类别对视图证据门控：训练侧试验

设计见工作区 `research/20261019_classifier_innovation/训练侧试验r1.md`。

本轮只使用官方训练清单中的28730个词表内对象。565个物理房间三外折，外折训练部分内三折交叉拟合；原冻结PointNeXt的预训练暴露未排除。旧3413对象与404对象均不参与本轮训练、选模和评分。本轮结果不能直接与包含OOV的旧开发指标比较。

模型：原冻结3D、普通固定融合、长尾校正／Balanced Softmax固定融合、普通置信度门控、普通多视图质量门控、类别对支持门控。三个门控均289参数、相同训练轮数，最终保留完整197类分数。

## 执行顺序

在项目根目录、具有GPU的原环境运行。下面的路径为本轮服务器实际目录。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/root/autodl-tmp/venvs/classifier-heads/bin/python local/cache_pair_views.py \
  --reference /root/autodl-tmp/classifier_scan_domain_20261019 \
  --output /root/autodl-tmp/classifier_pair_views_20261019_r1/views \
  --clip /root/.cache/huggingface/hub/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3/pytorch_model.bin
```

新建实验目录后，以相同参数先执行 `train_pair_views.py experts`，四视图缓存完成后执行 `train_pair_views.py gates`：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/root/autodl-tmp/venvs/classifier-heads/bin/python local/train_pair_views.py experts \
  --reference /root/autodl-tmp/classifier_scan_domain_20261019 \
  --output /root/autodl-tmp/classifier_pair_views_20261019_r1/experiment \
  --views /root/autodl-tmp/classifier_pair_views_20261019_r1/views \
  --design /root/autodl-tmp/classifier_pair_views_20261019_r1/design.md
```

已有比较输出时不可重复执行gates；程序会拒绝覆盖。修改设计或训练／缓存源代码后使用新目录。专家阶段可复用同一固定配置中已完成的小头；有问题的中间产物保留供核查。

最终执行 `audit_pair_views.py --reference ... --output ... --views ...`，通过后执行 `summarize_pair_views.py --output ...`。审计核对文件哈希、房间隔离、72个专家、OOF选择网格和全部外折分数；NumPy独立计算门控特征及网络前向，重算全部指标和房间配对区间。

## 主要产物

- `views/COMPLETE.json`：四视图缓存哈希与对旧均值特征的桥接误差。
- `experiment/config.json`、`folds.json`、`source_snapshot/`：冻结设计、房间分折与实际源码。
- `EXPERTS_COMPLETE.json`、`RESULTS_COMPLETE.json`：专家和比较输出的哈希。
- `audited_summary.json`、`review_outcomes.npz`：审计结果、逐对象三种子正确性与Top-5。
- `analysis/report.md`、`table.tsv`、`decision.json`：可阅读报告、完整结果表与预定继续投入门槛。

正确数保留每个种子的整数；先合并三外折再计算全体Top-1／Top-5和出现类别Macro。种子标准差与物理房间配对区间分别报告。不得把分类结果称为Nr3D定位准确率或零样本成绩。

## r1实际结果

72个专家与27个门控均完成。独立重算与本地归档核验通过；六种方法三种子均为28106/28730，Top-1为97.83%、Macro为93.42%。三个外折固定融合均选择RGB权重0，主候选E的所有外折权重也为0，净纠错为0，未达到预定门槛。冻结3D在训练侧对象上接近饱和，这轮无法区分候选证据机制，不应作为新场景泛化结果。

详细解释见工作区`research/20261019_classifier_innovation/训练侧试验r1结果.md`。本地归档位于`local_diagnostics/classifier_pair_views_20261019_r1/`，保留首次审计数值容差失败与后处理修复记录，模型和预测未修改。

## 软件检查

在`local/`执行 `../.venv/bin/python -m unittest test_pair_views -v`。另已用临时合成数据验证完整的72专家、27门控、独立审计及报告流程；合成准确率不作为研究结果。

## r2：训练—验证—历史测试

r2设计见工作区`research/20261019_classifier_innovation/训练验证测试r2.md`。28730对象仅作梯度训练；404对象用于三种子共用的融合配置选择；配置、模型及验证结果封存后才读取3413对象特征，报告历史测试对比。两批评估对象此前均已揭示，不能称为独立盲测。

顺序为`cache_pair_evaluation.py --split eval`生成验证四视图、`run_pair_train_test.py train-select`训练6个最终RGB头和9个门控并封存、`cache_pair_evaluation.py --split dev --selection-root ...`生成历史测试四视图、`run_pair_train_test.py test`评分、`audit_pair_train_test.py`独立复算。训练程序使用`--reference`、`--r1`、`--output`、`--design`；审计使用`--reference`和`--output`。实际输出位于服务器`/root/autodl-tmp/classifier_pair_views_20261019_r2`。

门控主候选E仅在404验证对象上严格增加True正确数且Macro不降时入选，否则回退固定融合B。3413对象不能用于本轮重新选型。True按既有同义词/复数归一化，Exact Top-1为字符串精确匹配；表格两列分别报告，OOV对象保留。

软件检查：在项目根目录执行`PYTHONPATH=local .venv/bin/python -m unittest local/test_pair_train_test.py local/test_pair_views.py`，共10项。审计还会核对完整验证搜索网格、独立NumPy分类头/门控预测、训练OOF专家来源、房间隔离以及房间配对bootstrap区间。

r2已完成并通过服务器独立审计、本地130文件哈希与全部预测/指标/bootstrap复核。3413对象上B的三种子True正确数为2333/2336/2332，平均True68.38%、Exact Top-1 62.09%、Top-5 87.45%、Macro25.33%；原3D为66.28%/60.47%/84.35%/28.51%。E平均True66.32%，未胜过固定融合。B提高总体准确率但降低Macro，不能称为全面提升；完整八列表在`local_diagnostics/classifier_pair_views_20261019_r2/review/table.tsv`，解释见工作区`research/20261019_classifier_innovation/训练验证测试r2结果.md`。
