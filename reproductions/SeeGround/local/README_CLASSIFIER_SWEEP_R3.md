# 多结构分类器与长尾训练 r3

设计：工作区`research/20261019_classifier_innovation/多结构分类器r3.md`。

`classifier_sweep_r3.py train-select`在28730训练对象上运行12种网络/损失×3种子，并拟合12种确定性原型/岭回归参照。冻结PointNeXt与CLIP、渲染与数据清单；404对象选共同融合设置。P优先True且验证Macro不降；M优先Macro且验证True不降。代码、模型、验证网格和选型封存后，`evaluate`才读取3413对象特征，统一报告全部候选。两批评估均已历史揭示，不能作为独立盲测或Nr3D完整定位结论。

```sh
python local/classifier_sweep_r3.py train-select \
  --reference /root/autodl-tmp/classifier_scan_domain_20261019 \
  --r1 /root/autodl-tmp/classifier_pair_views_20261019_r1 \
  --r2 /root/autodl-tmp/classifier_pair_views_20261019_r2 \
  --output /root/autodl-tmp/classifier_architectures_20261019_r3 \
  --design /root/autodl-tmp/classifier_architectures_20261019_r3/design.md
```

使用同一组参数，将`train-select`改为`evaluate`进行冻结评估。随后执行`audit_classifier_sweep_r3.py --output ...`，独立NumPy重建全部网络/解析分类器输出，重算验证选型和全部八列指标、房间bootstrap；训练轮次/参数变化量和哈希一并核对。不覆盖旧r1/r2结果；中断的实验目录须保留调查，不能直接再次训练。

网络包括RGB MLP、3D/RGB残差适配器、拼接MLP和4视图自注意力；损失为CE、Balanced Softmax、逆平方根加权CE及同步Mixup。全部固定40epoch最后一轮，不用3413选择epoch或融合设置。所有解析模型仅训练对象拟合均值、协方差和标签，确定性模型三种子复用同一结果。

在项目根目录运行`PYTHONPATH=local .venv/bin/python -m unittest local/test_classifier_sweep_r3.py`。覆盖初始化、视图顺序、独立NumPy全部架构前向、同步Mixup、长尾损失、解析模型、OOV计分、双目标选型和实际训练更新。

## 已完成结果

36次神经网络训练及12种解析模型完成。验证锁定P为RGB残差CE适配器（w=.6），历史True68.31%、Macro25.62%；M为逆平方根类别加权RGB MLP（w=.6），历史True69.70%、Exact63.45%、Top-588.09%、Macro30.18%。相对原3D，M同时提升True3.43、Macro1.67个百分点；没有据3413成绩改选P/M。普通MLP对照True69.91%、Macro28.90%，保持对照身份。

19项软件检查通过。服务器审计全部168组输出、578行验证网格、独立NumPy网络与解析分类器及训练拟合统计通过；本地146文件哈希、完整网格选型、24组S0/r2/P/M指标和房间bootstrap复核通过。完整结果/解释在工作区`research/20261019_classifier_innovation/多结构分类器r3结果.md`；八列表位于`local_diagnostics/classifier_architectures_20261019_r3/review/table.tsv`。全部网络与其他候选完整分数仍留服务器，轻量包不包括所有训练模型。
