本轮在相同四视图上适配CLIP ViT-B/16真实最后一块，比较frozen_head、tail_head和tail_anchor。协议见research/20261019_classifier_innovation/图像编码器适配r11.md。

顺序：preflight → cache train → cache eval → train-select → cache dev → evaluate → audit。
每步使用classifier_clip_tail_r11.py及相同--output/--reference/--r4/--r8/--clip/--design参数。cache另外指定--split。
必须用原float32 token，缓存约79GB。r11共享盘初版因FUSE写入等待停止，全部记录保留；r11b由队列父进程持有memfd内存缓存直到审计完成，模型/源码/预测/哈希持久保存。临时token不保证父进程退出后存在，届时须按固定数据与编码器重建。不能静默换精度。遇到超时先检查实际PID及状态，不重复启动。
原28730/404/3413物理房间协议不变；选择封存前禁止生成本轮dev token或计算其新分数。
运行local/test_clip_tail_r11.py执行5项行为检查。服务器preflight另核验原完整CLIP桥接、真实梯度、参数隔离、显存及20步学习。
原参照若被选中不能算新编码器收益；True与Exact分开；不编辑旧轮次缓存、模型或协议。
