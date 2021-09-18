# 评估

执行下面这个脚本对模型进行评估，通过字符错误率来评价模型的性能。
```shell
python eval.py --resume_model=./models/param/50.pdparams
```

输出结果：
```
-----------  Configuration Arguments -----------
alpha: 1.2
batch_size: 64
beam_size: 10
beta: 0.35
cutoff_prob: 1.0
cutoff_top_n: 40
decoding_method: ctc_greedy
error_rate_type: cer
lang_model_path: ./lm/zh_giga.no_cna_cmn.prune01244.klm
mean_std_path: ./dataset/mean_std.npz
resume_model: ./models/param/50.pdparams
num_conv_layers: 2
num_proc_bsearch: 8
num_rnn_layers: 3
rnn_layer_size: 1024
test_manifest: ./dataset/manifest.test
use_gpu: True
vocab_path: ./dataset/zh_vocab.txt
------------------------------------------------
W0318 16:38:49.200599 19032 device_context.cc:252] Please NOTE: device: 0, CUDA Capability: 75, Driver API Version: 11.0, Runtime API Version: 10.0
W0318 16:38:49.242089 19032 device_context.cc:260] device: 0, cuDNN Version: 7.6.
[INFO 2021-03-18 16:38:53,689 eval.py:83] 开始评估 ...
错误率：[cer] (64/284) = 0.077040
错误率：[cer] (128/284) = 0.062989
错误率：[cer] (192/284) = 0.055674
错误率：[cer] (256/284) = 0.054918
错误率：[cer] (284/284) = 0.055882
消耗时间：44526ms, 总错误率：[cer] (284/284) = 0.055882
[INFO 2021-03-18 16:39:38,215 eval.py:117] 完成评估！
```
