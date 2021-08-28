# 评估

这里我也提示几点，在预测中可以提升性能的几个参数，预测包括评估，推理，部署等等一系列使用到模型预测音频的程序。解码方法，通过`decoding_method`方法选择不同的解码方法，支持`ctc_beam_search`集束搜索和`ctc_greedy`贪心解码策略两种，Windows只支持`ctc_greedy`贪心解码策略，其中`ctc_beam_search`集束搜索效果是最好的，但是速度就比较慢，这个可以通过`beam_size`参数设置集束搜索的宽度，以提高执行速度，范围[5, 500]，越大准确率就越高，同时执行速度就越慢。如果对准确率没有太严格的要求，可以考虑直接使用`ctc_greedy`贪心解码策略，其实准确率也低不了多少，而且Windows，Linux都支持，省去编译`ctc_decoders`的麻烦。

 - 如果是如果是Windows用户可以直接掉过该步骤。使用`ctc_beam_search`集束搜索方法有两个参数通过调整可以达到最好的准确率。该程序主要是为了寻找`ctc_beam_search`集束搜索方法中最优的`alpha`，`beta`参数，以获得最好的识别准确率。默认的参数是比较好的，如果如果对准确率没有太严格的要求可以直接跳过这一步。
```shell script
python tools/tune.py --model_path=./models/step_final/
```

 - 我们可以使用这个脚本对模型进行评估，通过字符错误率来评价模型的性能。默认使用的是`ctc_beam_search`集束搜索，如果是Windows用户，可以指定解码方法为`ctc_greedy`贪心解码策略。
```shell
python eval.py --model_path=./models/step_final/ --decoding_method=ctc_beam_search
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
decoding_method: ctc_beam_search
error_rate_type: cer
lang_model_path: ./lm/zh_giga.no_cna_cmn.prune01244.klm
mean_std_path: ./dataset/mean_std.npz
model_path: models/step_final/
num_conv_layers: 2
num_proc_bsearch: 8
num_rnn_layers: 3
rnn_layer_size: 2048
share_rnn_weights: False
test_manifest: ./dataset/manifest.test
use_gpu: True
use_gru: True
vocab_path: ./dataset/zh_vocab.txt
------------------------------------------------
W0318 16:38:49.200599 19032 device_context.cc:252] Please NOTE: device: 0, CUDA Capability: 75, Driver API Version: 11.0, Runtime API Version: 10.0
W0318 16:38:49.242089 19032 device_context.cc:260] device: 0, cuDNN Version: 7.6.
成功加载了预训练模型：models/epoch_19/
初始化解码器...
======================================================================
language model: is_character_based = 1, max_order = 5, dict_size = 0
======================================================================
初始化解码器完成!
[INFO 2021-03-18 16:38:51,442 model.py:523] begin to initialize the external scorer for decoding
[INFO 2021-03-18 16:38:53,688 model.py:531] language model: is_character_based = 1, max_order = 5, dict_size = 0
[INFO 2021-03-18 16:38:53,689 model.py:532] end initializing scorer
[INFO 2021-03-18 16:38:53,689 eval.py:83] 开始评估 ...
错误率：[cer] (64/284) = 0.077040
错误率：[cer] (128/284) = 0.062989
错误率：[cer] (192/284) = 0.055674
错误率：[cer] (256/284) = 0.054918
错误率：[cer] (284/284) = 0.055882
消耗时间：44526ms, 总错误率：[cer] (284/284) = 0.055882
[INFO 2021-03-18 16:39:38,215 eval.py:117] 完成评估！
```
