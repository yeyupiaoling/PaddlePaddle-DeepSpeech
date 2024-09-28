# 评估

执行下面这个脚本对模型进行评估，通过字符错误率来评价模型的性能。
```shell
python eval.py --resume_model=./models/epoch_100/model.pdparams
```

输出结果：
```
2024-09-28 11:59:16.676 | INFO     | utils.utils:print_arguments:11 - ----------- 额外配置参数 -----------
2024-09-28 11:59:16.676 | INFO     | utils.utils:print_arguments:13 - alpha: 1.2
2024-09-28 11:59:16.677 | INFO     | utils.utils:print_arguments:13 - batch_size: 32
2024-09-28 11:59:16.677 | INFO     | utils.utils:print_arguments:13 - beam_size: 300
2024-09-28 11:59:16.677 | INFO     | utils.utils:print_arguments:13 - beta: 0.35
2024-09-28 11:59:16.677 | INFO     | utils.utils:print_arguments:13 - cutoff_prob: 0.99
2024-09-28 11:59:16.677 | INFO     | utils.utils:print_arguments:13 - cutoff_top_n: 40
2024-09-28 11:59:16.677 | INFO     | utils.utils:print_arguments:13 - decoder: ctc_greedy
2024-09-28 11:59:16.677 | INFO     | utils.utils:print_arguments:13 - lang_model_path: ./lm/zh_giga.no_cna_cmn.prune01244.klm
2024-09-28 11:59:16.677 | INFO     | utils.utils:print_arguments:13 - max_duration: 20.0
2024-09-28 11:59:16.677 | INFO     | utils.utils:print_arguments:13 - mean_istd_path: ./dataset/mean_istd.json
2024-09-28 11:59:16.677 | INFO     | utils.utils:print_arguments:13 - metrics_type: cer
2024-09-28 11:59:16.677 | INFO     | utils.utils:print_arguments:13 - min_duration: 0.5
2024-09-28 11:59:16.677 | INFO     | utils.utils:print_arguments:13 - num_proc_bsearch: 8
2024-09-28 11:59:16.677 | INFO     | utils.utils:print_arguments:13 - num_rnn_layers: 3
2024-09-28 11:59:16.678 | INFO     | utils.utils:print_arguments:13 - pretrained_model: ./models/epoch_15/
2024-09-28 11:59:16.678 | INFO     | utils.utils:print_arguments:13 - rnn_layer_size: 1024
2024-09-28 11:59:16.678 | INFO     | utils.utils:print_arguments:13 - test_manifest: ./dataset/manifest.test
2024-09-28 11:59:16.678 | INFO     | utils.utils:print_arguments:13 - use_gpu: True
2024-09-28 11:59:16.678 | INFO     | utils.utils:print_arguments:13 - vocab_path: ./dataset/vocabulary.txt
2024-09-28 11:59:16.678 | INFO     | utils.utils:print_arguments:14 - ------------------------------------------------
W0928 11:59:17.548969 22724 gpu_resources.cc:119] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 12.4, Runtime API Version: 11.7
W0928 11:59:17.550941 22724 gpu_resources.cc:164] device: 0, cuDNN Version: 8.4.
2024-09-28 11:59:18.096 | INFO     | utils.checkpoint:load_pretrained:40 - 成功加载预训练模型：./models/epoch_100/model.pdparams
2024-09-28 11:59:20.380 | INFO     | __main__:evaluate:95 - 预测结果为：在政府目标的引导下
2024-09-28 11:59:20.380 | INFO     | __main__:evaluate:96 - 实际标签为：在政府目标的引导下
2024-09-28 11:59:20.380 | INFO     | __main__:evaluate:97 - 这条数据的cer：0.0，当前cer：0.0
  0%|          | 0/9 [00:00<?, ?it/s]
```
