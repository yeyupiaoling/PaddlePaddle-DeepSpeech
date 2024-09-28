# 导出模型

训练保存的或者下载作者提供的模型都是模型参数，我们要将它导出为预测模型，这样可以直接使用模型，不再需要模型结构代码，同时使用Inference接口可以加速预测。
```shell
python export_model.py --resume_model=./models/epoch_100/model.pdparams
```

输出结果：
```
2024-09-28 12:01:11.360 | INFO     | utils.utils:print_arguments:11 - ----------- 额外配置参数 -----------
2024-09-28 12:01:11.360 | INFO     | utils.utils:print_arguments:13 - mean_istd_path: ./dataset/mean_istd.json
2024-09-28 12:01:11.360 | INFO     | utils.utils:print_arguments:13 - num_rnn_layers: 3
2024-09-28 12:01:11.361 | INFO     | utils.utils:print_arguments:13 - pretrained_model: ./models/epoch_15/
2024-09-28 12:01:11.361 | INFO     | utils.utils:print_arguments:13 - rnn_layer_size: 1024
2024-09-28 12:01:11.361 | INFO     | utils.utils:print_arguments:13 - save_model_path: ./models/infer/
2024-09-28 12:01:11.361 | INFO     | utils.utils:print_arguments:13 - vocab_path: ./dataset/vocabulary.txt
2024-09-28 12:01:11.361 | INFO     | utils.utils:print_arguments:14 - ------------------------------------------------
W0928 12:01:12.347131 35992 gpu_resources.cc:119] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 12.4, Runtime API Version: 11.7
W0928 12:01:12.350131 35992 gpu_resources.cc:164] device: 0, cuDNN Version: 8.4.
2024-09-28 12:01:13.202 | INFO     | utils.checkpoint:load_pretrained:40 - 成功加载预训练模型：./models/epoch_15/model.pdparams
2024-09-28 12:01:14.121 | INFO     | __main__:<module>:40 - 预测模型已保存：./models/infer/model
```