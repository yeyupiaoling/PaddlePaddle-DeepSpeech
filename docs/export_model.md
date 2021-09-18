# 导出模型

训练保存的或者下载作者提供的模型都是模型参数，我们要将它导出为预测模型，这样可以直接使用模型，不再需要模型结构代码，同时使用Inference接口可以加速预测。
```shell
python export_model.py --resume_model=./models/param/50.pdparams
```

输出结果：
```
成功加载了预训练模型：./models/param/50.pdparams
-----------  Configuration Arguments -----------
mean_std_path: ./dataset/mean_std.npz
num_conv_layers: 2
num_rnn_layers: 3
rnn_layer_size: 1024
resume_model: ./models/param/50.pdparams
save_model_path: ./models/infer/
use_gpu: True
vocab_path: ./dataset/zh_vocab.txt
------------------------------------------------
成功导出模型，模型保存在：./models/infer/
```