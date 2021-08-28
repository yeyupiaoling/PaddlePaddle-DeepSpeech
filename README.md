# DeepSpeech2 语音识别

![License](https://img.shields.io/badge/license-Apache%202-red.svg)
![python version](https://img.shields.io/badge/python-3.7+-orange.svg)
![support os](https://img.shields.io/badge/os-linux-yellow.svg)
![GitHub Repo stars](https://img.shields.io/github/stars/yeyupiaoling/PaddlePaddle-DeepSpeech?style=social)

本项目是基于PaddlePaddle的[DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech) 项目开发的，做了较大的修改，方便训练中文自定义数据集，同时也方便测试和使用。DeepSpeech2是基于PaddlePaddle实现的端到端自动语音识别（ASR）引擎，其论文为[《Baidu's Deep Speech 2 paper》](http://proceedings.mlr.press/v48/amodei16.pdf) ，本项目同时还支持各种数据增强方法，以适应不同的使用场景。支持在Windows，Linux下训练和预测，支持Nvidia Jetson等开发板推理预测。

本项目使用的环境：
 - Python 3.7
 - PaddlePaddle 2.1.2
 - Windows or Ubuntu

## 模型下载
| 数据集 | 字错率 | 下载地址 |
| :---: | :---: | :---: |
| AISHELL |训练中 | [训练中]() |
| free_st_chinese_mandarin_corpus | 训练中 | [训练中]() |
| thchs_30 | 训练中 | [训练中]() |
| 超大数据集(1600+小时) | 训练中 | [训练中]() |

**说明：** 这里提供的是训练参数，如果要用于预测，还需要执行[导出模型](./docs/export_model.md)

>有问题欢迎提 [issue](https://github.com/yeyupiaoling/PaddlePaddle-DeepSpeech/issues) 交流


## 文档教程

- [快速安装](./docs/install.md)
- [数据准备](./docs/dataset.md)
- [训练模型](./docs/train.md)
- [数据增强](./docs/augment.md)
- [集束搜索解码](./docs/beam_search.md)
- [执行评估](./docs/eval.md)
- [导出模型](./docs/export_model.md)
- 预测
   - [本地模型](./docs/infer.md)
   - [长语音模型](./docs/infer.md)
   - [Web部署模型](./docs/infer.md)


## 预测

我们可以使用这个脚本使用模型进行预测，如果如何还没导出模型，需要执行上面的操作把模型参数导出为预测模型，通过传递音频文件的路径进行识别。默认使用的是`ctc_greedy`贪心解码策略，这也是为了支持Windows用户。
```shell script
python infer_path.py --wav_path=./dataset/test.wav
```

输出结果：
```
-----------  Configuration Arguments -----------
alpha: 1.2
beam_size: 10
beta: 0.35
cutoff_prob: 1.0
cutoff_top_n: 40
decoding_method: ctc_greedy
enable_mkldnn: False
lang_model_path: ./lm/zh_giga.no_cna_cmn.prune01244.klm
mean_std_path: ./dataset/mean_std.npz
model_dir: ./models/infer/
use_gpu: True
use_tensorrt: False
vocab_path: ./dataset/zh_vocab.txt
wav_path: ./dataset/test.wav
------------------------------------------------
I0729 19:48:39.860693 14609 analysis_config.cc:424] use_dlnne_:0
I0729 19:48:39.860721 14609 analysis_config.cc:424] use_dlnne_:0
I0729 19:48:39.860729 14609 analysis_config.cc:424] use_dlnne_:0
I0729 19:48:39.860736 14609 analysis_config.cc:424] use_dlnne_:0
I0729 19:48:39.860746 14609 analysis_config.cc:424] use_dlnne_:0
I0729 19:48:39.860757 14609 analysis_config.cc:424] use_dlnne_:0
消耗时间：260, 识别结果: 近几年不但我用书给女儿儿压岁也劝说亲朋不要给女儿压岁钱而改送压岁书, 得分: 94
```

## TODO

1. 训练三个公开数据集的模型。
2. 增加音频数据集合成程序和变声程序。


## 相关项目
 - 基于PaddlePaddle实现的声纹识别：[VoiceprintRecognition-PaddlePaddle](https://github.com/yeyupiaoling/VoiceprintRecognition-PaddlePaddle)
 - 基于PaddlePaddle动态图实现的语音识别：[PPASR](https://github.com/yeyupiaoling/PPASR)
 - 基于Pytorch实现的语音识别：[MASR](https://github.com/yeyupiaoling/MASR)
