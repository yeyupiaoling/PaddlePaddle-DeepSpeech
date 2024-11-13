# DeepSpeech2 语音识别

![License](https://img.shields.io/badge/license-Apache%202-red.svg)
![python version](https://img.shields.io/badge/python-3.7+-orange.svg)
![support os](https://img.shields.io/badge/os-linux-yellow.svg)
![GitHub Repo stars](https://img.shields.io/github/stars/yeyupiaoling/PaddlePaddle-DeepSpeech?style=social)

DeepSpeech2是基于PaddlePaddle实现的端到端自动语音识别（ASR）引擎，其论文为[《Baidu's Deep Speech 2 paper》](http://proceedings.mlr.press/v48/amodei16.pdf) ，本项目同时还支持各种数据增强方法，以适应不同的使用场景。支持在Windows，Linux下训练和预测，支持Nvidia Jetson等开发板推理预测，该分支为新版本，如果要使用旧版本，请查看[release/1.1分支](https://github.com/yeyupiaoling/PaddlePaddle-DeepSpeech/tree/release/1.1)。

**动态图版本使用更简单，支持Deepspeech2、Conformer、Squeezeformer模型：[PPASR](https://github.com/yeyupiaoling/PPASR)**

本项目使用的环境：
 - Python 3.11
 - PaddlePaddle 2.6.1
 - Windows or Ubuntu

## 更新记录

 - 2021.11.26: 修改集束解码bug。
 - 2021.11.09: 提供WenetSpeech数据集制作脚本。
 - 2021.09.05: 提供GUI界面识别部署。
 - 2021.09.04: 提供三个公开数据的预训练模型。
 - 2021.08.30: 支持中文数字转阿拉伯数字，具体请看[预测文档](./docs/infer.md)。
 - 2021.08.29: 完成训练代码和预测代码，同时完善相关文档。
 - 2021.08.07: 支持导出预测模型，使用预测模型进行推理。使用webrtcvad工具，实现长语音识别。
 - 2021.08.06: 将项目大部分的代码修改为PaddlePaddle2.0之后的新API。
 - 2024.10.01: 重构项目，抛弃就得fluid接口。

## 模型下载
|                                   数据集                                    | 循环神经网络的数量 | 循环神经网络的大小 | 错误率(贪心解码器) | 错误率(集束搜索解码器) | 下载地址 |
|:------------------------------------------------------------------------:|:---------:|:---------:|:----------:|:------------:|:----:|
|  [AIShell](https://openslr.magicdatatech.com/resources/33) (179小时，普通话)   |     3     |   1024    |  0.14927   |              |      |
| [Librispeech](https://openslr.magicdatatech.com/resources/12) (960小时，英语) |     3     |   1024    |            |              |      |
|            [WenetSpeech](./docs/wenetspeech.md) (10000小时，普通话)            |     3     |   1024    |            |              |      |

1. 中文的错误率为字错率（cer），英语的错误率为词错率（wer）。


## 文档教程

- [快速安装](./docs/install.md)
- [数据准备](./docs/dataset.md)
- [WenetSpeech数据集](./docs/wenetspeech.md)
- [合成语音数据](./docs/generate_audio.md)
- [数据增强](./docs/augment.md)
- [训练模型](./docs/train.md)
- [集束搜索解码](./docs/beam_search.md)
- [执行评估](./docs/eval.md)
- [导出模型](./docs/export_model.md)
- 预测
   - [本地模型](./docs/infer.md)
   - [长语音模型](./docs/infer.md)
   - [Web部署模型](./docs/infer.md)
   - [Nvidia Jetson部署](./docs/nvidia-jetson.md)


## 快速预测

 - 下载作者提供的模型或者训练模型，然后执行[导出模型](./docs/export_model.md)，使用`infer_path.py`预测音频，通过参数`--wav_path`指定需要预测的音频路径，完成语音识别，详情请查看[模型部署](./docs/infer.md)。
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
is_long_audio: False
lang_model_path: ./lm/zh_giga.no_cna_cmn.prune01244.klm
mean_std_path: ./dataset/mean_std.npz
model_dir: ./models/infer/
to_an: True
use_gpu: True
use_tensorrt: False
vocab_path: ./dataset/zh_vocab.txt
wav_path: ./dataset/test.wav
------------------------------------------------
消耗时间：132, 识别结果: 近几年不但我用书给女儿儿压岁也劝说亲朋不要给女儿压岁钱而改送压岁书
```


 - Web部署

![录音测试页面](./docs/images/infer_server.jpg)


 - GUI界面部署

![GUI界面](./docs/images/infer_gui.jpg)

## 打赏作者

<br/>
<div align="center">
<p>打赏一块钱支持一下作者</p>
<img src="https://yeyupiaoling.cn/reward.png" alt="打赏作者" width="400">
</div>

## 相关项目
 - 基于PaddlePaddle实现的声纹识别：[VoiceprintRecognition-PaddlePaddle](https://github.com/yeyupiaoling/VoiceprintRecognition-PaddlePaddle)
 - 基于PaddlePaddle动态图实现的语音识别：[PPASR](https://github.com/yeyupiaoling/PPASR)
 - 基于Pytorch实现的语音识别：[MASR](https://github.com/yeyupiaoling/MASR)
