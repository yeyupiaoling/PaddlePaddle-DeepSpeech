# 语音识别

![License](https://img.shields.io/badge/license-Apache%202-red.svg)
![python version](https://img.shields.io/badge/python-3.7+-orange.svg)
![support os](https://img.shields.io/badge/os-linux-yellow.svg)
![GitHub Repo stars](https://img.shields.io/github/stars/yeyupiaoling/PaddlePaddle-DeepSpeech?style=social)

本项目是基于PaddlePaddle的[DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech) 项目开发的，做了较大的修改，方便训练中文自定义数据集，同时也方便测试和使用。DeepSpeech2是基于PaddlePaddle实现的端到端自动语音识别（ASR）引擎，其论文为[《Baidu's Deep Speech 2 paper》](http://proceedings.mlr.press/v48/amodei16.pdf) ，本项目同时还支持各种数据增强方法，以适应不同的使用场景。支持在Windows，Linux下训练和预测，支持Nvidia Jetson开发板预测。

本项目使用的环境：
 - Python 3.7
 - PaddlePaddle 2.1.1
 - Windows or Ubuntu


## 目录

- [搭建本地环境](#搭建本地环境)
- [搭建Docker环境](docs/install.md)
- [Nvidia Jetson预测环境搭建](#nvidia-jetson预测环境搭建)
- [数据准备](#数据准备)
- [训练模型](#训练模型)
- [评估](#评估)
- [本地预测](#本地预测)
- [Web部署](#web部署)
- [模型下载](#模型下载)
- [相关项目](#相关项目)


## 搭建本地环境

本人用的就是本地环境和使用Anaconda，并创建了Python3.7的虚拟环境，建议读者也本地环境，方便交流，出现安装问题，随时提[issue](https://github.com/yeyupiaoling/PaddlePaddle-DeepSpeech/issues) ，如果想使用docker，请查看[搭建Docker环境](docs/install.md)。

 - 首先安装的是PaddlePaddle 2.1.1的GPU版本，如果已经安装过了，请跳过。
```shell
conda install paddlepaddle-gpu==2.1.1 cudatoolkit=10.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
```

 - 安装其他依赖库。
```shell
python -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

**注意：** 如果出现LLVM版本错误，解决办法[LLVM版本错误](docs/faq.md)。

## Nvidia Jetson预测环境搭建

1. 安装PaddlePaddle的Inference预测库。
```shell
wget https://paddle-inference-lib.bj.bcebos.com/2.1.1-nv-jetson-jetpack4.4-all/paddlepaddle_gpu-2.1.1-cp36-cp36m-linux_aarch64.whl
pip3 install paddlepaddle_gpu-2.1.1-cp36-cp36m-linux_aarch64.whl
```

2. 安装scikit-learn依赖库。
```shell
git clone git://github.com/scikit-learn/scikit-learn.git
cd scikit-learn
pip3 install cython
git checkout 0.24.2
pip3 install --verbose --no-build-isolation --editable .
```

3. 安装其他依赖库。
```shell
pip3 install -r requirements.txt
```

4. 在Nvidia Jetson开发板上预测跟[本地预测](#本地预测) 一样，请查看这部分操作。

## 数据准备

1. 在`data`目录下是公开数据集的下载和制作训练数据列表和词汇表的，本项目提供了下载公开的中文普通话语音数据集，分别是Aishell，Free ST-Chinese-Mandarin-Corpus，THCHS-30 这三个数据集，总大小超过28G。下载这三个数据只需要执行一下代码即可，当然如果想快速训练，也可以只下载其中一个。
```shell script
cd download_data/
python aishell.py
python free_st_chinese_mandarin_corpus.py
python thchs_30.py
```

 - 如果开发者有自己的数据集，可以使用自己的数据集进行训练，当然也可以跟上面下载的数据集一起训练。自定义的语音数据需要符合以下格式，另外对于音频的采样率，本项目默认使用的是16000Hz，在`create_manifest.py`中也提供了统一音频数据的采样率转换为16000Hz，只要`is_change_frame_rate`参数设置为True就可以。
    1. 语音文件需要放在`DeepSpeech/dataset/audio/`目录下，例如我们有个`wav`的文件夹，里面都是语音文件，我们就把这个文件存放在`DeepSpeech/dataset/audio/`。
    2. 然后把数据列表文件存在`DeepSpeech/dataset/annotation/`目录下，程序会遍历这个文件下的所有数据列表文件。例如这个文件下存放一个`my_audio.txt`，它的内容格式如下。每一行数据包含该语音文件的相对路径和该语音文件对应的中文文本，要注意的是该中文文本只能包含纯中文，不能包含标点符号、阿拉伯数字以及英文字母。
```shell script
dataset/audio/wav/0175/H0175A0171.wav 我需要把空调温度调到二十度
dataset/audio/wav/0175/H0175A0377.wav 出彩中国人
dataset/audio/wav/0175/H0175A0470.wav 据克而瑞研究中心监测
dataset/audio/wav/0175/H0175A0180.wav 把温度加大到十八
```

 - 然后执行下面的数据集处理脚本，这个是把我们的数据集生成三个JSON格式的数据列表，分别是`manifest.dev、manifest.test、manifest.train`。然后计算均值和标准差用于归一化，脚本随机采样2000个的语音频谱特征的均值和标准差，并将结果保存在`mean_std.npz`中。建立词表。最后建立词表，把所有出现的字符都存放子在`zh_vocab.txt`文件中，一行一个字符。以上生成的文件都存放在`DeepSpeech/dataset/`目录下。
```shell script
# 生成数据列表
python tools/create_manifest.py
# 计算均值和标准差
python tools/compute_mean_std.py
# 构建词汇表
python tools/build_vocab.py
```

在生成数据列表的是要注意，该程序除了生成训练数据列表，还提供对音频帧率的转换和生成噪声数据列表，开发者有些自定义的数据集音频的采样率不是16000Hz的，所以提供了`change_audio_rate()`函数，帮助开发者把指定的数据集的音频采样率转换为16000Hz。提供的生成噪声数据列表`create_noise`函数，前提是要有噪声数据集，使用噪声数据在训练中实现数据增强。
```python
if __name__ == '__main__':
    # 生成噪声的数据列表
    create_noise()
    # 生成训练数据列表
    main()
```


## 训练模型

 - 执行训练脚本，开始训练语音识别模型， 每训练一轮保存一次模型，模型保存在`DeepSpeech/models/`目录下，默认会使用噪声音频作为数据增强一起训练的，如果没有这些音频，自动忽略噪声数据增强。关于数据增强，请查看[数据增强](docs/faq.md)部分。如果没有关闭测试，在每一轮训练结果之后，都会执行一次测试，为了提高测试的速度，测试使用的是最优解码路径解码，这个解码方式结果没有集束搜索的方法准确率高，所以测试的输出的准确率可以理解为保底的准确率。
```shell script
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

 - 在训练过程中，程序会使用VisualDL记录训练结果，可以通过以下的命令启动VisualDL。
```shell
visualdl --logdir=log --host=0.0.0.0
```

 - 然后再浏览器上访问`http://localhost:8040`可以查看结果显示，如下。

![Learning rate](https://img-blog.csdnimg.cn/20210318165719805.png)
![Test Cer](https://s3.ax1x.com/2021/03/01/6PJaZV.jpg)
![Train Loss](https://s3.ax1x.com/2021/03/01/6PJNq0.jpg)


## 语言模型
如果是Windows环境，请忽略。语言模型是集束搜索解码方法使用的，集束搜索解码方法只能在Ubuntu下编译，Windows用户只能使用贪心策略解码方法。下载语言模型并放在lm目录下，下面下载的小语言模型，如何有足够大性能的机器，可以下载70G的超大语言模型，点击下载[Mandarin LM Large](https://deepspeech.bj.bcebos.com/zh_lm/zhidao_giga.klm) ，这个模型会大超多。
```shell script
cd DeepSpeech/
mkdir lm
cd lm
wget https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm
```

## 评估

这里我也提示几点，在预测中可以提升性能的几个参数，预测包括评估，推理，部署等等一系列使用到模型预测音频的程序。解码方法，通过`decoding_method`方法选择不同的解码方法，支持`ctc_beam_search`集束搜索和`ctc_greedy`贪心解码策略两种，Windows只支持`ctc_greedy`贪心解码策略，其中`ctc_beam_search`集束搜索效果是最好的，但是速度就比较慢，这个可以通过`beam_size`参数设置集束搜索的宽度，以提高执行速度，范围[5, 500]，越大准确率就越高，同时执行速度就越慢。如果对准确率没有太严格的要求，可以考虑直接使用`ctc_greedy`贪心解码策略，其实准确率也低不了多少，而且Windows，Linux都支持，省去编译`ctc_decoders`的麻烦。

 - 如果需要使用`ctc_beam_search`集束搜索，需要编译`ctc_decoders`库，该编译只支持Ubuntu，其他Linux版本没测试过。
```shell
cd decoders
sh setup.sh
```

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

## 导出模型

训练保存的是模型参数，我们要将它到处为预测模型，这样可以直接使用模型，不再需要模型结构代码，同时使用Inference接口可以加速预测，在一些设备也可以使用TensorRT加速。
```shell
python export_model.py --pretrained_model=./models/step_final/
```

输出结果：
```
成功加载了预训练模型：./models/step_final/
-----------  Configuration Arguments -----------
mean_std_path: ./dataset/mean_std.npz
num_conv_layers: 2
num_rnn_layers: 3
pretrained_model: ./models/step_final/
rnn_layer_size: 2048
save_model_path: ./models/infer/
share_rnn_weights: False
use_gpu: True
use_gru: True
vocab_path: ./dataset/zh_vocab.txt
wav_path: ./dataset/test.wav
------------------------------------------------
成功导出模型，模型保存在：./models/infer/
```

## 本地预测

我们可以使用这个脚本使用模型进行预测，通过传递音频文件的路径进行识别。默认使用的是`ctc_greedy`贪心解码策略，这也是为了支持Windows用户。
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


## 长语音预测

通过参数is_long_audio可以指定使用长语音识别方式，这种方式通过VAD分割音频，再对短音频进行识别，拼接结果，最终得到长语音识别结果。
```shell script
python infer_path.py --wav_path=./dataset/test_vad.wav --is_long_audio=True
```


## Web部署
 - 在服务器执行下面命令通过创建一个Web服务，通过提供HTTP接口来实现语音识别，默认使用的是`ctc_greedy`贪心解码策略。启动服务之后，在浏览器上访问`http://localhost:5000`
```shell script
python infer_server.py
```

![录音测试页面](https://img-blog.csdnimg.cn/20210402091159951.png)

## 模型下载
| 数据集 | 字错率 | 下载地址 |
| :---: | :---: | :---: |
| AISHELL |训练中 | [训练中]() |
| free_st_chinese_mandarin_corpus | 训练中 | [训练中]() |
| thchs_30 | 训练中 | [训练中]() |
| 自收集(1600+小时) | 训练中 | [训练中]() |

**说明：** 这里提供的是训练参数，如果要用于预测，还需要执行[导出模型](#导出模型)

>有问题欢迎提 [issue](https://github.com/yeyupiaoling/PaddlePaddle-DeepSpeech/issues) 交流

## TODO

1. 数据集读取修改为读取修改为`paddle.io.DataLoader`接口。
2. 训练三个公开数据集的模型。
3. 增加音频数据集合成程序和变声程序。


## 相关项目
 - 基于PaddlePaddle实现的声纹识别：[VoiceprintRecognition-PaddlePaddle](https://github.com/yeyupiaoling/VoiceprintRecognition-PaddlePaddle)
 - 基于PaddlePaddle动态图实现的语音识别：[PPASR](https://github.com/yeyupiaoling/PPASR)
 - 基于Pytorch实现的语音识别：[MASR](https://github.com/yeyupiaoling/MASR)
