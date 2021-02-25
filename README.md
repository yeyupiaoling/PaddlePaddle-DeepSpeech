# 语音识别

本项目是基于PaddlePaddle的[DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech) 项目开发的，做了较大的修改，方便训练中文自定义数据集，同时也方便测试和使用。DeepSpeech2是基于PaddlePaddle实现的端到端自动语音识别（ASR）引擎，其论文为[《Baidu's Deep Speech 2 paper》](http://proceedings.mlr.press/v48/amodei16.pdf) ，本项目同时还支持各种数据增强方法，以适应不同的使用场景。
本项目使用的环境：
 - Python 3.7
 - PaddlePaddle 1.8.5

## 目录

- [搭建Docker环境](#搭建Docker环境)
- [数据准备](#数据准备)
- [训练模型](#训练模型)
- [评估和预测](#评估和预测)
- [项目部署](#项目部署)

## 搭建Docker环境

 - 请提前安装好显卡驱动，然后执行下面的命令。
```shell script
# 卸载系统原有docker
sudo apt-get remove docker docker-engine docker.io containerd runc
# 更新apt-get源 
sudo apt-get update
# 安装docker的依赖 
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
# 添加Docker的官方GPG密钥：
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
# 验证拥有指纹
sudo apt-key fingerprint 0EBFCD88
# 设置稳定存储库
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
```

 - 安装Docker
```shell script
# 再次更新apt-get源 
sudo apt-get update
# 开始安装docker 
sudo apt-get install docker-ce
# 加载docker 
sudo apt-cache madison docker-ce
# 验证docker是否安装成功
sudo docker run hello-world
```

 - 安装nvidia-docker
```shell script
# 设置stable存储库和GPG密钥
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 更新软件包清单后
sudo apt-get update

# 安装软件包
sudo apt-get install -y nvidia-docker2

# 设置默认运行时后，重新启动Docker守护程序以完成安装：
sudo systemctl restart docker

# 测试
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

 - 拉取PaddlePaddle 1.8.5镜像，因为这个项目需要在PaddlePaddle 1.8版本才可以运行。
```shell script
sudo nvidia-docker pull hub.baidubce.com/paddlepaddle/paddle:1.8.5-gpu-cuda10.0-cudnn7
```

- git clone 本项目源码
```shell script
git clone https://github.com/yeyupiaoling/DeepSpeech.git
```

- 运行PaddlePaddle语音识别镜像，这里设置与主机共同拥有IP和端口号。
```shell script
sudo nvidia-docker run -it --net=host -v $(pwd)/DeepSpeech:/DeepSpeech hub.baidubce.com/paddlepaddle/paddle:1.8.5-gpu-cuda10.0-cudnn7 /bin/bash
```

 - 切换到`/DeepSpeech/`目录下，执行`setup.sh`脚本安装依赖环境，执行前需要去掉`setup.sh`安装依赖库时使用的`sudo`命令，因为在docker中本来就是root环境，等待安装即可。还有将`scipy==1.5.4 resampy==0.2.2`的版本改为`scipy==1.2.1 resampy==0.1.5`。
```shell script
cd DeepSpeech/
sh setup.sh
```

### 搭建本地环境

 - 并不建议使用本地进行训练和预测，但是如果开发者必须使用本地环境，可以执行下面的命令。因为每个电脑的环境不一样，不能保证能够正常使用。首先切换到`DeepSpeech/`根目录下，执行`setup.sh`脚本安装依赖环境，等待安装即可。默认安装的是PaddlePaddle 1.8.5.post107的GPU版本，需要自行安装相关的CUDA和CUDNN。
```shell script
cd DeepSpeech/
sudo sh setup.sh
```

- git clone 本项目源码
```shell script
git clone https://github.com/yeyupiaoling/DeepSpeech.git
```

## 数据准备

1. 在`data`目录下是公开数据集的下载和制作训练数据列表和字典的，本项目提供了下载公开的中文普通话语音数据集，分别是Aishell，Free ST-Chinese-Mandarin-Corpus，THCHS-30 这三个数据集，总大小超过28G。下载这三个数据只需要执行一下代码即可，当然如果想快速训练，也可以只下载其中一个。
```shell script
cd data/
python3 aishell.py
python3 free_st_chinese_mandarin_corpus.py
python3 thchs_30.py
```

 - 如果开发者有自己的数据集，可以使用自己的数据集进行训练，当然也可以跟上面下载的数据集一起训练。自定义的语音数据需要符合一下格式：
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
PYTHONPATH=.:$PYTHONPATH python3 tools/create_manifest.py
# 计算均值和标准差
PYTHONPATH=.:$PYTHONPATH python3 tools/compute_mean_std.py
# 构建字典
PYTHONPATH=.:$PYTHONPATH python3 tools/build_vocab.py
```

在生成数据列表的是要注意，该程序除了生成训练数据列表，还提供对音频帧率的转换和生成噪声数据列表，开发者有些自定义的数据集音频的采样率不是16000Hz的，所以提供了`change_audio_rate()`函数，帮助开发者把指定的数据集的音频采样率转换为16000Hz。提供的生成噪声数据列表`create_noise`函数，前提是要有噪声数据集，使用噪声数据在训练中实现数据增强。
```python
if __name__ == '__main__':
    # 改变音频的帧率为16000Hz
    # change_audio_rate(args.annotation_path)
    # 生成噪声的数据列表
    # create_noise()
    # 生成训练数据列表
    main()
```


## 训练模型

 - 执行训练脚本，开始训练语音识别模型， 每训练一轮保存一次模型，模型保存在`DeepSpeech/models/`目录下，默认会使用噪声音频一起训练的，如果没有这些音频，可以删除`conf/augmentation.config`中的`noise`项。
```shell script
CUDA_VISIBLE_DEVICES=0,1 python3 train.py
```

### 数据增强流水线

数据增强是用来提升深度学习性能的非常有效的技术。通过在原始音频中添加小的随机扰动（标签不变转换）获得新音频来增强的语音数据。开发者不必自己合成，因为数据增强已经嵌入到数据生成器中并且能够即时完成，在训练模型的每个epoch中随机合成音频。

目前提供六个可选的增强组件供选择，配置并插入处理过程。

  - 音量扰动
  - 速度扰动
  - 移动扰动
  - 在线贝叶斯归一化
  - 噪声干扰（需要背景噪音的音频文件）
  - 脉冲响应（需要脉冲音频文件）

为了让训练模块知道需要哪些增强组件以及它们的处理顺序，需要事先准备一个JSON格式的*扩展配置文件*。例如：

```json
[{
    "type": "speed",
    "params": {"min_speed_rate": 0.95,
               "max_speed_rate": 1.05},
    "prob": 0.6
},
{
    "type": "shift",
    "params": {"min_shift_ms": -5,
               "max_shift_ms": 5},
    "prob": 0.8
}]
```

当`trainer.py`的`--augment_conf_file`参数被设置为上述示例配置文件的路径时，每个 epoch 中的每个音频片段都将被处理。首先，均匀随机采样速率会有60％的概率在 0.95 和 1.05 之间对音频片段进行速度扰动。然后，音频片段有 80％ 的概率在时间上被挪移，挪移偏差值是 -5 毫秒和 5 毫秒之间的随机采样。最后，这个新合成的音频片段将被传送给特征提取器，以用于接下来的训练。

有关其他配置实例，请参考`conf/augmenatation.config`.

使用数据增强技术时要小心，由于扩大了训练和测试集的差异，不恰当的增强会对训练模型不利，导致训练和预测的差距增大。

## 语言模型
下载语言模型并放在lm目录下，下面下载的小语言模型，如何有足够大性能的机器，可以下载70G的超大语言模型，点击下载[Mandarin LM Large](https://deepspeech.bj.bcebos.com/zh_lm/zhidao_giga.klm) ，这个模型会大超多。
```shell script
cd DeepSpeech/
mkdir lm
cd lm
wget https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm
```

## 评估和预测

这里我也提示几点，在预测中可以提升性能的几个参数，预测包括评估，推理，部署等等一系列使用到模型预测音频的程序。解码方法，通过`decoding_method`选择不同的解码方法，支持`ctc_beam_search`定向搜索和`ctc_greedy`最优路径两种，其中`ctc_beam_search`定向搜索效果是最好的，但是速度就比较慢，这个可以通过`beam_size`参数设置定向搜索的宽度，以提高执行速度，范围[5, 500]，越大准确率就越高，同时执行速度就越慢。如果对准确率没有太严格的要求，可以考虑直接使用`ctc_greedy`最优路径方法，其实准确率也低不了多少。

 - 在训练结束之后，我们要使用这个脚本对模型进行超参数调整，提高语音识别性能。该程序主要是为了寻找`ctc_beam_search`定向搜索方法中最优的`alpha`，`beta`参数，以获得最好的识别准确率。如果使用的是`ctc_greedy`最优路径，可以直接跳过这一步。
```shell script
PYTHONPATH=.:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 python3 tools/tune.py
```

 - 我们可以使用这个脚本对模型进行评估，通过字符错误率来评价模型的性能。通过这个程序的输出，开发者就可以考虑使用哪种解码方法，以及`ctc_beam_search`定向搜索方法中`beam_size`参数的大小。
```shell script
CUDA_VISIBLE_DEVICES=0 python3 eval.py
```

 - 我们可以使用这个脚本使用模型进行预测，通过传递音频文件的路径进行识别。
```shell script
CUDA_VISIBLE_DEVICES=0 python3 infer_path.py --wav_path=./dataset/test.wav
```

 - 我们可以使用这个脚本使用模型进行预测，通过本地录音然后进行识别。
```shell script
CUDA_VISIBLE_DEVICES=0 python3 infer_record.py
```

 - 我们可以使用这个脚本使用模型进行预测，通过创建一个Web服务，通过提供HTTP接口来实现语音识别，同时还提供了一个页面来测试，可以选择本地音频文件，或者是在线录音。
```shell script
CUDA_VISIBLE_DEVICES=0 python3 infer_server.py --host=localhost --port=5000
```


## 项目部署

 - 启动语音识别服务，使用Socket通讯。需要注意的是`host_ip`参数是电脑本机的IP地址，其他使用默认就可以。
```shell script
CUDA_VISIBLE_DEVICES=0 python3 deploy/server.py
```

 - 测试服务，执行下面这个程序调用语音识别服务。在控制台中，按下`空格键`，按住并开始讲话。讲话完毕请释放该键以让控制台中显示语音的文本结果。要退出客户端，只需按`ESC键`。
```shell script
python3 deploy/client.py
```

## 模型下载
| 模型 | 下载地址 |
| :---: | :---: |
| 官方提供的模型 | [点击下载](https://deepspeech.bj.bcebos.com/demo_models/baidu_cn1.2k_model_fluid.tar.gz) |
| 自训练超大数据集(超过1300小时)的模型 | [点击下载](https://resource.doiduoyi.com/#c3cm472) |
