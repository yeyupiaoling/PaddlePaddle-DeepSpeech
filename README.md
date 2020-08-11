# 语音识别

本项目是基于PaddlePaddle的[DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech)项目修改的，方便训练中文自定义数据集。

本项目使用的环境：
 - Python 2.7
 - PaddlePaddle 1.8

## 目录

- [环境搭建](#环境搭建)
- [数据准备](#数据准备)
- [训练模型](#训练模型)
- [评估和预测](#评估和预测)
- [项目部署](#项目部署)

## 环境搭建

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
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
```

 - 拉取PaddlePaddle 1.8镜像，因为这个项目必须要在 PaddlePaddle 1.8版本以上才可以运行。
```shell script
sudo nvidia-docker pull hub.baidubce.com/paddlepaddle/paddle:1.8.3-gpu-cuda10.0-cudnn7
```

- git clone 本项目源码
```shell script
git clone https://github.com/yeyupiaoling/DeepSpeech.git
```

- 运行PaddlePaddle 1.8镜像，这里设置与主机共同拥有IP和端口号。
```shell script
sudo nvidia-docker run -it --net=host -v $(pwd)/DeepSpeech:/DeepSpeech hub.baidubce.com/paddlepaddle/paddle:1.8.3-gpu-cuda10.0-cudnn7 /bin/bash
```

 - 切换到`/DeepSpeech/`目录下，执行`setup.sh`脚本安装依赖环境，等待安装即可。
```shell script
cd DeepSpeech/
sh setup.sh
```

### 搭建本地环境

 - 并不建议使用本地进行训练和预测，但是如何开发者必须使用本地环境，可以执行下面的命令。因为每个电脑的环境不一样，不能保证能够正常使用。首先需要正确安装 PaddlePaddle 1.8的GPU版本，并安装相关的CUDA和CUDNN。
```shell script
pip2 install paddlepaddle-gpu==1.8.0.post107 -i https://mirrors.aliyun.com/pypi/simple/
```

- git clone 本项目源码
```shell script
git clone https://github.com/yeyupiaoling/DeepSpeech.git
```

 - 切换到`DeepSpeech/`根目录下，执行`setup.sh`脚本安装依赖环境，等待安装即可。
```shell script
cd DeepSpeech/
sudo sh setup.sh
```

## 数据准备

1. 在`data`目录下是公开数据集的下载和制作训练数据列表和字典的，本项目提供了下载公开的中文普通话语音数据集，分别是Aishell，Free ST-Chinese-Mandarin-Corpus，THCHS-30 这三个数据集，总大小超过28G。下载这三个数据只需要执行一下代码即可，当然如何想快速训练，也可以只下载其中一个。
```shell script
cd data/
python aishell.py
python free_st_chinese_mandarin_corpus.py
python thchs_30.py
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
PYTHONPATH=.:$PYTHONPATH python tools/create_manifest.py
# 计算均值和标准差
PYTHONPATH=.:$PYTHONPATH python tools/compute_mean_std.py
# 构建字典
PYTHONPATH=.:$PYTHONPATH python tools/build_vocab.py
```


## 训练模型

 - 执行训练脚本，开始训练语音识别模型， 每训练一轮保存一次模型，模型保存在`DeepSpeech/models/`目录下。
```shell script
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

## 语言模型
下载语言模型并放在lm目录下，下面下载的小语言模型，如何有足够大性能的机器，可以下载70G的超大语言模型，点击下载[Mandarin LM Large](https://deepspeech.bj.bcebos.com/zh_lm/zhidao_giga.klm) ，这个模型会大超多。
```shell script
cd DeepSpeech/
mkdir lm
cd lm
wget https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm
```

## 评估和预测

 - 在训练结束之后，我们要使用这个脚本对模型进行超参数调整，提高语音识别性能。最后输出的`alpha`，`beta`这两个参数的值需要在之后的推理中使用这个参数值，以获得最好的识别准确率。
```shell script
PYTHONPATH=.:$PYTHONPATH CUDA_VISIBLE_DEVICES=0,1 python tools/tune.py
```

 - 我们可以使用这个脚本对模型进行评估，通过字符错误率来评价模型的性能。
```shell script
CUDA_VISIBLE_DEVICES=0,1 python eval.py
```


## 项目部署

 - 启动语音识别服务，使用Socket通讯。需要注意的是`host_ip`参数是电脑本机的IP地址，其他使用默认就可以。
```shell script
CUDA_VISIBLE_DEVICES=0,1 python deploy/server.py
```

 - 测试服务，执行下面这个程序调用语音识别服务。在控制台中，按下`空格键`，按住并开始讲话。讲话完毕请释放该键以让控制台中显示语音的文本结果。要退出客户端，只需按`ESC键`。
```shell script
python deploy/client.py
```
