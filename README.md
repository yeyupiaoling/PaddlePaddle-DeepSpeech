# 语音识别

本项目是基于PaddlePaddle的[DeepSpeech2](https://github.com/PaddlePaddle/DeepSpeech)项目修改的，方便训练中文自定义数据集。

## 目录
- [环境搭建](#环境搭建)
- [数据准备](#数据准备)
- [训练模型](#训练模型)
- [评估和预测](#评估和预测)
- [项目部署](#项目部署)

## 环境搭建
Docker 是一个开源工具，用于在孤立的环境中构建、发布和运行分布式应用程序。此项目的 Docker 镜像已在[hub.docker.com](https://hub.docker.com)中提供，并安装了所有依赖项，其中包括预先构建的PaddlePaddle，CTC解码器以及其他必要的 Python 和第三方库。这个 Docker 映像需要NVIDIA GPU的支持，所以请确保它的可用性并已完成[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)的安装。

采取以下步骤来启动 Docker 镜像：

- 下载 Docker 镜像

```bash
nvidia-docker pull hub.baidubce.com/paddlepaddle/deep_speech_fluid:latest-gpu
```

- git clone 这个资源库

```
git clone https://github.com/PaddlePaddle/DeepSpeech.git
```

- 运行 Docker 镜像

```bash
sudo nvidia-docker run -it -v $(pwd)/DeepSpeech:/DeepSpeech hub.baidubce.com/paddlepaddle/deep_speech_fluid:latest-gpu /bin/bash
```

## 数据准备

```bash
cd run/
```

```bash
sh download_public_data.sh
```


```bash
sh prepare_train_data.sh
```


## 训练模型
```bash
sh download_model.sh
```


```bash
sh train.sh
```

## 评估和预测

```bash
sh hyper_parameter_tune.sh
```


```bash
sh eval.sh
```

```bash
sh infer.sh
```