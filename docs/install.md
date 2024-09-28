# 搭建本地环境

本人用的就是本地环境和使用Anaconda，并创建了Python3.11的虚拟环境，建议读者也本地环境，方便交流，出现安装问题，随时提[issue](https://github.com/yeyupiaoling/PaddlePaddle-DeepSpeech/issues)。

 - 首先安装的是PaddlePaddle 2.6.1 的GPU版本，如果已经安装过了，请跳过。
```shell
conda install paddlepaddle-gpu==2.6.1 cudatoolkit=11.7 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
```

 - 安装其他依赖库。
```shell
python -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```
