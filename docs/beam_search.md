# 集束搜索解码

本项目目前支持两种解码方法，分别是集束搜索解码器(ctc_beam_search)和贪心解码器(ctc_greedy)，项目全部默认都是使用贪婪策略解码的，首先要安装`paddlespeech_ctcdecoders`，执行下面命令安装。
```shell
python -m pip install  -U -i https://ppasr.yeyupiaoling.cn/pypi/simple/
```

# 语言模型

集束搜索解码需要使用到语言模型，下载语言模型并放在lm目录下，下面下载的小语言模型，如何有足够大性能的机器，可以下载70G的超大语言模型，点击下载[Mandarin LM Large](https://deepspeech.bj.bcebos.com/zh_lm/zhidao_giga.klm) ，这个模型会大超多。
```shell script
cd PaddlePaddle-DeepSpeech/
mkdir lm
cd lm
wget https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm
```

# 使用集束搜索解码

在需要使用到解码器的程序，如评估，预测，指定参数`--decoder`为`ctc_beam_search`即可，如果alpha和beta参数值有改动，修改对应的值即可。
