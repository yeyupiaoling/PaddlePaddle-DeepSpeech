# 数据准备

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
python create_data.py
```

在生成数据列表的是要注意，该程序除了生成训练数据列表，还提供对音频帧率的转换和生成噪声数据列表，开发者有些自定义的数据集音频的采样率不是16000Hz的，所以提供了`change_audio_rate()`函数，帮助开发者把指定的数据集的音频采样率转换为16000Hz。提供的生成噪声数据列表`create_noise`函数，前提是要有噪声数据集，使用噪声数据在训练中实现数据增强。
```python
if __name__ == '__main__':
    # 生成噪声的数据列表
    create_noise()
    # 生成训练数据列表
    main()
```