# 合成语音数据

1. 为了拟补数据集的不足，我们合成一批语音用于训练，使用PaddlePaddle官方的Parakeet合成中文语音，不过效果一般，慎重考虑使用。首先安装Parakeet，执行下面命令即可安装完成。
```shell
pip install -U paddle-parakeet -i https://mirrors.aliyun.com/pypi/simple/
```

2. 然后分别下载以下三个模型并解压到`tools/generate_audio/models`目录下。
```shell
https://paddlespeech.bj.bcebos.com/Parakeet/ge2e_ckpt_0.3.zip
https://paddlespeech.bj.bcebos.com/Parakeet/tacotron2_aishell3_ckpt_0.3.zip
https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_ljspeech_ckpt_0.3.zip
```

3. 把需要说话人的语音放在`tools/generate_audio/speaker_audio`目录下，可以使用`dataset/test.wav`文件，可以到找多个人的音频放在`tools/generate_audio/speaker_audio`目录下，开发者也可以尝试入自己的音频放入该目录，这样训练出来的模型能更好识别开发者的语音，采样率最好是16000Hz。

4. 然后下载一个语料，如果开发者有其他更好的语料也可以替换。然后解压`dgk_lost_conv/results`目录下的压缩文件，windows用户可以手动解压。
```shell
cd tools/generate_audio
git clone https://github.com/aceimnorstuvwxz/dgk_lost_conv.git
cd dgk_lost_conv/results
unzip dgk_shooter_z.conv.zip
unzip xiaohuangji50w_fenciA.conv.zip
unzip xiaohuangji50w_nofenci.conv.zip
```

5. 最后执行以下命令即可自动合成语音，生成的中文语料存放在`tools/generate_audio/corpus.txt`，合成的语音会放在`dataset/audio/generate`， 标注文件会放在`dataset/annotation/generate.txt`。
```shell
cd tools/generate_audio/
python generate.py
```