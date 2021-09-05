# 本地预测

我们可以使用这个脚本使用模型进行预测，如果如何还没导出模型，需要执行[导出模型](./export_model.md)操作把模型参数导出为预测模型，通过传递音频文件的路径进行识别，通过参数`--wav_path`指定需要预测的音频路径。支持中文数字转阿拉伯数字，将参数`--to_an`设置为True即可，默认为True。
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
vocab_path: ./dataset/zh_vocab.txt
wav_path: ./dataset/test.wav
------------------------------------------------
消耗时间：132, 识别结果: 近几年不但我用书给女儿儿压岁也劝说亲朋不要给女儿压岁钱而改送压岁书, 得分: 94
```

## 长语音预测

通过参数`--is_long_audio`可以指定使用长语音识别方式，这种方式通过VAD分割音频，再对短音频进行识别，拼接结果，最终得到长语音识别结果。
```shell script
python infer_path.py --wav_path=./dataset/test_vad.wav --is_long_audio=True
```


## Web部署

在服务器执行下面命令通过创建一个Web服务，通过提供HTTP接口来实现语音识别。启动服务之后，如果在本地运行的话，在浏览器上访问`http://localhost:5000`，否则修改为对应的 IP地址。打开页面之后可以选择上传长音或者短语音音频文件，也可以在页面上直接录音，录音完成之后点击上传，播放功能只支持录音的音频。支持中文数字转阿拉伯数字，将参数`--to_an`设置为True即可，默认为True。
```shell script
python infer_server.py
```

打开页面如下：
![录音测试页面](./images/infer_server.jpg)


## GUI界面部署
通过打开页面，在页面上选择长语音或者短语音进行识别，也支持录音识别，同时播放识别的音频。默认使用的是贪心解码策略，如果需要使用集束搜索方法的话，需要在启动参数的时候指定。
```shell script
python infer_gui.py
```

打开界面如下：
![GUI界面](./images/infer_gui.jpg)
