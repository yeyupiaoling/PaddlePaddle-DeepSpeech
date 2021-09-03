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
 - 在服务器执行下面命令通过创建一个Web服务，通过提供HTTP接口来实现语音识别。启动服务之后，在浏览器上访问`http://localhost:5000`。支持中文数字转阿拉伯数字，将参数`--to_an`设置为True即可，默认为True。
```shell script
python infer_server.py
```

![录音测试页面](https://img-blog.csdnimg.cn/20210402091159951.png)
