
# 本地预测

我们可以使用这个脚本使用模型进行预测，如果如何还没导出模型，需要执行上面的操作把模型参数导出为预测模型，通过传递音频文件的路径进行识别。默认使用的是`ctc_greedy`贪心解码策略，这也是为了支持Windows用户。
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

