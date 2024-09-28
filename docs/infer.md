# 本地预测

我们可以使用这个脚本使用模型进行预测，如果如何还没导出模型，需要执行[导出模型](./export_model.md)操作把模型参数导出为预测模型，通过传递音频文件的路径进行识别，通过参数`--wav_path`指定需要预测的音频路径。
```shell script
python infer_path.py --wav_path=./dataset/test.wav
```

输出结果：
```
2024-09-28 12:02:04.321 | INFO     | utils.utils:print_arguments:11 - ----------- 额外配置参数 -----------
2024-09-28 12:02:04.321 | INFO     | utils.utils:print_arguments:13 - alpha: 1.2
2024-09-28 12:02:04.321 | INFO     | utils.utils:print_arguments:13 - beam_size: 300
2024-09-28 12:02:04.321 | INFO     | utils.utils:print_arguments:13 - beta: 0.35
2024-09-28 12:02:04.321 | INFO     | utils.utils:print_arguments:13 - cutoff_prob: 0.99
2024-09-28 12:02:04.321 | INFO     | utils.utils:print_arguments:13 - cutoff_top_n: 40
2024-09-28 12:02:04.321 | INFO     | utils.utils:print_arguments:13 - decoder: ctc_greedy
2024-09-28 12:02:04.321 | INFO     | utils.utils:print_arguments:13 - enable_mkldnn: False
2024-09-28 12:02:04.322 | INFO     | utils.utils:print_arguments:13 - lang_model_path: ./lm/zh_giga.no_cna_cmn.prune01244.klm
2024-09-28 12:02:04.322 | INFO     | utils.utils:print_arguments:13 - model_dir: ./models/infer/
2024-09-28 12:02:04.322 | INFO     | utils.utils:print_arguments:13 - to_itn: False
2024-09-28 12:02:04.322 | INFO     | utils.utils:print_arguments:13 - use_gpu: True
2024-09-28 12:02:04.322 | INFO     | utils.utils:print_arguments:13 - vocab_path: ./dataset/vocabulary.txt
2024-09-28 12:02:04.322 | INFO     | utils.utils:print_arguments:13 - wav_path: ./dataset/test.wav
2024-09-28 12:02:04.322 | INFO     | utils.utils:print_arguments:14 - ------------------------------------------------
消耗时间：132, 识别结果: 近几年不但我用书给女儿儿压岁也劝说亲朋不要给女儿压岁钱而改送压岁书
```

## Web部署

在服务器执行下面命令通过创建一个Web服务，通过提供HTTP接口来实现语音识别。启动服务之后，如果在本地运行的话，在浏览器上访问`http://localhost:5000`，否则修改为对应的 IP地址。打开页面之后可以选择上传长音或者短语音音频文件，也可以在页面上直接录音，录音完成之后点击上传，播放功能只支持录音的音频。
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
