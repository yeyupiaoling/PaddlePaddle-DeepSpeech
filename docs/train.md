# 训练模型

 - 执行训练脚本，开始训练语音识别模型， 每训练一轮保存一次模型，模型保存在`DeepSpeech/models/`目录下，默认会使用噪声音频作为数据增强一起训练的，如果没有这些音频，自动忽略噪声数据增强。关于数据增强，请查看[数据增强](docs/faq.md)部分。如果没有关闭测试，在每一轮训练结果之后，都会执行一次测试，为了提高测试的速度，测试使用的是最优解码路径解码，这个解码方式结果没有集束搜索的方法准确率高，所以测试的输出的准确率可以理解为保底的准确率。
```shell script
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

 - 在训练过程中，程序会使用VisualDL记录训练结果，可以通过以下的命令启动VisualDL。
```shell
visualdl --logdir=log --host=0.0.0.0
```

 - 然后再浏览器上访问`http://localhost:8040`可以查看结果显示，如下。

![Learning rate](https://img-blog.csdnimg.cn/20210318165719805.png)
![Test Cer](https://s3.ax1x.com/2021/03/01/6PJaZV.jpg)
![Train Loss](https://s3.ax1x.com/2021/03/01/6PJNq0.jpg)