# 训练模型

 - 执行训练脚本，开始训练语音识别模型， 每训练一轮和每2000个batch都会保存一次模型，模型保存在`PaddlePaddle-DeepSpeech/models/param/`目录下，默认会使用数据增强训练，如何不想使用数据增强，只需要将参数`augment_conf_path`设置为`None`即可。关于数据增强，请查看[数据增强](./faq.md)部分。如果没有关闭测试，在每一轮训练结果之后，都会执行一次测试计算模型在测试集的准确率。执行训练时，如果是Linux下，通过`CUDA_VISIBLE_DEVICES`可以指定多卡训练。
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