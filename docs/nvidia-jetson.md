# Nvidia Jetson部署

1. 这对Nvidia Jetson设备，如Nano、Nx、AGX等设备，可以通过下面命令安装PaddlePaddle的Inference预测库。
```shell
wget https://paddle-inference-lib.bj.bcebos.com/2.6.1/python/Jetson/jetpack5.0.2_gcc9.4/all/paddlepaddle_gpu-2.6.1-cp38-cp38-linux_aarch64.whl
pip3 install paddlepaddle_gpu-2.6.1-cp38-cp38-linux_aarch64.whl
```

2. 安装其他依赖库，如果缺少库，可以手动安装。
```shell
pip3 install -r requirements.txt
```

3. 执行预测，直接使用根目录下的预测代码。
```shell
python infer_path.py --wav_path=./dataset/test.wav
```

以Nvidia AGX为例，输出结果如下：
```
WARNING: AVX is not support on your machine. Hence, no_avx core will be imported, It has much worse preformance than avx core.
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
use_tensorrt: False
vocab_path: ./dataset/zh_vocab.txt
wav_path: ./dataset/test.wav
------------------------------------------------
消耗时间：416ms, 识别结果: 近几年不但我用书给女儿压岁也劝说亲朋不要给女儿压岁钱而改送压岁书, 得分: 97
```