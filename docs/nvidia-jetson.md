# Nvidia Jetson 搭建 Paddle Inference 环境

1. 安装PaddlePaddle的Inference预测库。
```shell
wget https://paddle-inference-lib.bj.bcebos.com/2.1.1-nv-jetson-jetpack4.4-all/paddlepaddle_gpu-2.1.1-cp36-cp36m-linux_aarch64.whl
pip3 install paddlepaddle_gpu-2.1.1-cp36-cp36m-linux_aarch64.whl
```

2. 安装scikit-learn依赖库。
```shell
git clone git://github.com/scikit-learn/scikit-learn.git
cd scikit-learn
pip3 install cython
git checkout 0.24.2
pip3 install --verbose --no-build-isolation --editable .
```

3. 安装其他依赖库。
```shell
pip3 install -r requirements.txt
```
