# 搭建Docker环境

 - 请提前安装好显卡驱动，然后执行下面的命令。
```shell script
# 卸载系统原有docker
sudo apt-get remove docker docker-engine docker.io containerd runc
# 更新apt-get源 
sudo apt-get update
# 安装docker的依赖 
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
# 添加Docker的官方GPG密钥：
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
# 验证拥有指纹
sudo apt-key fingerprint 0EBFCD88
# 设置稳定存储库
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
```

 - 安装Docker
```shell script
# 再次更新apt-get源 
sudo apt-get update
# 开始安装docker 
sudo apt-get install docker-ce
# 加载docker 
sudo apt-cache madison docker-ce
# 验证docker是否安装成功
sudo docker run hello-world
```

 - 安装nvidia-docker
```shell script
# 设置stable存储库和GPG密钥
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 更新软件包清单后
sudo apt-get update

# 安装软件包
sudo apt-get install -y nvidia-docker2

# 设置默认运行时后，重新启动Docker守护程序以完成安装：
sudo systemctl restart docker

# 测试
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

 - 拉取PaddlePaddle 1.8.5镜像，因为这个项目需要在PaddlePaddle 1.8版本才可以运行。
```shell script
sudo nvidia-docker pull registry.baidubce.com/paddlepaddle/paddle:2.1.1-gpu-cuda10.2-cudnn7
```

- git clone 本项目源码
```shell script
git clone https://github.com/yeyupiaoling/DeepSpeech.git
```

- 运行PaddlePaddle语音识别镜像，这里设置与主机共同拥有IP和端口号。
```shell script
sudo nvidia-docker run -it --net=host -v $(pwd)/DeepSpeech:/DeepSpeech registry.baidubce.com/paddlepaddle/paddle:2.1.1-gpu-cuda10.2-cudnn7 /bin/bash
```

 - 切换到`/DeepSpeech/`目录下，首先将docker的Python3默认为Python3.7，然后切换g++为g++5，然后安装LLVM。最后执行`setup.sh`脚本安装依赖环境，执行前需要去掉`setup.sh`和`decoder/setup.sh`安装依赖库时使用的`sudo`命令，因为在docker中本来就是root环境，等待安装即可。
```shell script
# 修改Docker的Python3版本为3.7
rm -rf /usr/local/bin/python3
ln -s /home/Python-3.7.0/python /usr/local/bin/python3

# 切换默认的g++版本为5
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 30 --slave /usr/bin/g++ g++ /usr/bin/g++-5
update-alternatives --config gcc

# 开始安装LLVM
cd /home
wget https://releases.llvm.org/9.0.0/llvm-9.0.0.src.tar.xz
wget http://releases.llvm.org/9.0.0/cfe-9.0.0.src.tar.xz
wget http://releases.llvm.org/9.0.0/clang-tools-extra-9.0.0.src.tar.xz
tar xvf llvm-9.0.0.src.tar.xz
tar xvf cfe-9.0.0.src.tar.xz
tar xvf clang-tools-extra-9.0.0.src.tar.xz
mv llvm-9.0.0.src llvm-src
mv cfe-9.0.0.src llvm-src/tools/clang
mv clang-tools-extra-9.0.0.src llvm-src/tools/clang/tools/extra
mkdir -p /usr/local/llvm
mkdir -p llvm-src/build
cd llvm-src/build
cmake -G "Unix Makefiles" -DLLVM_TARGETS_TO_BUILD=X86 -DCMAKE_BUILD_TYPE="Release" -DCMAKE_INSTALL_PREFIX="/usr/local/llvm" ..
make -j8 && make install
export LLVM_CONFIG=/usr/local/llvm/bin/llvm-config
```

