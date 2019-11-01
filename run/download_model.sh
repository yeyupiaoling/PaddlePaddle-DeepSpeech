#! /usr/bin/env bash

. utility.sh

# Download pretrained model
#URL='https://deepspeech.bj.bcebos.com/demo_models/baidu_cn1.2k_model_fluid.tar.gz'
URL='http://192.168.1.118:55000/aishell_model_fluid.tar.gz'
MD5=2bf0cc8b6d5da2a2a787b5cc36a496b5
TARGET=./models/baidu_cn1.2k_model_fluid.tar.gz


echo "Download Aishell model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download pretrained model model!"
    exit 1
fi
tar -zxvf $TARGET


# Download language model
#URL='https://deepspeech.bj.bcebos.com/zh_lm/zhidao_giga.klm'
URL='http://192.168.1.118:55000/zhidao_giga.klm'
MD5=2bf0cc8b6d5da2a2a787b5cc36a496b5
TARGET=./models/zhidao_giga.klm

echo "Download language model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download language model!"
    exit 1
fi


exit 0

