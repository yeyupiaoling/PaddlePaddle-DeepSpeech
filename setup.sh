#!/usr/bin/env bash

sudo apt-get update
if [ $? != 0 ]; then
    echo "Update failed !!!"
    exit 1
fi

# install dependencies
sudo apt-get install -y pkg-config libflac-dev libogg-dev libvorbis-dev libboost-dev swig libsndfile1 git vim gcc
if [ $? != 0 ]; then
    echo "Install dependencies failed !!!"
    exit 1
fi
echo "Success installde pkg-config libflac-dev libogg-dev libvorbis-dev libboost-dev swig..."


# install python dependencies
pip3 install scipy==1.5.4 resampy==0.2.2 SoundFile==0.9.0.post1 python_speech_features==0.6 flask flask-cors paddlepaddle-gpu==1.8.5.post107 visualdl==2.1.1 -i https://mirrors.aliyun.com/pypi/simple/
if [ $? != 0 ]; then
    echo "Install python dependencies failed !!!"
    exit 1
fi
echo "Success install scipy resampy SoundFile python_speech_features..."


# install package libsndfile
python3 -c "import soundfile"
if [ $? != 0 ]; then
    echo "Install package libsndfile into default system path."
    wget "http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.28.tar.gz"
    if [ $? != 0 ]; then
        echo "Download libsndfile-1.0.28.tar.gz failed !!!"
        exit 1
    fi
    tar -zxvf libsndfile-1.0.28.tar.gz
    cd libsndfile-1.0.28
    ./configure > /dev/null && make > /dev/null && make install > /dev/null
    cd ..
    rm -rf libsndfile-1.0.28
    rm libsndfile-1.0.28.tar.gz
fi

cd decoders/
sh setup.sh

echo "Install all dependencies successfully."

