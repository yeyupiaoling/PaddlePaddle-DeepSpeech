#! /usr/bin/env bash

cd ../ > /dev/null

# download aishell data
PYTHONPATH=.:$PYTHONPATH python data/aishell.py \
--target_dir='./dataset/audio/' \
--annotation_text='./dataset/annotation/'

if [ $? -ne 0 ]; then
    echo "Prepare Aishell failed. Terminated."
    exit 1
fi


# download Free ST-Chinese-Mandarin-Corpus data
PYTHONPATH=.:$PYTHONPATH python data/free_st_chinese_mandarin_corpus.py \
--target_dir='./dataset/audio/' \
--annotation_text='./dataset/annotation/'

if [ $? -ne 0 ]; then
    echo "Prepare Free ST-Chinese-Mandarin-Corpus failed. Terminated."
    exit 1
fi


# download THCHS-30 data
PYTHONPATH=.:$PYTHONPATH python data/thchs_30.py \
--target_dir='./dataset/audio/' \
--annotation_text='./dataset/annotation/'

if [ $? -ne 0 ]; then
    echo "Prepare THCHS-30 failed. Terminated."
    exit 1
fi


echo "Pubilc data download done."
exit 0
