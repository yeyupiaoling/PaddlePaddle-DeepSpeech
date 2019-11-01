#! /usr/bin/env bash

cd ../ > /dev/null

# Create manifest file
PYTHONPATH=.:$PYTHONPATH python data/create_manifest.py \
--annotation_path='./dataset/annotation/' \
--manifest_prefix='./dataset/'

if [ $? -ne 0 ]; then
    echo "Prepare manifest file failed. Terminated."
    exit 1
fi


# Compute mean and std
PYTHONPATH=.:$PYTHONPATH python tools/compute_mean_std.py \
--num_samples=2000 \
--specgram_type='linear' \
--manifest_path='./dataset/manifest.train' \
--output_path='./dataset/mean_std.npz'

if [ $? -ne 0 ]; then
    echo "Prepare z-score failed. Terminated."
    exit 1
fi


# Build vocab
PYTHONPATH=.:$PYTHONPATH python tools/build_vocab.py \
--count_threshold=0 \
--vocab_path='./dataset/zh_vocab.txt' \
--manifest_paths='./dataset/manifest.train'

if [ $? -ne 0 ]; then
    echo "Prepare build vocab failed. Terminated."
    exit 1
fi


echo "Prepare train data done."
exit 0