#! /usr/bin/env bash

cd ../ > /dev/null

# Hyper-parameter tune
PYTHONPATH=.:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python tools/tune.py \
--alpha_from=1.0 \
--alpha_to=3.2 \
--num_alphas=45 \
--beta_from=0.1 \
--beta_to=0.45 \
--num_betas=8 \
--tune_manifest='./dataset/manifest.test' \
--mean_std_path='./dataset/mean_std.npz' \
--vocab_path='./dataset/zh_vocab.txt' \
--model_path='./models/checkpoints/srep_final' \
--lang_model_path='./models/zhidao_giga.klm'

if [ $? -ne 0 ]; then
    echo "Hyper-parameter tune failed. Terminated."
    exit 1
fi

echo "Hyper-parameter tune done."
exit 0
