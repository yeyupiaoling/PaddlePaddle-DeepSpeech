#! /usr/bin/env bash

cd ../ > /dev/null

# start server
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=.:$PYTHONPATH python -u deploy/server.py \
--host_ip="192.168.1.119" \
--host_port=10086 \
--beam_size=500 \
--num_conv_layers=2 \
--num_rnn_layers=3 \
--rnn_layer_size=2048 \
--alpha=2.5 \
--beta=0.3 \
--cutoff_prob=1.0 \
--cutoff_top_n=40 \
--use_gru=True \
--use_gpu=True \
--share_rnn_weights=False \
--speech_save_dir="./audios_cache" \
--warmup_manifest="./dataset/manifest.test" \
--mean_std_path="./models/baidu_cn1.2k_model_fluid/mean_std.npz" \
--vocab_path="./models/baidu_cn1.2k_model_fluid/vocab.txt" \
--model_path="./models/baidu_cn1.2k_model_fluid/" \
--lang_model_path="./models/zhidao_giga.klm" \
--decoding_method="ctc_beam_search" \
--specgram_type="linear"


if [ $? -ne 0 ]; then
    echo "Failed in start server!"
    exit 1
fi


exit 0


