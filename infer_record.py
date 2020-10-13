# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import time
import wave

import numpy as np
import paddle.fluid as fluid
import pyaudio

from data_utils.data import DataGenerator
from model_utils.model import DeepSpeech2Model
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('wav_path',         str,    './dataset/test.wav', "Server's IP port.")
add_arg('beam_size',        int,    500,    "Beam search width.")
add_arg('num_conv_layers',  int,    2,      "# of convolution layers.")
add_arg('num_rnn_layers',   int,    3,      "# of recurrent layers.")
add_arg('rnn_layer_size',   int,    2048,   "# of recurrent cells per layer.")
add_arg('alpha',            float,  1.2,   "Coef of LM for beam search.")
add_arg('beta',             float,  0.35,   "Coef of WC for beam search.")
add_arg('cutoff_prob',      float,  1.0,    "Cutoff probability for pruning.")
add_arg('cutoff_top_n',     int,    40,     "Cutoff number for pruning.")
add_arg('use_gru',          bool,   True,  "Use GRUs instead of simple RNNs.")
add_arg('use_gpu',          bool,   True,   "Use GPU or not.")
add_arg('share_rnn_weights',bool,   False,   "Share input-hidden weights across bi-directional RNNs. Not for GRU.")
add_arg('mean_std_path',    str,    './dataset/mean_std.npz',        "Filepath of normalizer's mean & std.")
add_arg('vocab_path',       str,    './dataset/zh_vocab.txt',        "Filepath of vocabulary.")
add_arg('model_path',       str,    './models/step_final/',        "If None, the training starts from scratch,  otherwise, it resumes from the pre-trained model.")
add_arg('lang_model_path',  str,    './lm/zh_giga.no_cna_cmn.prune01244.klm',        "Filepath for language model.")
add_arg('decoding_method',  str,    'ctc_beam_search',        "Decoding method. Options: ctc_beam_search, ctc_greedy", choices=['ctc_beam_search', 'ctc_greedy'])
add_arg('specgram_type',    str,    'linear',        "Audio feature type. Options: linear, mfcc.", choices=['linear', 'mfcc'])
# yapf: disable
args = parser.parse_args()

if args.use_gpu:
    place = fluid.CUDAPlace(0)
else:
    place = fluid.CPUPlace()

data_generator = DataGenerator(vocab_filepath=args.vocab_path,
                               mean_std_filepath=args.mean_std_path,
                               augmentation_config='{}',
                               specgram_type=args.specgram_type,
                               keep_transcription_text=True,
                               place=place,
                               is_training=False)
# prepare ASR model
ds2_model = DeepSpeech2Model(vocab_size=data_generator.vocab_size,
                             num_conv_layers=args.num_conv_layers,
                             num_rnn_layers=args.num_rnn_layers,
                             rnn_layer_size=args.rnn_layer_size,
                             use_gru=args.use_gru,
                             init_from_pretrained_model=args.model_path,
                             place=place,
                             share_rnn_weights=args.share_rnn_weights)

vocab_list = [chars.encode("utf-8") for chars in data_generator.vocab_list]

if args.decoding_method == "ctc_beam_search":
    ds2_model.init_ext_scorer(args.alpha, args.beta, args.lang_model_path,
                              vocab_list)


def predict(filename):
    feature = data_generator.process_utterance(filename, "")
    audio_len = feature[0].shape[1]
    mask_shape0 = (feature[0].shape[0] - 1) // 2 + 1
    mask_shape1 = (feature[0].shape[1] - 1) // 3 + 1
    mask_max_len = (audio_len - 1) // 3 + 1
    mask_ones = np.ones((mask_shape0, mask_shape1))
    mask_zeros = np.zeros((mask_shape0, mask_max_len - mask_shape1))
    mask = np.repeat(np.reshape(np.concatenate((mask_ones, mask_zeros), axis=1),
                                (1, mask_shape0, mask_max_len)), 32, axis=0)
    feature = [np.array([feature[0]]).astype('float32'),
               None,
               np.array([audio_len]).astype('int64').reshape([-1, 1]),
               np.array([mask]).astype('float32')]
    probs_split = ds2_model.infer_batch_probs(infer_data=feature)

    if args.decoding_method == "ctc_greedy":
        result_transcript = ds2_model.decode_batch_greedy(probs_split=probs_split,
                                                          vocab_list=vocab_list)
    else:
        result_transcript = ds2_model.decode_batch_beam_search(probs_split=probs_split,
                                                               beam_alpha=args.alpha,
                                                               beam_beta=args.beta,
                                                               beam_size=args.beam_size,
                                                               cutoff_prob=args.cutoff_prob,
                                                               cutoff_top_n=args.cutoff_top_n,
                                                               vocab_list=vocab_list,
                                                               num_processes=1)
    return result_transcript[0]


def save_wave_file(filename, data):
    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(SAMPWIDTH)
    wf.setframerate(RATE)
    wf.writeframes(b"".join(data))
    wf.close()


def record(wav_path, time=5):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    my_buf = []
    print("录音中(%ds)" % time)
    for i in range(0, int(RATE / CHUNK * time)):
        data = stream.read(CHUNK)
        my_buf.append(data)
        print(".", end="", flush=True)

    save_wave_file(wav_path, my_buf)
    stream.close()


if __name__ == '__main__':
    print_arguments(args)
    # 录音格式
    RATE = 16000
    CHUNK = 1024
    CHANNELS = 1
    SAMPWIDTH = 4
    # 临时保存路径
    while True:
        _ = input("按下回车键开机录音，录音%s秒中：" % args.record_time)
        record(args.wav_path, time=args.record_time)
        start = time.time()
        result_text = predict(filename=args.wav_path)
        end = time.time()
        print("识别时间：%dms，识别结果：%s" % (round((end - start) * 1000), result_text))
