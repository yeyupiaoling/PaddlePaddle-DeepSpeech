import os
import random
import re
import time

import numpy as np
import paddle
import soundfile
from parakeet.frontend import Vocab
from parakeet.models import ConditionalWaveFlow
from parakeet.models.lstm_speaker_encoder import LSTMSpeakerEncoder
from parakeet.models.tacotron2 import Tacotron2
from tqdm import tqdm

from audio_processor import SpeakerVerificationPreprocessor
from chinese_g2p import convert_sentence, is_uchar
from preprocess_transcription import _phones, _tones

# 说话人解码器
speaker_encoder = LSTMSpeakerEncoder(n_mels=40, num_layers=3, hidden_size=256, output_size=256)

# Tacotron2模型
synthesizer = Tacotron2(vocab_size=68,
                        n_tones=10,
                        d_mels=80,
                        d_encoder=512,
                        encoder_conv_layers=3,
                        encoder_kernel_size=5,
                        d_prenet=256,
                        d_attention_rnn=1024,
                        d_decoder_rnn=1024,
                        attention_filters=32,
                        attention_kernel_size=31,
                        d_attention=128,
                        d_postnet=512,
                        postnet_kernel_size=5,
                        postnet_conv_layers=5,
                        reduction_factor=1,
                        p_encoder_dropout=0.5,
                        p_prenet_dropout=0.5,
                        p_attention_dropout=0.1,
                        p_decoder_dropout=0.1,
                        p_postnet_dropout=0.5,
                        d_global_condition=256,
                        use_stop_token=False, )

# 声码器
vocoder = ConditionalWaveFlow(upsample_factors=[16, 16], n_flows=8, n_layers=8, n_group=16, channels=128, n_mels=80,
                              kernel_size=[3, 3])

# 音频预处理
p = SpeakerVerificationPreprocessor(sampling_rate=16000,
                                    audio_norm_target_dBFS=-30,
                                    vad_window_length=30,
                                    vad_moving_average_width=8,
                                    vad_max_silence_length=6,
                                    mel_window_length=25,
                                    mel_window_step=10,
                                    n_mels=40,
                                    partial_n_frames=160,
                                    min_pad_coverage=0.75,
                                    partial_overlap_ratio=0.5)

# 说话人解码器的模型，下载地址：https://paddlespeech.bj.bcebos.com/Parakeet/ge2e_ckpt_0.3.zip
speaker_encoder_params_path = "models/step-3000000.pdparams"
speaker_encoder.set_state_dict(paddle.load(speaker_encoder_params_path))
speaker_encoder.eval()

# Tacotron2的模型文件，下载地址：https://paddlespeech.bj.bcebos.com/Parakeet/tacotron2_aishell3_ckpt_0.3.zip
params_path = "models/step-450000.pdparams"
synthesizer.set_state_dict(paddle.load(params_path))
synthesizer.eval()

# WaveFlow模型，下载地址：https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_ljspeech_ckpt_0.3.zip
params_path = "models/step-2000000.pdparams"
vocoder.set_state_dict(paddle.load(params_path))
vocoder.eval()

voc_phones = Vocab(sorted(list(_phones)))
voc_tones = Vocab(sorted(list(_tones)))

# 说话人的音频路径
speakers_audio_path = "speaker_audio/"
embeds = []

# 获取全部说话人音频编码
for speaker_audio_path in os.listdir(speakers_audio_path):
    speaker_audio_path = os.path.join(speakers_audio_path, speaker_audio_path)
    # 获取模仿人音频的编码
    mel_sequences = p.extract_mel_partials(p.preprocess_wav(speaker_audio_path))
    with paddle.no_grad():
        embed = speaker_encoder.embed_utterance(paddle.to_tensor(mel_sequences))
        embeds.append(embed)

# 制作中文语料
with open('corpus.txt', 'w', encoding='utf-8') as f_write:
    corpus_dir = 'dgk_lost_conv/results/'
    for corpus_path in os.listdir(corpus_dir):
        if corpus_path[-5:] != '.conv': continue
        if corpus_path == 'dgk_shooter_z.conv': continue
        corpus_path = os.path.join(corpus_dir, corpus_path)
        print(corpus_path)
        with open(corpus_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            line = line[2:].replace('/', '').replace('\n', '').replace('?', '$').replace(' ', '%').replace('！', '%')
            line = line.replace('。', '%').replace('!', '%').replace('？', '%').replace('"', '').replace('.', '')
            line = line.replace('～', '$').replace('，', '$').replace(',', '$').replace('、', '%')
            line = line.replace('%%', '%').replace('$$', '$')
            line = line.replace('%%', '%').replace('$$', '$')
            if len(line) < 2: continue
            if not is_uchar(line.replace('$', '').replace('%', '')): continue
            my_re = re.compile(r'[A-Za-z0-9]', re.S)
            res = re.findall(my_re, line)
            if len(res) > 0: continue
            f_write.write('%s\n' % line)

# 创建保存目录
save_path = '../../dataset/audio/generate'
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_ann_path = '../../dataset/annotation'
if not os.path.exists(save_ann_path):
    os.makedirs(save_ann_path)
f_label = open(os.path.join(save_ann_path, 'generate.txt'), 'a', encoding='utf-8')

# 开始合成语音
with open('corpus.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for sentence in tqdm(lines):
    # 要合成的文本
    sentence = sentence.replace('\n', '')
    phones, tones = convert_sentence(sentence)

    try:
        phones = np.array([voc_phones.lookup(item) for item in phones], dtype=np.int64)
        tones = np.array([voc_tones.lookup(item) for item in tones], dtype=np.int64)
    except:
        continue

    phones = paddle.to_tensor(phones).unsqueeze(0)
    tones = paddle.to_tensor(tones).unsqueeze(0)
    embed = random.choice(embeds)
    utterance_embeds = paddle.unsqueeze(embed, 0)

    outputs = synthesizer.infer(phones, tones=tones, global_condition=utterance_embeds)
    mel_input = paddle.transpose(outputs["mel_outputs_postnet"], [0, 2, 1])

    with paddle.no_grad():
        wav = vocoder.infer(mel_input)
    wav = wav.numpy()[0]
    # 保存语音文件
    save_audio_path = os.path.join(save_path, "%s.wav" % int(time.time() * 1000))
    soundfile.write(save_audio_path, wav, samplerate=16000)
    f_label.write('%s\t%s' % (save_audio_path, sentence.replace('%', '').replace('$', '')))
