import json
from typing import List

import numpy as np
from paddle.io import Dataset

from yeaudio.audio import AudioSegment
from yeaudio.augmentation import ReverbPerturbAugmentor, SpecAugmentor, SpecSubAugmentor
from yeaudio.augmentation import SpeedPerturbAugmentor, VolumePerturbAugmentor, NoisePerturbAugmentor

from data_utils.audio_featurizer import AudioFeaturizer
from data_utils.tokenizer import Tokenizer


# 音频数据加载器
class CustomDataset(Dataset):
    def __init__(self,
                 data_manifest: [str or List],
                 audio_featurizer: AudioFeaturizer,
                 tokenizer: Tokenizer = None,
                 min_duration=0,
                 max_duration=20,
                 aug_conf=None,
                 sample_rate=16000,
                 use_dB_normalization=True,
                 target_dB=-20,
                 mode="train"):
        super(CustomDataset, self).__init__()
        self._target_sample_rate = sample_rate
        self._use_dB_normalization = use_dB_normalization
        self._target_dB = target_dB
        self.mode = mode
        self.dataset_reader = None
        self.speed_augment = None
        self.volume_augment = None
        self.noise_augment = None
        self.reverb_augment = None
        self.spec_augment = None
        self.spec_sub_augment = None
        self._audio_featurizer = audio_featurizer
        self._tokenizer = tokenizer
        # 获取数据增强器
        if mode == "train" and aug_conf is not None:
            self.get_augmentor(aug_conf)
        # 获取文本格式数据列表
        with open(data_manifest, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.data_list = []
        for line in lines:
            line = json.loads(line)
            # 跳过超出长度限制的音频
            if line["duration"] < min_duration:
                continue
            if max_duration != -1 and line["duration"] > max_duration:
                continue
            self.data_list.append(dict(line))

    def __getitem__(self, idx):
        data_list = self.data_list[idx]
        # 分割音频路径和标签
        audio_file, transcript = data_list["audio_filepath"], data_list["text"]
        audio_segment = AudioSegment.from_file(audio_file)
        # 音频增强
        if self.mode == 'train':
            audio_segment = self.augment_audio(audio_segment)
        # 重采样
        if audio_segment.sample_rate != self._target_sample_rate:
            audio_segment.resample(self._target_sample_rate)
        # 音量归一化
        if self._use_dB_normalization:
            audio_segment.normalize(target_db=self._target_dB)
        # 预处理，提取特征
        feature = self._audio_featurizer.featurize(audio_segment.samples, audio_segment.sample_rate)
        text_ids = self._tokenizer.text2ids(transcript)
        # 特征增强
        if self.mode == 'train':
            if self.spec_augment is not None:
                feature = self.spec_augment(feature)
            if self.spec_sub_augment is not None:
                feature = self.spec_sub_augment(feature)
        feature = feature.astype(np.float32)
        text_ids = np.array(text_ids, dtype=np.int32)
        return feature, text_ids

    def __len__(self):
        return len(self.data_list)

    @property
    def feature_dim(self):
        """返回音频特征大小

        :return: 词汇表大小
        :rtype: int
        """
        return self._audio_featurizer.feature_dim

    @property
    def vocab_size(self):
        """返回词汇表大小

        :return: 词汇表大小
        :rtype: int
        """
        return self._tokenizer.vocab_size

    @property
    def vocab_list(self):
        """返回词汇表列表

        :return: 词汇表列表
        :rtype: list
        """
        return self._tokenizer.vocab_list

    # 获取数据增强器
    def get_augmentor(self, aug_conf):
        if aug_conf.speed is not None:
            self.speed_augment = SpeedPerturbAugmentor(**aug_conf.speed)
        if aug_conf.volume is not None:
            self.volume_augment = VolumePerturbAugmentor(**aug_conf.volume)
        if aug_conf.noise is not None:
            self.noise_augment = NoisePerturbAugmentor(**aug_conf.noise)
        if aug_conf.reverb is not None:
            self.reverb_augment = ReverbPerturbAugmentor(**aug_conf.reverb)
        if aug_conf.spec_aug is not None:
            self.spec_augment = SpecAugmentor(**aug_conf.spec_aug)
        if aug_conf.spec_sub_aug is not None:
            self.spec_sub_augment = SpecSubAugmentor(**aug_conf.spec_sub_aug)

    # 音频增强
    def augment_audio(self, audio_segment):
        if self.speed_augment is not None:
            audio_segment = self.speed_augment(audio_segment)
        if self.volume_augment is not None:
            audio_segment = self.volume_augment(audio_segment)
        if self.noise_augment is not None:
            audio_segment = self.noise_augment(audio_segment)
        if self.reverb_augment is not None:
            audio_segment = self.reverb_augment(audio_segment)
        return audio_segment
