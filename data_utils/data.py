import random

import numpy as np
import paddle
import paddle.static as static
from paddle.io import Dataset
import paddle.fluid as fluid
from data_utils.augmentor.augmentation import AugmentationPipeline
from data_utils.featurizer.speech_featurizer import SpeechFeaturizer
from data_utils.normalizer import FeatureNormalizer
from data_utils.speech import SpeechSegment
from data_utils.utility import read_manifest


class DataGenerator(Dataset):
    """
    DataGenerator provides basic audio data preprocessing pipeline, and offers
    data reader interfaces of PaddlePaddle requirements.
    :param vocab_filepath: Vocabulary filepath for indexing tokenized
                           transcripts.
    :type vocab_filepath: str
    :param mean_std_filepath: File containing the pre-computed mean and stddev.
    :type mean_std_filepath: None|str
    :param manifest_path: Filepath of manifest for audio files.
    :type manifest_path: str
    :param augmentation_config: Augmentation configuration in json string.
                                Details see AugmentationPipeline.__doc__.
    :type augmentation_config: str
    :param max_duration: Audio with duration (in seconds) greater than
                         this will be discarded.
    :type max_duration: float
    :param min_duration: Audio with duration (in seconds) smaller than
                         this will be discarded.
    :type min_duration: float
    :param stride_ms: Striding size (in milliseconds) for generating frames.
    :type stride_ms: float
    :param window_ms: Window size (in milliseconds) for generating frames.
    :type window_ms: float
    :param max_freq: Used when specgram_type is 'linear', only FFT bins
                     corresponding to frequencies between [0, max_freq] are
                     returned.
    :types max_freq: None|float
    :param use_dB_normalization: Whether to normalize the audio to -20 dB
                                before extracting the features.
    :type use_dB_normalization: bool
    :param random_seed: Random seed.
    :type random_seed: int
    :param keep_transcription_text: If set to True, transcription text will
                                    be passed forward directly without
                                    converting to index sequence.
    :type keep_transcription_text: bool
    :param place: The place to run the program.
    :type place: CPUPlace or CUDAPlace
    :param is_training: If set to True, generate text data for training,
                        otherwise,  generate text data for infer.
    :type is_training: bool
    """

    def __init__(self,
                 vocab_filepath,
                 mean_std_filepath,
                 manifest_path,
                 augmentation_config='{}',
                 max_duration=float('inf'),
                 min_duration=0.0,
                 stride_ms=10.0,
                 window_ms=20.0,
                 max_freq=None,
                 use_dB_normalization=True,
                 random_seed=0,
                 keep_transcription_text=False,
                 place=paddle.CPUPlace(),
                 is_training=True):
        super(DataGenerator, self).__init__()
        self._max_duration = max_duration
        self._min_duration = min_duration
        self._normalizer = FeatureNormalizer(mean_std_filepath)
        self._augmentation_pipeline = AugmentationPipeline(augmentation_config=augmentation_config,
                                                           random_seed=random_seed)
        self._speech_featurizer = SpeechFeaturizer(vocab_filepath=vocab_filepath,
                                                   stride_ms=stride_ms,
                                                   window_ms=window_ms,
                                                   max_freq=max_freq,
                                                   use_dB_normalization=use_dB_normalization)
        self._keep_transcription_text = keep_transcription_text
        self._epoch = 0
        self._is_training = is_training
        self._place = place
        # 读取数据列表
        self.manifest = read_manifest(manifest_path=manifest_path,
                                      max_duration=self._max_duration,
                                      min_duration=self._min_duration)
        # 将数据列表长到短排序
        if self._epoch == 0:
            self.manifest.sort(key=lambda x: x["duration"])

    def __getitem__(self, idx):
        instance = self.manifest[idx]
        audio_file, transcript = instance["audio_filepath"], instance["text"]
        speech_segment = SpeechSegment.from_file(audio_file, transcript)
        self._augmentation_pipeline.transform_audio(speech_segment)
        specgram, transcript_part = self._speech_featurizer.featurize(speech_segment, self._keep_transcription_text)
        specgram = self._normalizer.apply(specgram)
        return specgram, transcript_part

    def __len__(self):
        return len(self.manifest)

    @staticmethod
    def create_input():
        audio_data = paddle.static.data(name='audio_data',
                                        shape=[None, 39, None],
                                        dtype='float32',
                                        lod_level=0)
        text_data = paddle.static.data(name='text_data',
                                       shape=[None, 1],
                                       dtype='int32',
                                       lod_level=1)
        seq_len_data = paddle.static.data(name='seq_len_data',
                                          shape=[None, 1],
                                          dtype='int64',
                                          lod_level=0)
        masks = paddle.static.data(name='masks',
                                   shape=[None, 32, 20, None],
                                   dtype='float32',
                                   lod_level=0)
        return audio_data, text_data, seq_len_data, masks

    @property
    def feed_list(self):
        return self.create_input()

    @property
    def vocab_size(self):
        """返回词汇表大小
        :return: 词汇表大小
        :rtype: int
        """
        return self._speech_featurizer.vocab_size

    @property
    def vocab_list(self):
        """返回词汇表列表
        :return: 词汇表列表
        :rtype: list
        """
        return self._speech_featurizer.vocab_list

    def padding_batch(self, batch):
        """
        用零填充音频功能，使它们在同一个batch具有相同的形状(或一个用户定义的形状)
        """
        # 获取目标形状
        max_length = max([audio.shape[1] for audio, text in batch])
        # 填充操作
        padded_audios = []
        texts, text_lens = [], []
        audio_lens = []
        masks = []
        for audio, text in batch:
            padded_audio = np.zeros([audio.shape[0], max_length])
            padded_audio[:, :audio.shape[1]] = audio
            padded_audios.append(padded_audio)
            if self._is_training:
                texts += text
            else:
                texts.append(text)
            text_lens.append(len(text))
            audio_lens.append(audio.shape[1])
            mask_shape0 = (audio.shape[0] - 1) // 2 + 1
            mask_shape1 = (audio.shape[1] - 1) // 3 + 1
            mask_max_len = (max_length - 1) // 3 + 1
            mask_ones = np.ones((mask_shape0, mask_shape1))
            mask_zeros = np.zeros((mask_shape0, mask_max_len - mask_shape1))
            mask = np.repeat(np.reshape(np.concatenate((mask_ones, mask_zeros), axis=1),
                                        (1, mask_shape0, mask_max_len)), 32, axis=0)
            masks.append(mask)
        padded_audios = np.array(padded_audios).astype('float32')
        if self._is_training:
            texts = np.expand_dims(np.array(texts).astype('int32'), axis=-1)
            texts = fluid.create_lod_tensor(texts, recursive_seq_lens=[text_lens], place=self._place)
        audio_lens = np.array(audio_lens).astype('int64').reshape([-1, 1])
        masks = np.array(masks).astype('float32')
        return padded_audios, texts, audio_lens, masks
