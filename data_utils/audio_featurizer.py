import kaldi_native_fbank as knf
import numpy as np


class AudioFeaturizer(object):
    """音频特征器

    :param sample_rate: 用于训练的音频的采样率
    :type sample_rate: int
    :param mode: 使用模式
    :type mode: str
    """

    def __init__(self, sample_rate=16000, mode="train"):
        self._target_sample_rate = sample_rate
        self._mode = mode
        self._opts = knf.FbankOptions()
        self._opts.energy_floor = 1.0
        if self._mode != "train":
            self._opts.frame_opts.dither = 0
        # 默认参数
        self._opts.frame_opts.samp_freq = sample_rate
        self._opts.mel_opts.num_bins = 80

    def featurize(self, waveform, sample_rate):
        """计算音频特征

        :param waveform: 音频数据
        :type waveform: np.ndarray
        :param sample_rate: 音频采样率
        :type sample_rate: int
        :return: 二维的音频特征
        :rtype: np.ndarray
        """
        if waveform.ndim != 1:
            assert waveform.ndim == 1, f'输入的音频数据必须是一维的，但是现在是{waveform.ndim}维'
        fbank_fn = knf.OnlineFbank(self._opts)
        # 计算音频特征
        fbank_fn.accept_waveform(sample_rate, waveform.tolist())
        frames = fbank_fn.num_frames_ready
        feature = np.empty([frames, self.feature_dim], dtype=np.float32)
        for i in range(fbank_fn.num_frames_ready):
            feature[i, :] = fbank_fn.get_frame(i)
        return feature

    @property
    def feature_dim(self):
        """返回特征大小

        :return: 特征大小
        :rtype: int
        """
        return 80
