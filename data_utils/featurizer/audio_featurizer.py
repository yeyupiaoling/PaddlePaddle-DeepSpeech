import numpy as np
import paddle
from paddleaudio.compliance.kaldi import fbank


class AudioFeaturizer(object):
    """音频特征器

    :param sample_rate: 用于训练的音频的采样率
    :type sample_rate: int
    :param use_dB_normalization: 是否对音频进行音量归一化
    :type use_dB_normalization: bool
    :param target_dB: 对音频进行音量归一化的音量分贝值
    :type target_dB: float
    :param mode: 使用模式
    :type mode: str
    """

    def __init__(self,
                 sample_rate=16000,
                 use_dB_normalization=True,
                 target_dB=-20,
                 mode="train"):
        self._target_sample_rate = sample_rate
        self._use_dB_normalization = use_dB_normalization
        self._target_dB = target_dB
        self._mode = mode

    def featurize(self, audio_segment):
        """从AudioSegment中提取音频特征

        :param audio_segment: Audio segment to extract features from.
        :type audio_segment: AudioSegment
        :return: Spectrogram audio feature in 2darray.
        :rtype: ndarray
        """
        # upsampling or downsampling
        if audio_segment.sample_rate != self._target_sample_rate:
            audio_segment.resample(self._target_sample_rate)
        # decibel normalization
        if self._use_dB_normalization:
            audio_segment.normalize(target_db=self._target_dB)
        dither = 0.1 if self._mode == "train" else 0.0
        waveform = paddle.to_tensor(np.expand_dims(audio_segment.samples, 0), dtype=paddle.float32)
        # 计算Fbank
        mat = fbank(waveform,
                    n_mels=80,
                    frame_shift=10,
                    frame_length=25,
                    dither=dither,
                    sr=audio_segment.sample_rate)
        fbank_feat = mat.numpy()
        return fbank_feat

    @property
    def feature_dim(self):
        """返回特征大小

        :return: 特征大小
        :rtype: int
        """
        return 80
