"""Contains the audio featurizer class."""

import numpy as np

from python_speech_features import mfcc
from python_speech_features import delta
from data_utils.audio import AudioSegment


class AudioFeaturizer(object):
    """音频特征器,用于从AudioSegment或SpeechSegment内容中提取特性。

    Currently, it supports feature types of linear spectrogram and mfcc.

    :param stride_ms: Striding size (in milliseconds) for generating frames.
    :type stride_ms: float
    :param window_ms: Window size (in milliseconds) for generating frames.
    :type window_ms: float
    :param max_freq: When specgram_type is 'linear', only FFT bins
                     corresponding to frequencies between [0, max_freq] are
                     returned; when specgram_type is 'mfcc', max_feq is the
                     highest band edge of mel filters.
    :types max_freq: None|float
    :param target_sample_rate: Audio are resampled (if upsampling or
                               downsampling is allowed) to this before
                               extracting spectrogram features.
    :type target_sample_rate: int
    :param use_dB_normalization: Whether to normalize the audio to a certain
                                 decibels before extracting the features.
    :type use_dB_normalization: bool
    :param target_dB: Target audio decibels for normalization.
    :type target_dB: float
    """

    def __init__(self,
                 stride_ms=10.0,
                 window_ms=20.0,
                 max_freq=None,
                 target_sample_rate=16000,
                 use_dB_normalization=True,
                 target_dB=-20):
        self._stride_ms = stride_ms
        self._window_ms = window_ms
        self._max_freq = max_freq
        self._target_sample_rate = target_sample_rate
        self._use_dB_normalization = use_dB_normalization
        self._target_dB = target_dB

    def featurize(self,
                  audio_segment,
                  allow_downsampling=True,
                  allow_upsampling=True):
        """从AudioSegment或SpeechSegment中提取音频特征

        :param audio_segment: Audio/speech segment to extract features from.
        :type audio_segment: AudioSegment|SpeechSegment
        :param allow_downsampling: Whether to allow audio downsampling before
                                   featurizing.
        :type allow_downsampling: bool
        :param allow_upsampling: Whether to allow audio upsampling before
                                 featurizing.
        :type allow_upsampling: bool
        :return: Spectrogram audio feature in 2darray.
        :rtype: ndarray
        :raises ValueError: If audio sample rate is not supported.
        """
        # upsampling or downsampling
        if ((audio_segment.sample_rate > self._target_sample_rate and
             allow_downsampling) or
                (audio_segment.sample_rate < self._target_sample_rate and
                 allow_upsampling)):
            audio_segment.resample(self._target_sample_rate)
        if audio_segment.sample_rate != self._target_sample_rate:
            raise ValueError("Audio sample rate is not supported. "
                             "Turn allow_downsampling or allow up_sampling on.")
        # decibel normalization
        if self._use_dB_normalization:
            audio_segment.normalize(target_db=self._target_dB)
        # extract spectrogram
        return self._compute_mfcc(audio_segment.samples, audio_segment.sample_rate,
                                  self._stride_ms, self._window_ms, self._max_freq)

    # 计算音频梅尔频谱倒谱系数（MFCCs）
    def _compute_mfcc(self,
                      samples,
                      sample_rate,
                      stride_ms=10.0,
                      window_ms=20.0,
                      max_freq=None):
        """Compute mfcc from samples."""
        if max_freq is None:
            max_freq = sample_rate / 2
        if max_freq > sample_rate / 2:
            raise ValueError("max_freq must not be greater than half of sample rate.")
        if stride_ms > window_ms:
            raise ValueError("Stride size must not be greater than window size.")
        # compute the 13 cepstral coefficients, and the first one is replaced
        # by log(frame energy)
        mfcc_feat = mfcc(signal=samples,
                         samplerate=sample_rate,
                         winlen=0.001 * window_ms,
                         winstep=0.001 * stride_ms,
                         highfreq=max_freq)
        # Deltas
        d_mfcc_feat = delta(mfcc_feat, 2)
        # Deltas-Deltas
        dd_mfcc_feat = delta(d_mfcc_feat, 2)
        # transpose
        mfcc_feat = np.transpose(mfcc_feat)
        d_mfcc_feat = np.transpose(d_mfcc_feat)
        dd_mfcc_feat = np.transpose(dd_mfcc_feat)
        # concat above three features
        concat_mfcc_feat = np.concatenate((mfcc_feat, d_mfcc_feat, dd_mfcc_feat))
        return concat_mfcc_feat
