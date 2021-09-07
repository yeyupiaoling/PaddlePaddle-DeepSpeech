from data_utils.featurizer.speech_featurizer import SpeechFeaturizer
from data_utils.normalizer import FeatureNormalizer
from data_utils.speech import SpeechSegment


class AudioInferProcess(object):
    """
    识别程序所使用的是对音频预处理的工具

    :param vocab_filepath: 词汇表文件路径
    :type vocab_filepath: str
    :param mean_std_filepath: 平均值和标准差的文件路径
    :type mean_std_filepath: str
    :param stride_ms: 生成帧的跨步大小(以毫秒为单位)
    :type stride_ms: float
    :param window_ms: 用于生成帧的窗口大小(毫秒)
    :type window_ms: float
    :param use_dB_normalization: 提取特征前是否将音频归一化至-20 dB
    :type use_dB_normalization: bool
    """

    def __init__(self,
                 vocab_filepath,
                 mean_std_filepath,
                 stride_ms=10.0,
                 window_ms=20.0,
                 use_dB_normalization=True):
        self._normalizer = FeatureNormalizer(mean_std_filepath)
        self._speech_featurizer = SpeechFeaturizer(vocab_filepath=vocab_filepath,
                                                   stride_ms=stride_ms,
                                                   window_ms=window_ms,
                                                   use_dB_normalization=use_dB_normalization)

    def process_utterance(self, audio_file):
        """对语音数据加载、预处理

        :param audio_file: 音频文件的文件路径或文件对象
        :type audio_file: str | file
        :return: 预处理的音频数据
        :rtype: 2darray
        """
        speech_segment = SpeechSegment.from_file(audio_file, "")
        specgram, _ = self._speech_featurizer.featurize(speech_segment, False)
        specgram = self._normalizer.apply(specgram)
        return specgram

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
