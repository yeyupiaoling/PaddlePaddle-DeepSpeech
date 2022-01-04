"""Contains data generator for orgnaizing various audio data preprocessing
pipeline and offering data reader interface of PaddlePaddle requirements.
"""

import random
import numpy as np
import paddle
import paddle.fluid as fluid
from threading import local
from data_utils.utility import read_manifest
from data_utils.augmentor.augmentation import AugmentationPipeline
from data_utils.featurizer.speech_featurizer import SpeechFeaturizer
from data_utils.speech import SpeechSegment
from data_utils.normalizer import FeatureNormalizer


class DataGenerator(object):
    """
    DataGenerator provides basic audio data preprocessing pipeline, and offers
    data reader interfaces of PaddlePaddle requirements.

    :param vocab_filepath: Vocabulary filepath for indexing tokenized
                           transcripts.
    :type vocab_filepath: str
    :param mean_std_filepath: File containing the pre-computed mean and stddev.
    :type mean_std_filepath: None|str
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
                 augmentation_config='{}',
                 max_duration=float('inf'),
                 min_duration=0.0,
                 stride_ms=10.0,
                 window_ms=20.0,
                 use_dB_normalization=True,
                 random_seed=0,
                 keep_transcription_text=False,
                 place=paddle.CPUPlace(),
                 is_training=True):
        self._max_duration = max_duration
        self._min_duration = min_duration
        self._normalizer = FeatureNormalizer(mean_std_filepath)
        self._augmentation_pipeline = AugmentationPipeline(augmentation_config=augmentation_config,
                                                           random_seed=random_seed)
        self._speech_featurizer = SpeechFeaturizer(vocab_filepath=vocab_filepath,
                                                   stride_ms=stride_ms,
                                                   window_ms=window_ms,
                                                   use_dB_normalization=use_dB_normalization)
        self._rng = random.Random(random_seed)
        self._keep_transcription_text = keep_transcription_text
        self.epoch = 0
        self._is_training = is_training
        # for caching tar files info
        self._local_data = local()
        self._local_data.tar2info = {}
        self._local_data.tar2object = {}
        self._place = place

    def process_utterance(self, audio_file, transcript):
        """对语音数据加载、扩充、特征化和归一化

        :param audio_file: 音频文件的文件路径或文件对象
        :type audio_file: str | file
        :param transcript: 音频对应的文本
        :type transcript: str
        :return: 经过归一化等预处理的音频数据，音频文件对应文本的ID
        :rtype: tuple of (2darray, list)
        """
        speech_segment = SpeechSegment.from_file(audio_file, transcript)
        self._augmentation_pipeline.transform_audio(speech_segment)
        specgram, transcript_part = self._speech_featurizer.featurize(speech_segment, self._keep_transcription_text)
        specgram = self._normalizer.apply(specgram)
        specgram = self._augmentation_pipeline.transform_feature(specgram)
        return specgram, transcript_part

    def batch_reader_creator(self,
                             manifest_path,
                             batch_size,
                             padding_to=-1,
                             flatten=False,
                             shuffle_method="batch_shuffle"):
        """
        Batch data reader creator for audio data. Return a callable generator
        function to produce batches of data.

        Audio features within one batch will be padded with zeros to have the
        same shape, or a user-defined shape.

        :param manifest_path: Filepath of manifest for audio files.
        :type manifest_path: str
        :param batch_size: Number of instances in a batch.
        :type batch_size: int
        :param padding_to:  If set -1, the maximun shape in the batch
                            will be used as the target shape for padding.
                            Otherwise, `padding_to` will be the target shape.
        :type padding_to: int
        :param flatten: If set True, audio features will be flatten to 1darray.
        :type flatten: bool
        :param shuffle_method: Shuffle method. Options:
                                '' or None: no shuffle.
                                'instance_shuffle': instance-wise shuffle.
                                'batch_shuffle': similarly-sized instances are
                                                 put into batches, and then
                                                 batch-wise shuffle the batches.
                                                 For more details, please see
                                                 ``_batch_shuffle.__doc__``.
                                'batch_shuffle_clipped': 'batch_shuffle' with
                                                         head shift and tail
                                                         clipping. For more
                                                         details, please see
                                                         ``_batch_shuffle``.
                              If sortagrad is True, shuffle is disabled
                              for the first epoch.
        :type shuffle_method: None|str
        :return: Batch reader function, producing batches of data when called.
        :rtype: callable
        """

        def batch_reader():
            # 读取数据列表
            manifest = read_manifest(manifest_path=manifest_path,
                                     max_duration=self._max_duration,
                                     min_duration=self._min_duration)
            # 将数据列表长到短排序
            if self.epoch == 0:
                manifest.sort(key=lambda x: x["duration"], reverse=False)
            else:
                if shuffle_method == "batch_shuffle":
                    manifest = self._batch_shuffle(manifest, batch_size, clipped=False)
                elif shuffle_method == "batch_shuffle_clipped":
                    manifest = self._batch_shuffle(manifest, batch_size, clipped=True)
                elif shuffle_method == "instance_shuffle":
                    self._rng.shuffle(manifest)
                elif shuffle_method is None:
                    pass
                else:
                    raise ValueError("Unknown shuffle method %s." % shuffle_method)
            # 准备批量数据
            batch = []
            instance_reader = self._instance_reader_creator(manifest)

            for instance in instance_reader():
                batch.append(instance)
                if len(batch) == batch_size:
                    yield self._padding_batch(batch, padding_to, flatten)
                    batch = []
            if len(batch) >= 1:
                yield self._padding_batch(batch, padding_to, flatten)
            self.epoch += 1

        return batch_reader

    @property
    def feeding(self):
        """返回数据读取器的exe读取字典

        :return: 数据读取字典
        :rtype: dict
        """
        feeding_dict = {"audio_spectrogram": 0, "transcript_text": 1}
        return feeding_dict

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

    def _instance_reader_creator(self, manifest):
        """
        创建一个数据生成器reader

        Instance: 生成器得到的数据是一个元组，包含了经过预处理音频数据和音频对应文本的ID
        """

        def reader():
            for instance in manifest:
                inst = self.process_utterance(instance["audio_filepath"], instance["text"])
                yield inst

        return reader

    def _padding_batch(self, batch, padding_to=-1, flatten=False):
        """
        用零填充音频功能，使它们在同一个batch具有相同的形状(或一个用户定义的形状)

        如果padding_to为-1，则批处理中的最大形状将被使用 作为填充的目标形状。
        否则，' padding_to '将是目标形状(仅指第二轴)。

        如果“flatten”为True，特征将被flatten为一维数据
        """
        # 获取目标形状
        max_length = max([audio.shape[1] for audio, text in batch])
        if padding_to != -1:
            if padding_to < max_length:
                raise ValueError("如果padding_to不是-1，它应该大于批处理中任何实例的形状")
            max_length = padding_to
        # 填充操作
        padded_audios = []
        texts, text_lens = [], []
        audio_lens = []
        masks = []
        for audio, text in batch:
            padded_audio = np.zeros([audio.shape[0], max_length])
            padded_audio[:, :audio.shape[1]] = audio
            if flatten:
                padded_audio = padded_audio.flatten()
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
            mask = np.repeat(
                np.reshape(np.concatenate((mask_ones, mask_zeros), axis=1),
                           (1, mask_shape0, mask_max_len)), 32, axis=0)
            masks.append(mask)
        padded_audios = np.array(padded_audios).astype('float32')
        if self._is_training:
            texts = np.expand_dims(np.array(texts).astype('int32'), axis=-1)
            texts = fluid.create_lod_tensor(texts, recursive_seq_lens=[text_lens], place=self._place)
        audio_lens = np.array(audio_lens).astype('int64').reshape([-1, 1])
        masks = np.array(masks).astype('float32')
        return padded_audios, texts, audio_lens, masks

    def _batch_shuffle(self, manifest, batch_size, clipped=False):
        """将大小相似的实例放入小批量中可以提高效率，并进行批量打乱

        1. 按持续时间对音频剪辑进行排序
        2. 生成一个随机数k， k的范围[0,batch_size)
        3. 随机移动k实例，为不同的epoch训练创建不同的批次
        4. 打乱minibatches.

        :param manifest: 数据列表
        :type manifest: list
        :param batch_size: 批量大小。这个大小还用于为批量洗牌生成一个随机数。
        :type batch_size: int
        :param clipped: 是否剪辑头部(小移位)和尾部(不完整批处理)实例。
        :type clipped: bool
        :return: Batch shuffled mainifest.
        :rtype: list
        """
        shift_len = self._rng.randint(0, batch_size - 1)
        batch_manifest = list(zip(*[iter(manifest[shift_len:])] * batch_size))
        self._rng.shuffle(batch_manifest)
        batch_manifest = [item for batch in batch_manifest for item in batch]
        if not clipped:
            res_len = len(manifest) - shift_len - len(batch_manifest)
            batch_manifest.extend(manifest[-res_len:])
            batch_manifest.extend(manifest[0:shift_len])
        return batch_manifest
