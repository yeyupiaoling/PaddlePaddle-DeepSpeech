import os
import sys

import numpy as np
import paddle.inference as paddle_infer
from loguru import logger
from yeaudio.audio import AudioSegment

from data_utils.audio_featurizer import AudioFeaturizer
from data_utils.tokenizer import Tokenizer
from decoders.ctc_greedy_search import ctc_greedy_search


class Predictor:
    def __init__(self,
                 model_dir,
                 vocab_dir,
                 decoder='ctc_greedy',
                 beam_search_conf=None,
                 use_gpu=True,
                 gpu_mem=500,
                 enable_mkldnn=False,
                 num_threads=10):
        self.audio_featurizer = AudioFeaturizer(mode="infer")
        self.tokenizer = Tokenizer(vocab_dir)
        self.decoder = decoder
        self.use_gpu = use_gpu
        self.inv_normalizer = None
        # 创建 config
        model_path = os.path.join(model_dir, 'model.pdmodel')
        params_path = os.path.join(model_dir, 'model.pdiparams')
        if not os.path.exists(model_path) or not os.path.exists(params_path):
            raise Exception(f"模型文件不存在，请检查{model_path}和{params_path}是否存在！")
        self.config = paddle_infer.Config(model_path, params_path)
        self.config.enable_use_gpu(1000, 0)
        self.config.enable_memory_optim()
        if self.use_gpu:
            self.config.enable_use_gpu(gpu_mem, 0)
        else:
            self.config.disable_gpu()
            self.config.set_cpu_math_library_num_threads(num_threads)
            if enable_mkldnn:
                self.config.set_mkldnn_cache_capacity(10)
                self.config.enable_mkldnn()

        # enable memory optim
        self.config.enable_memory_optim()
        self.config.disable_glog_info()

        # 根据 config 创建 predictor
        self.predictor = paddle_infer.create_predictor(self.config)

        # 获取输入层
        self.speech_handle = self.predictor.get_input_handle('speech')
        self.speech_lengths_handle = self.predictor.get_input_handle('speech_lengths')

        # 获取输出的名称
        self.output_names = self.predictor.get_output_names()

        # 初始化解码器
        vocab_list = self.tokenizer.vocab_list
        self.__init_decoder(beam_search_conf)
        # 预热
        warmup_audio_path = 'dataset/test.wav'
        if os.path.exists(warmup_audio_path):
            self.predict(warmup_audio_path, to_itn=False)
        else:
            logger.warning('预热文件不存在，忽略预热！', file=sys.stderr)

    # 初始化解码器
    def __init_decoder(self, beam_search_conf):
        # 集束搜索方法的处理
        if self.decoder == "ctc_beam_search":
            from decoders.beam_search_decoder import BeamSearchDecoder
            self.beam_search_decoder = BeamSearchDecoder(conf_path=beam_search_conf,
                                                         vocab_list=self.tokenizer.vocab_list,
                                                         blank_id=self.tokenizer.blank_id)

    # 预测音频
    def predict(self, audio_path, to_itn=False):
        audio_segment = AudioSegment.from_file(audio_path)
        if audio_segment.sample_rate != 16000:
            audio_segment.resample(16000)
        if audio_segment.duration <= 30:
            text = self._infer(audio_segment, to_itn=to_itn)
            return text
        else:
            logger.info('音频时长超过30秒，将进行分段识别:')
            # 获取语音活动区域
            speech_timestamps = audio_segment.vad()
            sentences = []
            last_audio_ndarray = None
            for i, t in enumerate(speech_timestamps):
                audio_ndarray = audio_segment.samples[int(t['start']): int(t['end'])]
                if last_audio_ndarray is not None:
                    audio_ndarray = np.concatenate((last_audio_ndarray, audio_ndarray))
                    last_audio_ndarray = None
                audio_segment_part = AudioSegment(audio_ndarray, sample_rate=audio_segment.sample_rate)
                if audio_segment_part.duration < 1.0:
                    last_audio_ndarray = audio_ndarray
                    continue
                # 执行识别
                text = self._infer(audio_segment_part, to_itn=to_itn)
                logger.info(f'第{i + 1}段识别结果: {text}')
                if len(text) > 0:
                    sentences.append(text)
            texts = '，'.join(sentences)
            return texts

    def _infer(self, audio_segment, to_itn=False):
        # 进行预处理
        audio_feature = self.audio_featurizer.featurize(audio_segment.samples, audio_segment.sample_rate)
        audio_len = audio_feature.shape[0]
        audio_data = np.array(audio_feature).astype('float32')[np.newaxis, :]
        seq_len_data = np.array([audio_len]).astype('int64')

        # 设置输入
        self.speech_handle.reshape([audio_data.shape[0], audio_data.shape[1], audio_data.shape[2]])
        self.speech_lengths_handle.reshape([audio_data.shape[0]])
        self.speech_handle.copy_from_cpu(audio_data)
        self.speech_lengths_handle.copy_from_cpu(seq_len_data)

        # 运行predictor
        self.predictor.run()

        # 获取输出
        ctc_probs_handle = self.predictor.get_output_handle(self.output_names[0])
        ctc_probs_data = ctc_probs_handle.copy_to_cpu()
        ctc_lens_handle = self.predictor.get_output_handle(self.output_names[1])
        ctc_lens_data = ctc_lens_handle.copy_to_cpu()

        # 执行解码
        if self.decoder == 'ctc_beam_search':
            # 集束搜索解码策略
            text = self.beam_search_decoder.ctc_beam_search_decoder(ctc_probs=ctc_probs_data[0])
        else:
            # 贪心解码策略
            out_tokens = ctc_greedy_search(ctc_probs=ctc_probs_data[0],
                                           blank_id=self.tokenizer.blank_id)
            text = self.tokenizer.ids2text(out_tokens)
        # 是否逆文本标准化
        if to_itn:
            if self.inv_normalizer is None:
                # 需要安装WeTextProcessing>=1.0.4.1
                from itn.chinese.inverse_normalizer import InverseNormalizer
                user_dir = os.path.expanduser('~')
                cache_dir = os.path.join(user_dir, '.cache/itn_v1.0.4.1')
                exists = os.path.exists(os.path.join(cache_dir, 'zh_itn_tagger.fst')) and \
                         os.path.exists(os.path.join(cache_dir, 'zh_itn_verbalizer.fst'))
                self.inv_normalizer = InverseNormalizer(cache_dir=cache_dir, enable_0_to_9=False,
                                                        overwrite_cache=not exists)
            text = self.inv_normalizer.normalize(text)
        return text
