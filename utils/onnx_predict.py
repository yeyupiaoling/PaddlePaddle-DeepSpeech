import os

import numpy as np
from loguru import logger
from onnxruntime import InferenceSession, SessionOptions
from yeaudio.audio import AudioSegment

from data_utils.audio_featurizer import AudioFeaturizer
from data_utils.tokenizer import Tokenizer
from decoders.ctc_greedy_search import ctc_greedy_search


class ONNXPredictor:
    def __init__(self,
                 model_path,
                 vocab_dir,
                 decoder='ctc_greedy',
                 beam_search_conf=None,
                 use_gpu=True,
                 num_threads=10):
        self.audio_featurizer = AudioFeaturizer(mode="infer")
        self.tokenizer = Tokenizer(vocab_dir)
        self.decoder = decoder
        self.use_gpu = use_gpu
        self.inv_normalizer = None
        self.__init_decoder(beam_search_conf)
        if not os.path.exists(model_path):
            raise Exception(f"模型文件不存在，请检查{model_path}是否存在！")
        sess_opt = SessionOptions()
        sess_opt.intra_op_num_threads = num_threads
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.session = InferenceSession(model_path, sess_options=sess_opt, providers=providers)
        if self.use_gpu and 'CUDAExecutionProvider' not in self.session.get_providers():
            logger.warning(f'当前无法使用GPU推理，请确认您的环境支持GPU加速！已自动切换到CPU推理！')
        # 预热
        warmup_audio_path = 'dataset/test.wav'
        if os.path.exists(warmup_audio_path):
            self.predict(warmup_audio_path, to_itn=False)
        else:
            logger.warning('预热文件不存在，忽略预热！')

    def get_input_names(self):
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(self):
        return [v.name for v in self.session.get_outputs()]

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
                audio_ndarray = audio_segment.samples[t['start']: t['end']]
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

        input_dict = dict(zip(self.get_input_names(), [audio_data, seq_len_data]))
        ctc_probs, ctc_lens = self.session.run(self.get_output_names(), input_dict)

        if self.decoder == 'ctc_beam_search':
            # 集束搜索解码策略
            text = self.beam_search_decoder.ctc_beam_search_decoder(ctc_probs=ctc_probs[0])
        else:
            # 贪心解码策略
            out_tokens = ctc_greedy_search(ctc_probs=ctc_probs[0],
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
