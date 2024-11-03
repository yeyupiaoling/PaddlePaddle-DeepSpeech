import os
import sys

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
                 alpha=1.2,
                 beta=0.35,
                 lang_model_path=None,
                 beam_size=10,
                 cutoff_prob=1.0,
                 cutoff_top_n=40,
                 use_gpu=True,
                 num_threads=10):
        self.audio_featurizer = AudioFeaturizer(mode="infer")
        self.tokenizer = Tokenizer(vocab_dir)
        self.decoder = decoder
        self.use_gpu = use_gpu
        self.inv_normalizer = None
        self.__init_decoder(alpha, beta, beam_size, cutoff_prob, cutoff_top_n, self.tokenizer.vocab_list, lang_model_path)
        if not os.path.exists(model_path):
            raise Exception(f"模型文件不存在，请检查{model_path}是否存在！")
        sess_opt = SessionOptions()
        sess_opt.intra_op_num_threads = num_threads
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.session = InferenceSession(model_path, sess_options=sess_opt, providers=providers)
        # 预热
        warmup_audio_path = 'dataset/test.wav'
        if os.path.exists(warmup_audio_path):
            self.predict(warmup_audio_path, to_itn=False)
        else:
            print('预热文件不存在，忽略预热！', file=sys.stderr)

    def get_input_names(self):
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(self):
        return [v.name for v in self.session.get_outputs()]

    # 初始化解码器
    def __init_decoder(self, alpha, beta, beam_size, cutoff_prob, cutoff_top_n, vocab_list, lang_model_path):
        # 集束搜索方法的处理
        if self.decoder == "ctc_beam_search":
            try:
                from decoders.beam_search_decoder import BeamSearchDecoder
                self.beam_search_decoder = BeamSearchDecoder(alpha, beta, beam_size, cutoff_prob, cutoff_top_n,
                                                             vocab_list, language_model_path=lang_model_path)
            except ModuleNotFoundError:
                logger.warning('==================================================================')
                logger.warning('缺少 paddlespeech_ctcdecoders 库，请执行以下命令安装。')
                logger.warning(
                    'python -m pip install paddlespeech_ctcdecoders -U -i https://ppasr.yeyupiaoling.cn/pypi/simple/')
                logger.warning('【注意】现在已自动切换为ctc_greedy解码器，ctc_greedy解码器准确率相对较低。')
                logger.warning('==================================================================\n')
                self.decoder = 'ctc_greedy'

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
        audio_len = audio_feature.shape[1]
        audio_data = np.array(audio_feature).astype('float32')[np.newaxis, :]
        seq_len_data = np.array([audio_len]).astype('int64')

        input_dict = dict(zip(self.get_input_names(), [audio_data, seq_len_data]))
        print(input_dict)
        ctc_probs, ctc_lens = self.session.run(self.get_output_names(), input_dict)
        print(ctc_probs.shape, ctc_lens)

        if self.decoder == 'ctc_beam_search':
            # 集束搜索解码策略
            text = self.beam_search_decoder.decode_beam_search_offline(probs_split=ctc_probs)
        else:
            # 贪心解码策略
            out_tokens = ctc_greedy_search(ctc_probs=ctc_probs,
                                           ctc_lens=ctc_lens,
                                           blank_id=self.tokenizer.blank_id)
            text = self.tokenizer.ids2text([t for t in out_tokens])[0]
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
