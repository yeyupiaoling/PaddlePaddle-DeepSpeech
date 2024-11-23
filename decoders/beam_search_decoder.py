import os

import numpy as np
import paddlespeech_ctcdecoders
import yaml
from loguru import logger

from utils.utils import download, print_arguments, dict_to_object


class BeamSearchDecoder:
    def __init__(self, conf_path, vocab_list, blank_id=0):
        # 读取数据增强配置文件
        with open(conf_path, 'r', encoding='utf-8') as f:
            decoder_configs = yaml.load(f.read(), Loader=yaml.FullLoader)
        print_arguments(configs=decoder_configs, title='集束搜索解码器参数')
        self.configs = dict_to_object(decoder_configs)
        self.vocab_list = vocab_list
        self.blank_id = blank_id
        if (not os.path.exists(self.configs.language_model_path) and
                self.configs.language_model_path == 'lm/zh_giga.no_cna_cmn.prune01244.klm'):
            logger.info('=' * 70)
            language_model_url = 'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm'
            logger.info("语言模型不存在，正在下载，下载地址： %s ..." % language_model_url)
            os.makedirs(os.path.dirname(self.configs.language_model_path), exist_ok=True)
            download(url=language_model_url, download_target=self.configs.language_model_path)
            logger.info('=' * 70)
        logger.info('=' * 70)
        logger.info("初始化解码器...")
        assert os.path.exists(self.configs.language_model_path), f'语言模型不存在：{self.configs.language_model_path}'
        self._ext_scorer = Scorer(self.configs.alpha, self.configs.beta, self.configs.language_model_path, vocab_list)
        lm_char_based = self._ext_scorer.is_character_based()
        lm_max_order = self._ext_scorer.get_max_order()
        lm_dict_size = self._ext_scorer.get_dict_size()
        logger.info(f"language model: "
                    f"model path = {self.configs.language_model_path}, "
                    f"is_character_based = {lm_char_based}, "
                    f"max_order = {lm_max_order}, "
                    f"dict_size = {lm_dict_size}")
        logger.info("初始化解码器完成!")
        logger.info('=' * 70)

    # 单个数据解码
    def ctc_beam_search_decoder(self, ctc_probs):
        if not isinstance(ctc_probs, list):
            ctc_probs = ctc_probs.tolist()
        # beam search decode
        beam_search_result = ctc_beam_search_decoding(probs_seq=ctc_probs,
                                                      vocabulary=self.vocab_list,
                                                      beam_size=self.configs.beam_size,
                                                      ext_scoring_func=self._ext_scorer,
                                                      cutoff_prob=self.configs.cutoff_prob,
                                                      cutoff_top_n=self.configs.cutoff_top_n,
                                                      blank_id=self.blank_id)
        result = beam_search_result[0][1].replace('▁', ' ').strip()
        return result

    # 一批数据解码
    def ctc_beam_search_decoder_batch(self, ctc_probs, ctc_lens):
        if not isinstance(ctc_probs, list):
            ctc_probs = ctc_probs.tolist()
            ctc_lens = ctc_lens.tolist()
        new_ctc_probs = []
        for i, ctc_prob in enumerate(ctc_probs):
            ctc_prob = ctc_prob[:ctc_lens[i]]
            new_ctc_probs.append(ctc_prob)
        # beam search decode
        self.num_processes = min(self.configs.num_processes, len(ctc_probs))
        beam_search_results = ctc_beam_search_decoding_batch(probs_split=new_ctc_probs,
                                                             vocabulary=self.vocab_list,
                                                             beam_size=self.configs.beam_size,
                                                             num_processes=self.num_processes,
                                                             ext_scoring_func=self._ext_scorer,
                                                             cutoff_prob=self.configs.cutoff_prob,
                                                             cutoff_top_n=self.configs.cutoff_top_n,
                                                             blank_id=self.blank_id)
        results = [result[0][1].replace('▁', ' ').strip() for result in beam_search_results]
        return results

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)


class Scorer(paddlespeech_ctcdecoders.Scorer):
    """Wrapper for Scorer.

    :param alpha: 与语言模型相关的参数。当alpha = 0时不要使用语言模型
    :type alpha: float
    :param beta: 与字计数相关的参数。当beta = 0时不要使用统计字
    :type beta: float
    :model_path: 语言模型的路径
    :type model_path: str
    :param vocabulary: 词汇列表
    :type vocabulary: list
    """

    def __init__(self, alpha, beta, model_path, vocabulary):
        paddlespeech_ctcdecoders.Scorer.__init__(self, alpha, beta, model_path, vocabulary)


def ctc_beam_search_decoding(probs_seq,
                             vocabulary,
                             beam_size,
                             cutoff_prob=1.0,
                             cutoff_top_n=40,
                             blank_id=0,
                             ext_scoring_func=None):
    """集束搜索解码器

    :param probs_seq: 单个2-D概率分布列表，每个元素是词汇表和空白上的标准化概率列表
    :type probs_seq: 2-D list
    :param vocabulary: 词汇列表
    :type vocabulary: list
    :param beam_size: 集束搜索宽度
    :type beam_size: int
    :param cutoff_prob: 剪枝中的截断概率，默认1.0，没有剪枝
    :type cutoff_prob: float
    :param cutoff_top_n: 剪枝时的截断数，仅在词汇表中具有最大probs的cutoff_top_n字符用于光束搜索，默认为40
    :type cutoff_top_n: int
    :param blank_id 空白索引
    :type blank_id int
    :param ext_scoring_func: 外部评分功能部分解码句子，如字计数或语言模型
    :type ext_scoring_func: callable
    :return: 解码结果为log概率和句子的元组列表，按概率降序排列
    :rtype: list
    """
    beam_results = paddlespeech_ctcdecoders.ctc_beam_search_decoding(
        probs_seq, vocabulary, beam_size, cutoff_prob, cutoff_top_n, ext_scoring_func, blank_id)
    beam_results = [(res[0], res[1]) for res in beam_results]
    return beam_results


def ctc_beam_search_decoding_batch(probs_split,
                                   vocabulary,
                                   beam_size,
                                   num_processes,
                                   cutoff_prob=1.0,
                                   cutoff_top_n=40,
                                   blank_id=0,
                                   ext_scoring_func=None):
    """Wrapper for the batched CTC beam search decoder.

    :param probs_split: 3-D列表，每个元素作为ctc_beam_search_decoder()使用的2-D概率列表的实例
    :type probs_split: 3-D list
    :param vocabulary: 词汇列表
    :type vocabulary: list
    :param beam_size: 集束搜索宽度
    :type beam_size: int
    :param cutoff_prob: 剪枝中的截断概率，默认1.0，没有剪枝
    :type cutoff_prob: float
    :param cutoff_top_n: 剪枝时的截断数，仅在词汇表中具有最大probs的cutoff_top_n字符用于光束搜索，默认为40
    :type cutoff_top_n: int
    :param blank_id 空白索引
    :type blank_id int
    :param num_processes: 并行解码进程数
    :type num_processes: int
    :param ext_scoring_func: 外部评分功能部分解码句子，如字计数或语言模型
    :type ext_scoring_func: callable
    :return: 解码结果为log概率和句子的元组列表，按概率降序排列的列表
    :rtype: list
    """

    batch_beam_results = paddlespeech_ctcdecoders.ctc_beam_search_decoding_batch(
        probs_split, vocabulary, beam_size, num_processes, cutoff_prob,
        cutoff_top_n, ext_scoring_func, blank_id)
    batch_beam_results = [[(res[0], res[1]) for res in beam_results]
                          for beam_results in batch_beam_results]
    return batch_beam_results
