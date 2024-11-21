import os

import kenlm
import numpy as np


class Scorer(object):
    """外部评分器评估波束搜索解码中的前缀或整个句子，包括n-gram语言模型的评分和单词计数

    :param alpha: 与语言模型相关的参数。当alpha=0时，不使用语言模型。
    :type alpha: float
    :param beta: 与单词计数相关的参数。当beta=0时，不使用单词计数。
    :type beta: float
    :model_path: 语言模型路径
    :type model_path: str
    """

    def __init__(self, alpha, beta, model_path):
        self._alpha = alpha
        self._beta = beta
        if not os.path.isfile(model_path):
            raise IOError("Invaid language model path: %s" % model_path)
        self._language_model = kenlm.LanguageModel(model_path)

    # n-gram语言模型评分
    def _language_model_score(self, sentence):
        # log10 prob of last word
        log_cond_prob = list(self._language_model.full_scores(sentence, eos=False))[-1][0]
        return np.power(10, log_cond_prob)

    @staticmethod
    def _word_count(sentence):
        words = sentence.strip().split(' ')
        return len(words)

    def reset_params(self, alpha, beta):
        self._alpha = alpha
        self._beta = beta

    def __call__(self, sentence, use_log=False):
        """评估函数，收集所有不同的得分并返回最终得分

        :param sentence: 用于计算的输入句子
        :type sentence: str
        :param use_log: 是否使用对数
        :type use_log: bool
        :return: 评价分数，用小数或对数表示
        :rtype: float
        """
        lm = self._language_model_score(sentence)
        word_cnt = self._word_count(sentence)
        if not use_log:
            score = np.power(lm, self._alpha) * np.power(word_cnt, self._beta)
        else:
            score = self._alpha * np.log(lm) + self._beta * np.log(word_cnt)
        return score
