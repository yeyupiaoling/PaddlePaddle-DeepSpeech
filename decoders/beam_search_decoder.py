import os
import multiprocessing
from math import log

import yaml

from decoders.scorer import Scorer
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
            print('=' * 70)
            language_model_url = 'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm'
            print("语言模型不存在，正在下载，下载地址： %s ..." % language_model_url)
            os.makedirs(os.path.dirname(self.configs.language_model_path), exist_ok=True)
            download(url=language_model_url, download_target=self.configs.language_model_path)
            print('=' * 70)
        print('=' * 70)
        print("初始化解码器...")
        assert os.path.exists(self.configs.language_model_path), f'语言模型不存在：{self.configs.language_model_path}'
        self._ext_scorer = Scorer(self.configs.alpha, self.configs.beta, self.configs.language_model_path)
        print("初始化解码器完成!")
        print('=' * 70)

    def ctc_beam_search_decoder(self, ctc_probs):
        if not isinstance(ctc_probs, list):
            ctc_probs = ctc_probs.tolist()
        prefix_set_prev = {'\t': 1.0}
        probs_b_prev, probs_nb_prev = {'\t': 1.0}, {'\t': 0.0}

        # extend prefix in loop
        for time_step in range(len(ctc_probs)):
            prefix_set_next, probs_b_cur, probs_nb_cur = {}, {}, {}

            prob_idx = list(enumerate(ctc_probs[time_step]))
            cutoff_len = len(prob_idx)
            # If pruning is enabled
            if self.configs.cutoff_prob < 1.0 or self.configs.cutoff_top_n < cutoff_len:
                prob_idx = sorted(prob_idx, key=lambda asd: asd[1], reverse=True)
                cutoff_len, cum_prob = 0, 0.0
                for i in range(len(prob_idx)):
                    cum_prob += prob_idx[i][1]
                    cutoff_len += 1
                    if cum_prob >= self.configs.cutoff_prob:
                        break
                cutoff_len = min(cutoff_len, self.configs.cutoff_top_n)
                prob_idx = prob_idx[0:cutoff_len]

            for l in prefix_set_prev:
                if l not in prefix_set_next:
                    probs_b_cur[l], probs_nb_cur[l] = 0.0, 0.0

                # extend prefix by travering prob_idx
                for index in range(cutoff_len):
                    c, prob_c = prob_idx[index][0], prob_idx[index][1]

                    if c == self.blank_id:
                        probs_b_cur[l] += prob_c * (probs_b_prev[l] + probs_nb_prev[l])
                    else:
                        last_char = l[-1]
                        new_char = self.vocab_list[c]
                        l_plus = l + new_char
                        if l_plus not in prefix_set_next:
                            probs_b_cur[l_plus], probs_nb_cur[l_plus] = 0.0, 0.0

                        if new_char == last_char:
                            probs_nb_cur[l_plus] += prob_c * probs_b_prev[l]
                            probs_nb_cur[l] += prob_c * probs_nb_prev[l]
                        elif new_char == ' ':
                            if len(l) == 1:
                                score = 1.0
                            else:
                                prefix = l[1:]
                                score = self._ext_scorer(prefix)
                            probs_nb_cur[l_plus] += score * prob_c * (probs_b_prev[l] + probs_nb_prev[l])
                        else:
                            probs_nb_cur[l_plus] += prob_c * (probs_b_prev[l] + probs_nb_prev[l])
                        # add l_plus into prefix_set_next
                        prefix_set_next[l_plus] = probs_nb_cur[l_plus] + probs_b_cur[l_plus]
                # add l into prefix_set_next
                prefix_set_next[l] = probs_b_cur[l] + probs_nb_cur[l]
            # update probs
            probs_b_prev, probs_nb_prev = probs_b_cur, probs_nb_cur

            # store top beam_size prefixes
            prefix_set_prev = sorted(prefix_set_next.items(), key=lambda asd: asd[1], reverse=True)
            if self.configs.beam_size < len(prefix_set_prev):
                prefix_set_prev = prefix_set_prev[:self.configs.beam_size]
            prefix_set_prev = dict(prefix_set_prev)

        beam_result = []
        for seq, prob in prefix_set_prev.items():
            if prob > 0.0 and len(seq) > 1:
                result = seq[1:]
                # score last word by external scorer
                if result[-1] != ' ':
                    prob = prob * self._ext_scorer(result)
                log_prob = log(prob)
                beam_result.append((log_prob, result))
            else:
                beam_result.append((float('-inf'), ''))

        # output top beam_size decoding results
        beam_result = sorted(beam_result, key=lambda asd: asd[0], reverse=True)
        return beam_result[0][1]

    def ctc_beam_search_decoder_batch(self, ctc_probs, ctc_lens):
        if not isinstance(ctc_probs, list):
            ctc_probs = ctc_probs.tolist()
            ctc_lens = ctc_lens.tolist()
        assert len(ctc_probs) == len(ctc_lens)
        pool = multiprocessing.Pool(processes=self.configs.num_processes)
        results = []
        for i, ctc_prob in enumerate(ctc_probs):
            ctc_prob = ctc_prob[:ctc_lens[i]]
            results.append(pool.apply_async(self.ctc_beam_search_decoder, args=(ctc_prob,)))
        pool.close()
        pool.join()
        beam_search_results = [result.get() for result in results]
        return beam_search_results
