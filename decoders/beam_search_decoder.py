import math
import os
from typing import List, Tuple, Optional, Dict

import kenlm
import numpy as np
import yaml
from loguru import logger

from utils.utils import download, print_arguments, dict_to_object

# 常量定义
NUM_FLT_INF = float('-inf')
NUM_FLT_MIN = 1e-10
kSPACE = " "
tSPACE = " "
START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK_TOKEN = "<unk>"
OOV_SCORE = -1000.0


def log_sum_exp(a: float, b: float) -> float:
    """计算 log(exp(a) + exp(b)) 的数值稳定版本"""
    if a == NUM_FLT_INF:
        return b
    if b == NUM_FLT_INF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))


def get_utf8_str_len(s: str) -> int:
    """获取 UTF-8 字符串的字符长度"""
    if not s:
        return 0
    count = 0
    for c in s:
        if (ord(c) & 0xc0) != 0x80:
            count += 1
    return count


def split_utf8_str(s: str) -> List[str]:
    """按 UTF-8 字符分割字符串"""
    result = []
    current_char = ""
    for c in s:
        if (ord(c) & 0xc0) != 0x80:  # 新的 UTF-8 字符开始
            if current_char:
                result.append(current_char)
            current_char = c
        else:
            current_char += c
    if current_char:
        result.append(current_char)
    return result


def split_str(s: str, delim: str) -> List[str]:
    """按分隔符分割字符串"""
    if not delim:
        return [s]
    result = []
    start = 0
    while True:
        end = s.find(delim, start)
        if end == -1:
            if start < len(s):
                result.append(s[start:])
            break
        if end > start:
            result.append(s[start:end])
        start = end + len(delim)
    return result


class PathTrie:
    """路径 Trie 树，用于存储解码路径"""
    ROOT = -1

    def __init__(self):
        self.log_prob_b_prev = NUM_FLT_INF
        self.log_prob_nb_prev = NUM_FLT_INF
        self.log_prob_b_cur = NUM_FLT_INF
        self.log_prob_nb_cur = NUM_FLT_INF
        self.score = NUM_FLT_INF
        self.approx_ctc = NUM_FLT_INF
        
        self.character = self.ROOT
        self.exists_ = True
        self.parent: Optional['PathTrie'] = None
        self.children_: Dict[int, 'PathTrie'] = {}

    def get_path_trie(self, new_char: int, reset: bool = True) -> Optional['PathTrie']:
        """获取或创建子节点"""
        if new_char in self.children_:
            child = self.children_[new_char]
            if not child.exists_:
                child.exists_ = True
                child.log_prob_b_prev = NUM_FLT_INF
                child.log_prob_nb_prev = NUM_FLT_INF
                child.log_prob_b_cur = NUM_FLT_INF
                child.log_prob_nb_cur = NUM_FLT_INF
            return child
        else:
            # 创建新节点
            new_path = PathTrie()
            new_path.character = new_char
            new_path.parent = self
            self.children_[new_char] = new_path
            return new_path

    def get_path_vec(self, output: List[int], stop: int = ROOT, max_steps: Optional[int] = None) -> 'PathTrie':
        """获取从根到当前节点的路径"""
        # 先收集路径
        path = []
        node = self
        steps = 0
        while node and node.character != self.ROOT:
            if node.character == stop:
                break
            if max_steps is not None and steps >= max_steps:
                break
            path.append(node.character)
            node = node.parent
            steps += 1
        
        # 反转路径并添加到输出
        path.reverse()
        output.extend(path)
        return self

    def iterate_to_vec(self, output: List['PathTrie']):
        """遍历所有存在的节点并更新概率"""
        if self.exists_:
            self.log_prob_b_prev = self.log_prob_b_cur
            self.log_prob_nb_prev = self.log_prob_nb_cur
            
            self.log_prob_b_cur = NUM_FLT_INF
            self.log_prob_nb_cur = NUM_FLT_INF
            
            self.score = log_sum_exp(self.log_prob_b_prev, self.log_prob_nb_prev)
            output.append(self)
        
        for child in self.children_.values():
            child.iterate_to_vec(output)

    def remove(self):
        """移除节点"""
        self.exists_ = False
        if len(self.children_) == 0:
            if self.parent:
                if self.character in self.parent.children_:
                    del self.parent.children_[self.character]
                if len(self.parent.children_) == 0 and not self.parent.exists_:
                    self.parent.remove()

    def is_empty(self) -> bool:
        """检查节点是否为空"""
        return self.character == self.ROOT


def prefix_compare(x: PathTrie, y: PathTrie) -> bool:
    """比较两个前缀的分数"""
    if x.score == y.score:
        if x.character == y.character:
            return False
        else:
            return x.character < y.character
    else:
        return x.score > y.score


def get_pruned_log_probs(prob_step: List[float], cutoff_prob: float, cutoff_top_n: int) -> List[Tuple[int, float]]:
    """获取剪枝后的对数概率"""
    prob_idx = [(i, prob) for i, prob in enumerate(prob_step)]
    
    # 剪枝
    cutoff_len = len(prob_step)
    if cutoff_prob < 1.0 or cutoff_top_n < cutoff_len:
        prob_idx.sort(key=lambda x: x[1], reverse=True)
        if cutoff_prob < 1.0:
            cum_prob = 0.0
            cutoff_len = 0
            for i, (idx, prob) in enumerate(prob_idx):
                cum_prob += prob
                cutoff_len += 1
                if cum_prob >= cutoff_prob or cutoff_len >= cutoff_top_n:
                    break
        prob_idx = prob_idx[:cutoff_len]
    
    log_prob_idx = [(idx, math.log(prob + NUM_FLT_MIN)) for idx, prob in prob_idx]
    return log_prob_idx


class Scorer:
    """语言模型评分器，使用 kenlm"""
    
    def __init__(self, alpha: float, beta: float, lm_path: str, vocab_list: List[str]):
        self.alpha = alpha
        self.beta = beta
        self.vocab_list = vocab_list
        self.char_list_ = vocab_list
        self.char_map_: Dict[str, int] = {}
        self.SPACE_ID_ = -1
        
        # 加载语言模型
        self.language_model_ = kenlm.Model(lm_path)
        self.max_order_ = self.language_model_.order
        
        # 获取词汇表
        self.vocabulary_ = []
        self.is_character_based_ = True
        
        # 检查是否是字符级别
        for i, vocab in enumerate(vocab_list):
            if vocab != UNK_TOKEN and vocab != START_TOKEN and vocab != END_TOKEN:
                if get_utf8_str_len(vocab) > 1:
                    self.is_character_based_ = False
        
        # 设置字符映射
        self.set_char_map(vocab_list)
    
    def reset_params(self, alpha: float, beta: float):
        """重置参数"""
        self.alpha = alpha
        self.beta = beta
    
    def is_character_based(self) -> bool:
        """检查是否是字符级别"""
        return self.is_character_based_
    
    def set_char_map(self, char_list: List[str]):
        """设置字符映射"""
        self.char_list_ = char_list
        self.char_map_.clear()
        
        for i, char in enumerate(char_list):
            if char == kSPACE:
                self.SPACE_ID_ = i
            # FST 的初始状态是 0，所以字符索引从 1 开始
            self.char_map_[char] = i + 1
    
    def vec2str(self, input_vec: List[int]) -> str:
        """将索引向量转换为字符串"""
        word = ""
        for idx in input_vec:
            if 0 <= idx < len(self.char_list_):
                word += self.char_list_[idx]
        return word
    
    def split_labels(self, labels: List[int]) -> List[str]:
        """将标签分割为单词列表"""
        if not labels:
            return []
        
        s = self.vec2str(labels)
        if self.is_character_based_:
            words = split_utf8_str(s)
        else:
            words = split_str(s, " ")
        return words
    
    def get_log_cond_prob(self, words: List[str]) -> float:
        """获取条件对数概率"""
        if not words:
            return 0.0
        
        # 使用 kenlm 计算条件概率
        # 对于 n-gram 模型，score() 方法计算的是整个序列的概率
        # 对于条件概率，我们需要计算最后一个词在给定前面词的条件下的概率
        try:
            text = " ".join(words)
            # 计算整个序列的分数（log10）
            score = self.language_model_.score(text, bos=False, eos=False)
            # kenlm 返回的是 log10，转换为自然对数
            return score * math.log(10)
        except Exception:
            # 如果遇到 OOV，返回 OOV_SCORE
            return OOV_SCORE
    
    def get_sent_log_prob(self, words: List[str]) -> float:
        """获取句子对数概率"""
        if not words:
            text = ""
        else:
            text = " ".join(words)
        
        # 使用 kenlm 计算句子概率（包含 bos 和 eos）
        try:
            # kenlm 的 score() 方法会自动添加 bos 和 eos
            score = self.language_model_.score(text, bos=True, eos=True)
            # 转换为自然对数
            return score * math.log(10)
        except Exception:
            return OOV_SCORE
    
    def get_log_prob(self, words: List[str]) -> float:
        """获取对数概率"""
        if not words:
            return 0.0
        
        # 对于长序列，使用滑动窗口计算 n-gram 概率
        if len(words) <= self.max_order_:
            return self.get_log_cond_prob(words)
        
        score = 0.0
        for i in range(len(words) - self.max_order_ + 1):
            ngram = words[i:i + self.max_order_]
            score += self.get_log_cond_prob(ngram)
        return score
    
    def make_ngram(self, prefix: PathTrie) -> List[str]:
        """从路径构建 n-gram"""
        ngram = []
        current_node = prefix
        
        for order in range(self.max_order_):
            prefix_vec = []
            
            if self.is_character_based_:
                new_node = current_node.get_path_vec(prefix_vec, self.SPACE_ID_, 1)
                current_node = new_node
            else:
                new_node = current_node.get_path_vec(prefix_vec, self.SPACE_ID_)
                if new_node.parent:
                    current_node = new_node.parent  # 跳过空格
            
            # 重构单词
            word = self.vec2str(prefix_vec)
            ngram.append(word)
            
            if new_node.character == -1:
                # 没有更多空格，但仍需要 order
                for _ in range(self.max_order_ - order - 1):
                    ngram.append(START_TOKEN)
                break
        
        ngram.reverse()
        return ngram


def get_beam_search_result(prefixes: List[PathTrie], vocabulary: List[str], beam_size: int) -> List[Tuple[float, str]]:
    """获取 beam search 结果"""
    space_prefixes = []
    for i in range(min(beam_size, len(prefixes))):
        space_prefixes.append(prefixes[i])
    
    space_prefixes.sort(key=lambda x: x.score, reverse=True)
    
    output_vecs = []
    for i in range(min(beam_size, len(space_prefixes))):
        output = []
        space_prefixes[i].get_path_vec(output)
        
        # 将索引转换为字符串
        output_str = ""
        for idx in output:
            if 0 <= idx < len(vocabulary):
                ch = vocabulary[idx]
                output_str += (ch == kSPACE) and tSPACE or ch
        
        output_pair = (-space_prefixes[i].approx_ctc, output_str)
        output_vecs.append(output_pair)
    
    return output_vecs


def ctc_beam_search_decoding(probs_seq: List[List[float]],
                             vocabulary: List[str],
                             beam_size: int,
                             cutoff_prob: float = 1.0,
                             cutoff_top_n: int = 40,
                             blank_id: int = 0,
                             ext_scoring_func: Optional[Scorer] = None) -> List[Tuple[float, str]]:
    """集束搜索解码器
    
    :param probs_seq: 单个2-D概率分布列表，每个元素是词汇表和空白上的标准化概率列表
    :param vocabulary: 词汇列表
    :param beam_size: 集束搜索宽度
    :param cutoff_prob: 剪枝中的截断概率，默认1.0，没有剪枝
    :param cutoff_top_n: 剪枝时的截断数，仅在词汇表中具有最大probs的cutoff_top_n字符用于光束搜索，默认为40
    :param blank_id: 空白索引
    :param ext_scoring_func: 外部评分功能部分解码句子，如字计数或语言模型
    :return: 解码结果为log概率和句子的元组列表，按概率降序排列
    """
    # 维度检查
    num_time_steps = len(probs_seq)
    for i in range(num_time_steps):
        assert len(probs_seq[i]) == len(vocabulary), \
            f"The shape of probs_seq does not match with the shape of the vocabulary"
    
    # 分配空格 ID
    try:
        space_id = vocabulary.index(kSPACE)
    except ValueError:
        space_id = -2
    
    # 初始化前缀的根
    root = PathTrie()
    root.score = root.log_prob_b_prev = 0.0
    prefixes: List[PathTrie] = [root]
    
    # 前缀搜索遍历时间步
    for time_step in range(num_time_steps):
        prob = probs_seq[time_step]
        
        min_cutoff = NUM_FLT_INF
        full_beam = False
        if ext_scoring_func is not None:
            num_prefixes = min(len(prefixes), beam_size)
            prefixes[:num_prefixes] = sorted(prefixes[:num_prefixes], key=lambda x: x.score, reverse=True)
            if num_prefixes > 0:
                min_cutoff = prefixes[num_prefixes - 1].score + \
                            math.log(prob[blank_id] + NUM_FLT_MIN) - \
                            max(0.0, ext_scoring_func.beta)
            full_beam = (num_prefixes == beam_size)
        
        log_prob_idx = get_pruned_log_probs(prob, cutoff_prob, cutoff_top_n)
        
        # 遍历字符
        for c, log_prob_c in log_prob_idx:
            for i in range(min(len(prefixes), beam_size)):
                prefix = prefixes[i]
                if full_beam and log_prob_c + prefix.score < min_cutoff:
                    break
                
                # 空白
                if c == blank_id:
                    prefix.log_prob_b_cur = log_sum_exp(
                        prefix.log_prob_b_cur, log_prob_c + prefix.score)
                    continue
                
                # 重复字符
                if c == prefix.character:
                    prefix.log_prob_nb_cur = log_sum_exp(
                        prefix.log_prob_nb_cur,
                        log_prob_c + prefix.log_prob_nb_prev)
                
                # 获取新前缀
                prefix_new = prefix.get_path_trie(c)
                
                if prefix_new is not None:
                    log_p = NUM_FLT_INF
                    
                    if c == prefix.character and prefix.log_prob_b_prev > NUM_FLT_INF:
                        log_p = log_prob_c + prefix.log_prob_b_prev
                    elif c != prefix.character:
                        log_p = log_prob_c + prefix.score
                    
                    # 语言模型评分
                    if ext_scoring_func is not None and \
                       (c == space_id or ext_scoring_func.is_character_based()):
                        prefix_to_score = None
                        # 跳过空格的评分
                        if ext_scoring_func.is_character_based():
                            prefix_to_score = prefix_new
                        else:
                            prefix_to_score = prefix
                        
                        score = 0.0
                        ngram = ext_scoring_func.make_ngram(prefix_to_score)
                        score = ext_scoring_func.get_log_cond_prob(ngram) * ext_scoring_func.alpha
                        log_p += score
                        log_p += ext_scoring_func.beta
                    
                    prefix_new.log_prob_nb_cur = log_sum_exp(prefix_new.log_prob_nb_cur, log_p)
        
        prefixes.clear()
        # 更新对数概率
        root.iterate_to_vec(prefixes)
        
        # 只保留 top beam_size 前缀
        if len(prefixes) >= beam_size:
            prefixes.sort(key=lambda x: x.score, reverse=True)
            for i in range(beam_size, len(prefixes)):
                prefixes[i].remove()
            prefixes = prefixes[:beam_size]
    
    # 对每个不以空格结尾的前缀的最后一个单词进行评分
    if ext_scoring_func is not None and not ext_scoring_func.is_character_based():
        for i in range(min(beam_size, len(prefixes))):
            prefix = prefixes[i]
            if not prefix.is_empty() and prefix.character != space_id:
                score = 0.0
                ngram = ext_scoring_func.make_ngram(prefix)
                score = ext_scoring_func.get_log_cond_prob(ngram) * ext_scoring_func.alpha
                score += ext_scoring_func.beta
                prefix.score += score
    
    num_prefixes = min(len(prefixes), beam_size)
    prefixes[:num_prefixes] = sorted(prefixes[:num_prefixes], key=lambda x: x.score, reverse=True)
    
    # 计算近似的 ctc 分数作为返回分数
    for i in range(min(beam_size, len(prefixes))):
        approx_ctc = prefixes[i].score
        if ext_scoring_func is not None:
            output = []
            prefixes[i].get_path_vec(output)
            prefix_length = len(output)
            words = ext_scoring_func.split_labels(output)
            # 移除单词插入
            approx_ctc = approx_ctc - prefix_length * ext_scoring_func.beta
            # 移除语言模型权重
            approx_ctc -= ext_scoring_func.get_sent_log_prob(words) * ext_scoring_func.alpha
        prefixes[i].approx_ctc = approx_ctc
    
    return get_beam_search_result(prefixes, vocabulary, beam_size)


def ctc_beam_search_decoding_batch(probs_split: List[List[List[float]]],
                                   vocabulary: List[str],
                                   beam_size: int,
                                   num_processes: int,
                                   cutoff_prob: float = 1.0,
                                   cutoff_top_n: int = 40,
                                   blank_id: int = 0,
                                   ext_scoring_func: Optional[Scorer] = None) -> List[List[Tuple[float, str]]]:
    """批量 CTC beam search 解码器（使用循环处理，不使用多进程）
    
    :param probs_split: 3-D列表，每个元素作为ctc_beam_search_decoder()使用的2-D概率列表的实例
    :param vocabulary: 词汇列表
    :param beam_size: 集束搜索宽度
    :param num_processes: 并行解码进程数（此实现中不使用）
    :param cutoff_prob: 剪枝中的截断概率，默认1.0，没有剪枝
    :param cutoff_top_n: 剪枝时的截断数，仅在词汇表中具有最大probs的cutoff_top_n字符用于光束搜索，默认为40
    :param blank_id: 空白索引
    :param ext_scoring_func: 外部评分功能部分解码句子，如字计数或语言模型
    :return: 解码结果为log概率和句子的元组列表，按概率降序排列的列表
    """
    batch_results = []
    for probs_seq in probs_split:
        result = ctc_beam_search_decoding(
            probs_seq, vocabulary, beam_size, cutoff_prob,
            cutoff_top_n, blank_id, ext_scoring_func)
        batch_results.append(result)
    return batch_results


class BeamSearchDecoder:
    def __init__(self, conf_path, vocab_list, blank_id=0):
        # 读取数据增强配置文件
        with open(conf_path, 'r', encoding='utf-8') as f:
            decoder_configs = yaml.load(f.read(), Loader=yaml.FullLoader)
        print_arguments(configs=decoder_configs, title='集束搜索解码器参数')
        self.configs = dict_to_object(decoder_configs)
        self.vocab_list = vocab_list
        self.alpha = self.configs.alpha
        self.beta = self.configs.beta
        self.beam_size = self.configs.beam_size
        self.cutoff_prob = self.configs.cutoff_prob
        self.cutoff_top_n = self.configs.cutoff_top_n
        self.vocab_list = vocab_list
        self.num_processes = self.configs.num_processes
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
        logger.info("初始化解码器完成!")
        logger.info('=' * 70)

    # 重置参数
    def reset_params(self, alpha, beta):
        self._ext_scorer.reset_params(alpha, beta)

    # 单个数据解码
    def ctc_beam_search_decoder(self, ctc_probs):
        if not isinstance(ctc_probs, list):
            ctc_probs = ctc_probs.tolist()
        # 必须要计算softmax，否则会导致结果不正确
        ctc_probs = np.array(ctc_probs, dtype=np.float32)
        ctc_probs = self.softmax(ctc_probs)[0]
        # beam search decode
        beam_search_result = ctc_beam_search_decoding(probs_seq=ctc_probs.tolist(),
                                                      vocabulary=self.vocab_list,
                                                      beam_size=self.beam_size,
                                                      ext_scoring_func=self._ext_scorer,
                                                      cutoff_prob=self.cutoff_prob,
                                                      cutoff_top_n=self.cutoff_top_n,
                                                      blank_id=self.blank_id)
        result = beam_search_result[0][1].replace('▁', ' ').strip()
        return result

    # 一批数据解码
    def ctc_beam_search_decoder_batch(self, ctc_probs, ctc_lens):
        if not isinstance(ctc_probs, list):
            ctc_probs = ctc_probs.tolist()
            ctc_lens = ctc_lens.tolist()
        # 必须要计算softmax，否则会导致结果不正确
        ctc_probs = np.array(ctc_probs, dtype=np.float32)
        ctc_probs = self.softmax(ctc_probs).tolist()
        new_ctc_probs = []
        for i, ctc_prob in enumerate(ctc_probs):
            ctc_prob = ctc_prob[:ctc_lens[i]]
            new_ctc_probs.append(ctc_prob)
        # beam search decode (使用循环处理，不使用多进程)
        beam_search_results = ctc_beam_search_decoding_batch(probs_split=new_ctc_probs,
                                                             vocabulary=self.vocab_list,
                                                             beam_size=self.beam_size,
                                                             num_processes=self.num_processes,
                                                             ext_scoring_func=self._ext_scorer,
                                                             cutoff_prob=self.cutoff_prob,
                                                             cutoff_top_n=self.cutoff_top_n,
                                                             blank_id=self.blank_id)
        results = [result[0][1].replace('▁', ' ').strip() for result in beam_search_results]
        return results

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
