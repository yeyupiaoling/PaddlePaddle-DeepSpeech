from typing import List

import numpy as np


def ctc_greedy_search(ctc_probs: np.ndarray,
                      blank_id: int = 0) -> List[List[int]]:
    """贪心解码器  

    param ctc_probs: (maxlen, vocab_size) 模型编码器输出的概率分布
    param blank_id: 空白标签的id  
    return: 解码结果  
    """
    topk_index = np.argmax(ctc_probs, axis=1)  # (maxlen)
    hyps = topk_index.tolist()
    results = [hyp for hyp in hyps if hyp != blank_id]
    return results


def ctc_greedy_search_batch(ctc_probs: np.ndarray,
                            ctc_lens: np.ndarray,
                            blank_id: int = 0) -> List[List[int]]:
    """贪心解码器

    param ctc_probs: (B, maxlen, vocab_size) 模型编码器输出的概率分布
    param ctc_lens: (B, ) 每个样本的实际长度
    param blank_id: 空白标签的id
    return: 解码结果
    """
    batch_size, maxlen, vocab_size = ctc_probs.shape
    topk_index = np.argmax(ctc_probs, axis=2)  # (B, maxlen)

    mask = make_pad_mask(ctc_lens, maxlen)  # (B, maxlen)
    topk_index[mask] = blank_id  # (B, maxlen)

    hyps = topk_index.tolist()
    results = [remove_duplicates_and_blank(hyp, blank_id) for hyp in hyps]
    return results


def make_pad_mask(lengths: np.ndarray, max_len: int = 0) -> np.ndarray:
    """生成包含填充部分索引的掩码张量。  

    Args:  
        lengths (np.ndarray): 长度批处理 (B,)。  
    Returns:  
        np.ndarray: 包含填充部分索引的掩码张量。  

    Examples:  
        >>> lengths = np.array([5, 3, 2])  
        >>> make_pad_mask(lengths)  
        array([[False, False, False, False, False],  
               [False, False, False,  True,  True],  
               [False, False,  True,  True,  True]])  
    """
    batch_size = lengths.shape[0]
    max_len = max_len if max_len > 0 else lengths.max()
    seq_range = np.arange(max_len)
    seq_range_expand = np.tile(seq_range, (batch_size, 1))
    seq_length_expand = np.expand_dims(lengths, axis=-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def remove_duplicates_and_blank(hyp: List[int], blank_id: int = 0) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp
