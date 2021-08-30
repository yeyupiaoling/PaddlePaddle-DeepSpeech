from typing import List, Tuple
from pypinyin import lazy_pinyin, Style
from preprocess_transcription import split_syllable


def convert_to_pinyin(text: str) -> List[str]:
    """convert text into list of syllables, other characters that are not chinese, thus
    cannot be converted to pinyin are splited.
    """
    syllables = lazy_pinyin(
        text, style=Style.TONE3, neutral_tone_with_five=True)
    return syllables


def convert_sentence(text: str) -> List[Tuple[str]]:
    """convert a sentence into two list: phones and tones"""
    syllables = convert_to_pinyin(text)
    phones = []
    tones = []
    for syllable in syllables:
        p, t = split_syllable(syllable)
        phones.extend(p)
        tones.extend(t)

    return phones, tones


# 判断是否为中文字符
def is_uchar(in_str):
    for i in range(len(in_str)):
        uchar = in_str[i]
        if u'\u4e00' <= uchar <= u'\u9fa5':
            pass
        else:
            return False
    return True
