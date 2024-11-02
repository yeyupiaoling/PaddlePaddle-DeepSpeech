import argparse
import functools
import time

from utils.predict import Predictor
from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('wav_path',         str,    'dataset/test.wav', "预测音频的路径")
add_arg('use_gpu',          bool,   True,   "是否使用GPU预测")
add_arg('enable_mkldnn',    bool,   False,  "是否使用mkldnn加速")
add_arg('to_itn',           bool,  False,   "是否逆文本标准化")
add_arg('beam_size',        int,    300,    "集束搜索解码相关参数，搜索的大小，范围:[5, 500]")
add_arg('alpha',            float,  1.2,    "集束搜索解码相关参数，LM系数")
add_arg('beta',             float,  0.35,   "集束搜索解码相关参数，WC系数")
add_arg('cutoff_prob',      float,  0.99,   "集束搜索解码相关参数，剪枝的概率")
add_arg('cutoff_top_n',     int,    40,     "集束搜索解码相关参数，剪枝的最大值")
add_arg('vocab_dir',        str,    'dataset/vocab_model',    "数据字典模型文件夹")
add_arg('model_dir',        str,    'models/inference/',      "导出的预测模型文件夹路径")
add_arg('lang_model_path',  str,    'lm/zh_giga.no_cna_cmn.prune01244.klm',   "集束搜索解码相关参数，语言模型文件路径")
add_arg('decoder',          str,    'ctc_greedy',    "结果解码方法，有集束搜索解码器(ctc_beam_search)、贪心解码器(ctc_greedy)", choices=['ctc_beam_search', 'ctc_greedy'])
args = parser.parse_args()
print_arguments(args)


predictor = Predictor(model_dir=args.model_dir,
                      vocab_dir=args.vocab_dir,
                      decoder=args.decoder,
                      alpha=args.alpha,
                      beta=args.beta,
                      lang_model_path=args.lang_model_path,
                      beam_size=args.beam_size,
                      cutoff_prob=args.cutoff_prob,
                      cutoff_top_n=args.cutoff_top_n,
                      use_gpu=args.use_gpu,
                      enable_mkldnn=args.enable_mkldnn)


def predict_audio():
    start = time.time()
    text = predictor.predict(audio_path=args.wav_path, to_itn=args.to_itn)
    print(f"消耗时间：{int((time.time() - start) * 1000)}ms, 识别结果: {text}")


if __name__ == "__main__":
    predict_audio()
