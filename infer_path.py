import argparse
import functools
import time

from data_utils.data import DataGenerator
from utils.predict import Predictor
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('wav_path',         str,    './dataset/test.wav', "预测音频的路径")
add_arg('beam_size',        int,    10,     "定向搜索的大小，范围:[5, 500]")
add_arg('alpha',            float,  1.2,    "定向搜索的LM系数")
add_arg('beta',             float,  0.35,   "定向搜索的WC系数")
add_arg('cutoff_prob',      float,  1.0,    "剪枝的概率")
add_arg('cutoff_top_n',     int,    40,     "剪枝的最大值")
add_arg('use_gpu',          bool,   True,   "是否使用GPU预测")
add_arg('use_tensorrt',     bool,   False,  "是否使用TensorRT加速")
add_arg('enable_mkldnn',    bool,   False,  "是否使用mkldnn加速")
add_arg('mean_std_path',    str,    './dataset/mean_std.npz',      "数据集的均值和标准值的npy文件路径")
add_arg('vocab_path',       str,    './dataset/zh_vocab.txt',      "数据集的词汇表文件路径")
add_arg('model_dir',       str,     './models/infer/',             "导出的预测模型文件夹路径")
add_arg('lang_model_path',  str,    './lm/zh_giga.no_cna_cmn.prune01244.klm',        "语言模型文件路径")
add_arg('decoding_method',  str,    'ctc_greedy',    "结果解码方法，有集束搜索(ctc_beam_search)、贪婪策略(ctc_greedy)", choices=['ctc_beam_search', 'ctc_greedy'])
args = parser.parse_args()
print_arguments(args)


# 获取数据生成器，处理数据和获取字典需要
data_generator = DataGenerator(vocab_filepath=args.vocab_path,
                               mean_std_filepath=args.mean_std_path,
                               keep_transcription_text=True,
                               is_training=False)

predictor = Predictor(model_dir=args.model_dir, data_generator=data_generator, decoding_method=args, alpha=args.alpha,
                      beta=args.beta, lang_model_path=args.lang_model_path, beam_size=args.beam_size,
                      cutoff_prob=args.cutoff_prob, cutoff_top_n=args.cutoff_top_n, use_gpu=args.use_gpu,
                      use_tensorrt=args.use_tensorrt, enable_mkldnn=args.enable_mkldnn)


def main():
    start = time.time()
    score, text = predictor.predict(audio_path=args.wav_path)
    print("消耗时间：%d, 识别结果: %s, 得分: %d" % (round((time.time() - start) * 1000), text, score))


if __name__ == "__main__":
    main()
