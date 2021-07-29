import argparse
import functools
import time
import paddle.fluid as fluid
from data_utils.data import DataGenerator
from model_utils.model import DeepSpeech2Model
from decoders.ctc_greedy_decoder import greedy_decoder
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('wav_path',         str,    './dataset/test.wav', "预测音频的路径")
add_arg('beam_size',        int,    10,     "定向搜索的大小，范围:[5, 500]")
add_arg('num_conv_layers',  int,    2,      "卷积层数量")
add_arg('num_rnn_layers',   int,    3,      "循环神经网络的数量")
add_arg('rnn_layer_size',   int,    2048,   "循环神经网络的大小")
add_arg('alpha',            float,  1.2,    "定向搜索的LM系数")
add_arg('beta',             float,  0.35,   "定向搜索的WC系数")
add_arg('cutoff_prob',      float,  1.0,    "剪枝的概率")
add_arg('cutoff_top_n',     int,    40,     "剪枝的最大值")
add_arg('use_gru',          bool,   True,   "是否使用GRUs模型，不使用RNN")
add_arg('use_gpu',          bool,   True,   "是否使用GPU训练")
add_arg('share_rnn_weights',bool,   False,   "是否在RNN上共享权重")
add_arg('mean_std_path',    str,    './dataset/mean_std.npz',      "数据集的均值和标准值的npy文件路径")
add_arg('vocab_path',       str,    './dataset/zh_vocab.txt',      "数据集的词汇表文件路径")
add_arg('model_path',       str,    './models/step_final/',        "训练保存的模型文件夹路径")
add_arg('lang_model_path',  str,    './lm/zh_giga.no_cna_cmn.prune01244.klm',        "语言模型文件路径")
add_arg('decoding_method',  str,    'ctc_greedy',    "结果解码方法，有集束搜索(ctc_beam_search)、贪婪策略(ctc_greedy)", choices=['ctc_beam_search', 'ctc_greedy'])
add_arg('specgram_type',    str,    'linear',        "对音频的预处理方式，有: linear, mfcc", choices=['linear', 'mfcc'])
args = parser.parse_args()

# 是否使用GPU
place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()

# 获取数据生成器，处理数据和获取字典需要
data_generator = DataGenerator(vocab_filepath=args.vocab_path,
                               mean_std_filepath=args.mean_std_path,
                               augmentation_config='{}',
                               specgram_type=args.specgram_type,
                               keep_transcription_text=True,
                               place=place,
                               is_training=False)
# 获取DeepSpeech2模型，并设置为预测
ds2_model = DeepSpeech2Model(vocab_size=data_generator.vocab_size,
                             num_conv_layers=args.num_conv_layers,
                             num_rnn_layers=args.num_rnn_layers,
                             rnn_layer_size=args.rnn_layer_size,
                             use_gru=args.use_gru,
                             init_from_pretrained_model=args.model_path,
                             place=place,
                             share_rnn_weights=args.share_rnn_weights,
                             is_infer=True)

# 集束搜索方法的处理
if args.decoding_method == "ctc_beam_search":
    try:
        from decoders.beam_search_decoder import BeamSearchDecoder
        beam_search_decoder = BeamSearchDecoder(args.alpha, args.beta, args.lang_model_path, data_generator.vocab_list)
    except ModuleNotFoundError:
        raise Exception('缺少ctc_decoders库，如果是Windows系统，请使用ctc_greedy。如果是Linux系统，且一定要使用ctc_beam_search解码策略'
                        '请执行`cd decoders && sh setup.sh`编译ctc_beam_search解码函数')


# 开始预测
def predict(filename):
    # 加载音频文件，并进行预处理
    feature = data_generator.process_utterance(filename, "")
    # 执行预测
    probs_split = ds2_model.infer(feature=feature)

    # 执行解码
    if args.decoding_method == 'ctc_greedy':
        # 贪心解码策略
        result = greedy_decoder(probs_seq=probs_split,
                                vocabulary=data_generator.vocab_list,
                                blank_index=len(data_generator.vocab_list))
    else:
        # 集束搜索解码策略
        result = beam_search_decoder.decode_beam_search(probs_split=probs_split,
                                                        beam_alpha=args.alpha,
                                                        beam_beta=args.beta,
                                                        beam_size=args.beam_size,
                                                        cutoff_prob=args.cutoff_prob,
                                                        cutoff_top_n=args.cutoff_top_n,
                                                        vocab_list=data_generator.vocab_list,
                                                        blank_id=len(data_generator.vocab_list))
    score, text = result[0], result[1]
    return score, text


def main():
    print_arguments(args)
    start = time.time()
    score, text = predict(filename=args.wav_path)
    print("消耗时间：%d, 识别结果: %s, 得分: %d" % (round((time.time() - start) * 1000), text, score))


if __name__ == "__main__":
    main()
