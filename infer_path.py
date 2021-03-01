import argparse
import functools
import time
import paddle.fluid as fluid
from data_utils.data import DataGenerator
from model_utils.model import DeepSpeech2Model
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('wav_path',         str,    './dataset/test.wav', "Server's IP port.")
add_arg('beam_size',        int,    10,     "Beam search width. range:[5, 500]")
add_arg('num_conv_layers',  int,    2,      "# of convolution layers.")
add_arg('num_rnn_layers',   int,    3,      "# of recurrent layers.")
add_arg('rnn_layer_size',   int,    2048,   "# of recurrent cells per layer.")
add_arg('alpha',            float,  1.2,   "Coef of LM for beam search.")
add_arg('beta',             float,  0.35,   "Coef of WC for beam search.")
add_arg('cutoff_prob',      float,  1.0,    "Cutoff probability for pruning.")
add_arg('cutoff_top_n',     int,    40,     "Cutoff number for pruning.")
add_arg('use_gru',          bool,   True,  "Use GRUs instead of simple RNNs.")
add_arg('use_gpu',          bool,   True,   "Use GPU or not.")
add_arg('share_rnn_weights',bool,   False,   "Share input-hidden weights across bi-directional RNNs. Not for GRU.")
add_arg('mean_std_path',    str,    './dataset/mean_std.npz',      "Filepath of normalizer's mean & std.")
add_arg('vocab_path',       str,    './dataset/zh_vocab.txt',      "Filepath of vocabulary.")
add_arg('model_path',       str,    './models/step_final/',        "The pre-trained model.")
add_arg('lang_model_path',  str,    './lm/zh_giga.no_cna_cmn.prune01244.klm',        "Filepath for language model.")
add_arg('decoding_method',  str,    'ctc_beam_search',        "Decoding method. Options: ctc_beam_search, ctc_greedy", choices=['ctc_beam_search', 'ctc_greedy'])
add_arg('specgram_type',    str,    'linear',        "Audio feature type. Options: linear, mfcc.", choices=['linear', 'mfcc'])
args = parser.parse_args()

# 是否使用GPU
if args.use_gpu:
    place = fluid.CUDAPlace(0)
else:
    place = fluid.CPUPlace()

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
# 定向搜索方法的处理
if args.decoding_method == "ctc_beam_search":
    ds2_model.init_ext_scorer(args.alpha, args.beta, args.lang_model_path, data_generator.vocab_list)


# 开始预测
def predict(filename):
    # 加载音频文件，并进行预处理
    feature = data_generator.process_utterance(filename, "")
    # 执行预测
    probs_split = ds2_model.infer(feature=feature)

    # 执行解码
    if args.decoding_method == "ctc_greedy":
        # 最优路径解码
        result_transcript = ds2_model.decode_batch_greedy(probs_split=probs_split,
                                                          vocab_list=data_generator.vocab_list)
    else:
        # 定向搜索解码
        result_transcript = ds2_model.decode_batch_beam_search(probs_split=probs_split,
                                                               beam_alpha=args.alpha,
                                                               beam_beta=args.beta,
                                                               beam_size=args.beam_size,
                                                               cutoff_prob=args.cutoff_prob,
                                                               cutoff_top_n=args.cutoff_top_n,
                                                               vocab_list=data_generator.vocab_list,
                                                               num_processes=1)
    return result_transcript[0]


def main():
    print_arguments(args)
    start = time.time()
    text = predict(filename=args.wav_path)
    print("消耗时间：%d, 识别结果: %s" % (round((time.time() - start) * 1000), text))


if __name__ == "__main__":
    main()
