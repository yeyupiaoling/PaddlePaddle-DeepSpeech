import argparse
import functools
import os
import time

import paddle
from loguru import logger
from paddle.io import DataLoader
from tqdm import tqdm

from data_utils.collate_fn import collate_fn
from data_utils.featurizer.audio_featurizer import AudioFeaturizer
from data_utils.featurizer.text_featurizer import TextFeaturizer
from data_utils.reader import CustomDataset
from decoders.ctc_greedy_decoder import greedy_decoder_batch
from model_utils.model import DeepSpeech2Model
from utils.checkpoint import load_pretrained
from utils.metrics import wer, cer
from utils.utils import add_arguments, print_arguments, labels_to_string

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_gpu',          bool,   True,   "是否使用GPU评估")
add_arg('batch_size',       int,    32,     "评估是每一批数据的大小")
add_arg('num_rnn_layers',   int,    3,      "循环神经网络的数量")
add_arg('rnn_layer_size',   int,    1024,   "循环神经网络的大小")
add_arg('min_duration',     float,  0.5,    "最短的用于训练的音频长度")
add_arg('max_duration',     float,  20.0,   "最长的用于训练的音频长度")
add_arg('beam_size',        int,    300,    "集束搜索解码相关参数，搜索大小，范围:[5, 500]")
add_arg('alpha',            float,  1.2,    "集束搜索解码相关参数，LM系数")
add_arg('num_proc_bsearch', int,    8,      "集束搜索解码相关参数，使用CPU数量")
add_arg('beta',             float,  0.35,   "集束搜索解码相关参数，WC系数")
add_arg('cutoff_prob',      float,  0.99,   "集束搜索解码相关参数，剪枝的概率")
add_arg('cutoff_top_n',     int,    40,     "集束搜索解码相关参数，剪枝的最大值")
add_arg('test_manifest',    str,    './dataset/manifest.test',     "需要评估的测试数据列表")
add_arg('mean_istd_path',   str,    './dataset/mean_istd.json',    "均值和标准值得json文件路径，后缀 (.json)")
add_arg('vocab_path',       str,    './dataset/vocabulary.txt',    "数据集的字典文件路径")
add_arg('pretrained_model', str,    './models/epoch_15/',          "模型文件路径")
add_arg('lang_model_path',  str,    './lm/zh_giga.no_cna_cmn.prune01244.klm',    "集束搜索解码相关参数，语言模型文件路径")
add_arg('decoder',          str,    'ctc_greedy',        "结果解码方法，有集束搜索(ctc_beam_search)、贪婪策略(ctc_greedy)", choices=['ctc_beam_search', 'ctc_greedy'])
add_arg('metrics_type',     str,    'cer',    "评估所使用的错误率方法，有字错率(cer)、词错率(wer)", choices=['wer', 'cer'])
args = parser.parse_args()

beam_search_decoder = None


# 评估模型
def evaluate():
    # 是否使用GPU
    if args.use_gpu:
        assert paddle.is_compiled_with_cuda(), 'GPU不可用'
        paddle.device.set_device("gpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        paddle.device.set_device("cpu")

    audio_featurizer = AudioFeaturizer(mode="train")
    text_featurizer = TextFeaturizer(args.vocab_path)
    # 获取苹果数据
    test_dataset = CustomDataset(data_manifest=args.test_manifest,
                                 audio_featurizer=audio_featurizer,
                                 text_featurizer=text_featurizer,
                                 min_duration=args.min_duration,
                                 max_duration=args.max_duration,
                                 mode="eval")
    test_loader = DataLoader(dataset=test_dataset,
                             collate_fn=collate_fn,
                             batch_size=args.batch_size,
                             num_workers=4)

    model = DeepSpeech2Model(input_dim=test_dataset.feature_dim,
                             vocab_size=test_dataset.vocab_size,
                             mean_istd_path=args.mean_istd_path,
                             num_rnn_layers=args.num_rnn_layers,
                             rnn_layer_size=args.rnn_layer_size)

    model = load_pretrained(model, args.pretrained_model)

    start = time.time()
    model.eval()
    error_results = []
    with paddle.no_grad():
        for batch_id, batch in enumerate(tqdm(test_loader())):
            inputs, labels, input_lens, label_lens = batch
            output = model.predict(inputs, input_lens).numpy()
            out_strings = decoder_result(output, text_featurizer.vocab_list)
            labels_str = labels_to_string(labels, text_featurizer.vocab_list)
            for out_string, label in zip(*(out_strings, labels_str)):
                # 计算字错率或者词错率
                if args.metrics_type == 'wer':
                    error_rate = wer(label, out_string)
                else:
                    error_rate = cer(label, out_string)
                error_results.append(error_rate)
                logger.info(f'预测结果为：{out_string}')
                logger.info(f'实际标签为：{label}')
                logger.info(f'这条数据的{args.metrics_type}：{round(error_rate, 6)}，'
                            f'当前{args.metrics_type}：{round(sum(error_results) / len(error_results), 6)}')
                logger.info('-' * 70)
    error_result = float(sum(error_results) / len(error_results)) if len(error_results) > 0 else -1
    print(f"消耗时间：{time.time() - start}s, [{args.metrics_type}]：{error_result}")


def decoder_result(outs, vocabulary):
    global beam_search_decoder
    # 集束搜索方法的处理
    if args.decoder == "ctc_beam_search" and beam_search_decoder is None:
        try:
            from decoders.beam_search_decoder import BeamSearchDecoder
            beam_search_decoder = BeamSearchDecoder(args.alpha, args.beta, args.beam_size, args.cutoff_prob,
                                                    args.cutoff_top_n, args.vocab_list,
                                                    language_model_path=args.lang_model_path)
        except ModuleNotFoundError:
            logger.warning('==================================================================')
            logger.warning('缺少 paddlespeech-ctcdecoders 库，请根据文档安装。')
            logger.warning(
                'python -m pip install paddlespeech_ctcdecoders -U -i https://ppasr.yeyupiaoling.cn/pypi/simple/')
            logger.warning('【注意】现在已自动切换为ctc_greedy解码器，ctc_greedy解码器准确率相对较低。')
            logger.warning('==================================================================\n')
            args.decoder = 'ctc_greedy'

    # 执行解码
    outs = [outs[i, :, :] for i, _ in enumerate(range(outs.shape[0]))]
    if args.decoder == 'ctc_greedy':
        result = greedy_decoder_batch(outs, vocabulary)
    else:
        result = beam_search_decoder.decode_batch_beam_search_offline(probs_split=outs)
    return result


def main():
    print_arguments(args)
    evaluate()


if __name__ == '__main__':
    main()
