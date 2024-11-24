import argparse
import functools
import os
import time

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
os.chdir(parent_dir)

import numpy as np
import paddle
from data_utils.audio_featurizer import AudioFeaturizer
from loguru import logger
from paddle.io import DataLoader
from tqdm import tqdm

from data_utils.collate_fn import collate_fn
from data_utils.reader import CustomDataset
from data_utils.tokenizer import Tokenizer
from decoders.beam_search_decoder import BeamSearchDecoder
from model_utils.model import DeepSpeech2Model
from utils.checkpoint import load_pretrained
from utils.metrics import wer, cer
from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_gpu', bool, True, "是否使用GPU评估")
add_arg('batch_size', int, 32, "评估是每一批数据的大小")
add_arg('num_conv_layers', int, 2, "卷积层数量")
add_arg('num_rnn_layers', int, 3, "循环神经网络的数量")
add_arg('rnn_layer_size', int, 1024, "循环神经网络的大小")
add_arg('min_duration', float, 0.5, "最短的用于训练的音频长度")
add_arg('max_duration', float, 20.0, "最长的用于训练的音频长度")
add_arg('test_manifest', str, 'dataset/manifest.test', "需要评估的测试数据列表")
add_arg('mean_istd_path', str, 'dataset/mean_istd.json', "均值和标准值得json文件路径，后缀 (.json)")
add_arg('vocab_dir', str, 'dataset/vocab_model', "数据字典模型文件夹")
add_arg('pretrained_model', str, 'models/best_model/', "模型文件路径")
add_arg('beam_search_conf', str, 'configs/decoder.yml', "集束搜索解码相关参数")
add_arg('metrics_type', str, 'cer', "评估所使用的错误率方法，有字错率(cer)、词错率(wer)", choices=['wer', 'cer'])
add_arg('num_alphas', int, 30, "用于调优的alpha候选项")
add_arg('num_betas', int, 20, "用于调优的beta候选项")
add_arg('alpha_from', float, 1.0, "alpha调优开始大小")
add_arg('alpha_to', float, 3.2, "alpha调优结速大小")
add_arg('beta_from', float, 0.1, "beta调优开始大小")
add_arg('beta_to', float, 4.5, "beta调优结速大小")
args = parser.parse_args()
print_arguments(args=args)


def tune():
    # 逐步调整alphas参数和betas参数
    assert args.num_alphas >= 0, "num_alphas must be non-negative!"
    assert args.num_betas >= 0, "num_betas must be non-negative!"

    # 创建用于搜索的alphas参数和betas参数
    cand_alphas = np.linspace(args.alpha_from, args.alpha_to, args.num_alphas)
    cand_betas = np.linspace(args.beta_from, args.beta_to, args.num_betas)
    params_grid = [(round(alpha, 2), round(beta, 2)) for alpha in cand_alphas for beta in cand_betas]
    logger.info(f'解码alpha和beta的组合数量：{len(params_grid)}，排列：{params_grid}')

    # 是否使用GPU
    if args.use_gpu:
        assert paddle.is_compiled_with_cuda(), 'GPU不可用'
        paddle.device.set_device("gpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        paddle.device.set_device("cpu")

    audio_featurizer = AudioFeaturizer(mode="train")
    tokenizer = Tokenizer(args.vocab_dir)
    # 获取苹果数据
    test_dataset = CustomDataset(data_manifest=args.test_manifest,
                                 audio_featurizer=audio_featurizer,
                                 tokenizer=tokenizer,
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
                             num_conv_layers=args.num_conv_layers,
                             num_rnn_layers=args.num_rnn_layers,
                             rnn_layer_size=args.rnn_layer_size)

    model = load_pretrained(model, args.pretrained_model)

    # 获取解码器
    beam_search_decoder = BeamSearchDecoder(conf_path=args.beam_search_conf,
                                            vocab_list=tokenizer.vocab_list,
                                            blank_id=tokenizer.blank_id)

    # 开始评估
    start = time.time()
    model.eval()
    all_ctc_probs, all_ctc_lens, all_label = [], [], []
    with paddle.no_grad():
        for batch_id, batch in enumerate(tqdm(test_loader())):
            inputs, labels, input_lens, label_lens = batch
            ctc_probs, ctc_lens = model.predict(inputs, input_lens)
            ctc_probs, ctc_lens = ctc_probs.numpy(), ctc_lens.numpy()
            labels = labels.numpy().tolist()
            # 移除每条数据的-1值
            labels = [list(filter(lambda x: x != -1, label)) for label in labels]
            labels_str = tokenizer.ids2text(labels)
            all_ctc_probs.append(ctc_probs)
            all_ctc_lens.append(ctc_lens)
            all_label.append(labels_str)
    logger.info(f'获取模型输出消耗时间：{int(time.time() - start)}s')

    logger.info('开始使用识别结果解码...')
    # 搜索alphas参数和betas参数
    best_alpha, best_beta, best_result = 0, 0, 1
    for i, (alpha, beta) in enumerate(params_grid):
        error_results = []
        for j in tqdm(range(len(all_ctc_probs))):
            ctc_probs, ctc_lens, label = all_ctc_probs[j], all_ctc_lens[j], all_label[j]
            beam_search_decoder.reset_params(alpha, beta)
            text = beam_search_decoder.ctc_beam_search_decoder_batch(ctc_probs=ctc_probs, ctc_lens=ctc_lens)
            for l, t in zip(label, text):
                # 计算字错率或者词错率
                if args.metrics_type == 'wer':
                    error_rate = wer(l, t)
                else:
                    error_rate = cer(l, t)
                error_results.append(error_rate)
        error_result = np.mean(error_results)
        if error_result < best_result:
            best_alpha = alpha
            best_beta = beta
            best_result = error_result
        logger.info(
            f'[{i + 1}/{len(params_grid)}] 当alpha为：{alpha}, beta为：{beta}，{args.metrics_type}：{error_result:.5f}, '
            f'【目前最优】当alpha为：{best_alpha}, beta为：{best_beta}，{args.metrics_type}：{best_result:.5f}')
    logger.info(f'【最终最优】当alpha为：%f, {best_alpha}, beta为：{best_beta}，{args.metrics_type}：{best_result:.5f}')


if __name__ == '__main__':
    tune()
