import argparse
import functools
import io
from datetime import datetime
from model_utils.model import DeepSpeech2Model
from model_utils.model_check import check_cuda, check_version
from data_utils.data import DataGenerator
from utils.utility import add_arguments, print_arguments, get_data_len

import paddle.fluid as fluid

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',      int,    16,     "训练每一批数据的大小")
add_arg('num_epoch',       int,    200,    "训练的轮数")
add_arg('num_conv_layers',  int,   2,      "卷积层数量")
add_arg('num_rnn_layers',   int,   3,      "循环神经网络的数量")
add_arg('rnn_layer_size',   int,   2048,   "循环神经网络的大小")
add_arg('learning_rate',   float,  5e-5,   "初始学习率")
add_arg('min_duration',    float,  1.0,    "最短的用于训练的音频长度")
add_arg('max_duration',    float,  15.0,   "最长的用于训练的音频长度")
add_arg('test_off',        bool,   False,  "是否关闭测试")
add_arg('use_gru',          bool,  True,   "是否使用GRUs模型，不使用RNN")
add_arg('use_gpu',          bool,  True,   "是否使用GPU训练")
add_arg('is_local',        bool,   True,   "是否在本地训练")
add_arg('share_rnn_weights',bool,  False,  "是否在RNN上共享权重")
add_arg('init_from_pretrained_model',   str, None,    "使用预训练模型的路径，当为None是不使用预训练模型")
add_arg('train_manifest',               str,   './dataset/manifest.train',    "训练的数据列表")
add_arg('dev_manifest',                 str,  './dataset/manifest.test',      "测试的数据列表")
add_arg('mean_std_path',                str,  './dataset/mean_std.npz',       "数据集的均值和标准值的npy文件路径")
add_arg('vocab_path',                   str,  './dataset/zh_vocab.txt',       "数据集的字典文件路径")
add_arg('output_model_dir',             str,  "./models",                     "保存训练模型的文件夹")
add_arg('augment_conf_path',            str,  './conf/augmentation.config',   "数据增强的配置文件，为json格式")
add_arg('specgram_type',                str,  'linear',    "对音频的预处理方式，有: linear, mfcc", choices=['linear', 'mfcc'])
add_arg('shuffle_method',               str,  'batch_shuffle_clipped',    "打乱数据的方法", choices=['instance_shuffle', 'batch_shuffle', 'batch_shuffle_clipped'])
args = parser.parse_args()


# 训练模型
def train():
    # 检测PaddlePaddle环境
    check_cuda(args.use_gpu)
    check_version()

    # 是否使用GPU
    if args.use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    # 获取训练数据生成器
    train_generator = DataGenerator(vocab_filepath=args.vocab_path,
                                    mean_std_filepath=args.mean_std_path,
                                    augmentation_config=io.open(args.augment_conf_path, mode='r', encoding='utf8').read(),
                                    max_duration=args.max_duration,
                                    min_duration=args.min_duration,
                                    specgram_type=args.specgram_type,
                                    place=place)

    # 获取测试数据生成器
    test_generator = DataGenerator(vocab_filepath=args.vocab_path,
                                   mean_std_filepath=args.mean_std_path,
                                   augmentation_config="{}",
                                   specgram_type=args.specgram_type,
                                   keep_transcription_text=True,
                                   place=place,
                                   is_training=False)
    # 获取训练数据
    train_batch_reader = train_generator.batch_reader_creator(manifest_path=args.train_manifest,
                                                              batch_size=args.batch_size,
                                                              shuffle_method=args.shuffle_method)
    # 获取测试数据
    test_batch_reader = test_generator.batch_reader_creator(manifest_path=args.dev_manifest,
                                                            batch_size=args.batch_size,
                                                            shuffle_method=None)
    # 获取DeepSpeech2模型
    ds2_model = DeepSpeech2Model(vocab_size=train_generator.vocab_size,
                                 num_conv_layers=args.num_conv_layers,
                                 num_rnn_layers=args.num_rnn_layers,
                                 rnn_layer_size=args.rnn_layer_size,
                                 use_gru=args.use_gru,
                                 share_rnn_weights=args.share_rnn_weights,
                                 place=place,
                                 init_from_pretrained_model=args.init_from_pretrained_model,
                                 output_model_dir=args.output_model_dir,
                                 vocab_list=test_generator.vocab_list)
    # 获取训练数据数量
    num_samples = get_data_len(args.train_manifest, args.max_duration, args.min_duration)
    print("[%s] 训练数据数量：%d\n" % (datetime.now(), num_samples))
    # 开始训练
    ds2_model.train(train_batch_reader=train_batch_reader,
                    dev_batch_reader=test_batch_reader,
                    learning_rate=args.learning_rate,
                    gradient_clipping=400,
                    batch_size=args.batch_size,
                    num_samples=num_samples,
                    num_epoch=args.num_epoch,
                    test_off=args.test_off)


def main():
    print_arguments(args)
    train()


if __name__ == '__main__':
    main()
