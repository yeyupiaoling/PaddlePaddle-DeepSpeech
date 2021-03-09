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
add_arg('batch_size',      int,   16, "Minibatch size.")
add_arg('num_epoch',       int,   200, "# of training epochs.")
add_arg('num_conv_layers', int,   2, "# of convolution layers.")
add_arg('num_rnn_layers',  int,   3, "# of recurrent layers.")
add_arg('rnn_layer_size',  int,   2048, "# of recurrent cells per layer.")
add_arg('learning_rate',   float, 5e-5, "Learning rate.")
add_arg('max_duration',    float, 15.0, "Longest audio duration allowed.")
add_arg('min_duration',    float, 1.0, "Shortest audio duration allowed.")
add_arg('test_off',        bool,  False, "Turn off testing.")
add_arg('use_sortagrad',   bool,  True, "Use SortaGrad or not.")
add_arg('use_gpu',         bool,  True, "Use GPU or not.")
add_arg('use_gru',         bool,  True, "Use GRUs instead of simple RNNs.")
add_arg('is_local',        bool,  True, "Use pserver or not.")
add_arg('share_rnn_weights',bool, False, "Share input-hidden weights across bi-directional RNNs. Not for GRU.")
add_arg('init_from_pretrained_model',   str, None,    "If None, the training starts from scratch, otherwise, it resumes from the pre-trained model.")
add_arg('train_manifest',               str,   './dataset/manifest.train',    "Filepath of train manifest.")
add_arg('dev_manifest',                 str,  './dataset/manifest.dev',     "Filepath of validation manifest.")
add_arg('mean_std_path',                str,  './dataset/mean_std.npz',     "Filepath of normalizer's mean & std.")
add_arg('vocab_path',                   str,  './dataset/zh_vocab.txt',     "Filepath of vocabulary.")
add_arg('output_model_dir',             str,  "./models",       "Directory for saving checkpoints.")
add_arg('augment_conf_path',            str,  './conf/augmentation.config',    "Filepath of augmentation configuration file (json-format).")
add_arg('specgram_type',                str,  'linear',    "Audio feature type. Options: linear, mfcc.", choices=['linear', 'mfcc'])
add_arg('shuffle_method',               str,  'batch_shuffle_clipped',    "Shuffle method.", choices=['instance_shuffle', 'batch_shuffle', 'batch_shuffle_clipped'])
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
                                                              sortagrad=args.use_sortagrad if args.init_from_pretrained_model is None else False,
                                                              shuffle_method=args.shuffle_method)
    # 获取测试数据
    test_batch_reader = test_generator.batch_reader_creator(manifest_path=args.dev_manifest,
                                                            batch_size=args.batch_size,
                                                            sortagrad=False,
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
