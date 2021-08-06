import argparse
import functools

import paddle

from data_utils.data import DataGenerator
from model_utils.model import DeepSpeech2Model
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('wav_path',         str,    './dataset/test.wav', "预测音频的路径")
add_arg('num_conv_layers',  int,    2,      "卷积层数量")
add_arg('num_rnn_layers',   int,    3,      "循环神经网络的数量")
add_arg('rnn_layer_size',   int,    1024,   "循环神经网络的大小")
add_arg('use_gru',          bool,   True,   "是否使用GRUs模型，不使用RNN")
add_arg('use_gpu',          bool,   True,   "是否使用GPU训练")
add_arg('share_rnn_weights',bool,   False,   "是否在RNN上共享权重")
add_arg('mean_std_path',    str,    './dataset/mean_std.npz',    "数据集的均值和标准值的npy文件路径")
add_arg('vocab_path',       str,    './dataset/zh_vocab.txt',    "数据集的词汇表文件路径")
add_arg('pretrained_model', str,    './models/epoch_2/',         "训练保存的模型文件夹路径")
add_arg('save_model_path',  str,    './models/infer/',           "保存导出的预测模型文件夹路径")
args = parser.parse_args()

# 是否使用GPU
place = paddle.CUDAPlace(0) if args.use_gpu else paddle.CPUPlace()

# 获取数据生成器，处理数据和获取字典需要
data_generator = DataGenerator(vocab_filepath=args.vocab_path,
                               mean_std_filepath=args.mean_std_path,
                               augmentation_config='{}',
                               keep_transcription_text=True,
                               place=place,
                               is_training=False)

# 获取DeepSpeech2模型，并设置为预测
ds2_model = DeepSpeech2Model(vocab_size=data_generator.vocab_size,
                             num_conv_layers=args.num_conv_layers,
                             num_rnn_layers=args.num_rnn_layers,
                             rnn_layer_size=args.rnn_layer_size,
                             use_gru=args.use_gru,
                             init_from_pretrained_model=args.pretrained_model,
                             place=place,
                             share_rnn_weights=args.share_rnn_weights)


def main():
    print_arguments(args)
    # 加载音频文件，并进行预处理
    feature = data_generator.process_utterance(args.wav_path, "")[0]
    # 执行预测
    ds2_model.export_model(data_feature=feature, model_path=args.save_model_path)
    print('成功导出模型，模型保存在：%s' % args.save_model_path)


if __name__ == "__main__":
    main()
