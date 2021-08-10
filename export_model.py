import argparse
import functools
import paddle
from model_utils.model import DeepSpeech2Model
from utils.utility import add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('num_conv_layers',  int,    2,      "卷积层数量")
add_arg('num_rnn_layers',   int,    3,      "循环神经网络的数量")
add_arg('rnn_layer_size',   int,    1024,   "循环神经网络的大小")
add_arg('use_gpu',          bool,   True,   "是否使用GPU加载模型")
add_arg('vocab_path',       str,    './dataset/zh_vocab.txt',     "数据集的词汇表文件路径")
add_arg('resume_model',     str,    './models/param/50.pdparams', "恢复模型文件路径")
add_arg('save_model_path',  str,    './models/infer/',            "保存导出的预测模型文件夹路径")
args = parser.parse_args()

# 是否使用GPU
place = paddle.CUDAPlace(0) if args.use_gpu else paddle.CPUPlace()

with open(args.vocab_path, 'r', encoding='utf-8') as f:
    vocab_size = len(f.readlines())

# 获取DeepSpeech2模型，并设置为预测
ds2_model = DeepSpeech2Model(vocab_size=vocab_size,
                             num_conv_layers=args.num_conv_layers,
                             num_rnn_layers=args.num_rnn_layers,
                             rnn_layer_size=args.rnn_layer_size,
                             resume_model=args.resume_model,
                             place=place)

ds2_model.export_model(model_path=args.save_model_path)
print('成功导出模型，模型保存在：%s' % args.save_model_path)
