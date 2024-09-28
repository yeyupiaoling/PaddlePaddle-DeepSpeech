import argparse
import functools
import os

import paddle
from loguru import logger

from data_utils.featurizer.audio_featurizer import AudioFeaturizer
from data_utils.featurizer.text_featurizer import TextFeaturizer
from model_utils.model import DeepSpeech2Model
from utils.checkpoint import load_pretrained
from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('num_rnn_layers',    int,    3,      "循环神经网络的数量")
add_arg('rnn_layer_size',    int,    1024,   "循环神经网络的大小")
add_arg('mean_istd_path',    str,    './dataset/mean_istd.json',   "均值和标准值得json文件路径，后缀 (.json)")
add_arg('vocab_path',        str,    './dataset/vocabulary.txt',   "数据集的词汇表文件路径")
add_arg('pretrained_model',  str,    './models/epoch_15/',         "训练模型文件路径")
add_arg('save_model_path',   str,    './models/infer/',            "保存导出的预测模型文件夹路径")
args = parser.parse_args()
print_arguments(args)


audio_featurizer = AudioFeaturizer(mode="infer")
textFeaturizer = TextFeaturizer(args.vocab_path)

model = DeepSpeech2Model(input_dim=audio_featurizer.feature_dim,
                         vocab_size=textFeaturizer.vocab_size,
                         mean_istd_path=args.mean_istd_path,
                         num_rnn_layers=args.num_rnn_layers,
                         rnn_layer_size=args.rnn_layer_size)
model = load_pretrained(model, args.pretrained_model)
infer_model = model.export()

os.makedirs(args.save_model_path, exist_ok=True)
infer_model_path = os.path.join(args.save_model_path, 'model')
paddle.jit.save(infer_model, infer_model_path)
logger.info("预测模型已保存：{}".format(infer_model_path))
