"""计算音频特征平均值和归一化，并保存到文件。"""
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import argparse
import functools
from data_utils.normalizer import FeatureNormalizer
from data_utils.augmentor.augmentation import AugmentationPipeline
from data_utils.featurizer.audio_featurizer import AudioFeaturizer
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('num_samples',      int,    5000,       "用于计算均值和标准值得音频数量")
add_arg('manifest_path',    str,    './dataset/manifest.train',   "用于计算均值和标准值的训练数据列表")
add_arg('output_path',      str,    './dataset/mean_std.npz',     "保存均值和标准值得numpy文件路径，后缀 (.npz).")
args = parser.parse_args()


def main():
    print_arguments(args)

    augmentation_pipeline = AugmentationPipeline('{}')
    audio_featurizer = AudioFeaturizer()

    def augment_and_featurize(audio_segment):
        augmentation_pipeline.transform_audio(audio_segment)
        return audio_featurizer.featurize(audio_segment)
    # 随机取指定的数量计算平均值归一化
    normalizer = FeatureNormalizer(
        mean_std_filepath=None,
        manifest_path=args.manifest_path,
        featurize_func=augment_and_featurize,
        num_samples=args.num_samples)
    # 将计算的结果保存的文件中
    normalizer.write_to_file(args.output_path)
    print('计算完成，文件保存于：%s' % args.output_path)


if __name__ == '__main__':
    main()
