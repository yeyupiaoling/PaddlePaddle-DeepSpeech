import json
import math
import os

import numpy as np
import random

import paddle
from loguru import logger
from tqdm import tqdm
from paddle.io import Dataset, DataLoader
from data_utils.utils import read_manifest
from yeaudio.audio import AudioSegment
from data_utils.featurizer.audio_featurizer import AudioFeaturizer


class FeatureNormalizer(object):
    """音频特征归一化类

    :param mean_istd_filepath: 均值和标准值的文件路径
    """

    def __init__(self, mean_istd_filepath, eps=1e-20):
        self.mean_std_filepath = mean_istd_filepath
        # 读取归一化文件
        if os.path.exists(mean_istd_filepath):
            self.mean, self.istd = self._read_mean_istd_from_file(mean_istd_filepath)
            self.istd = np.maximum(self.istd, eps)

    @staticmethod
    def _read_mean_istd_from_file(filepath):
        """从文件中加载均值和标准值"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            mean = np.array(data["mean"], dtype=np.float32)
            istd = np.array(data["istd"], dtype=np.float32)
        return mean, istd

    def compute_mean_istd(self,
                          manifest_path,
                          num_workers=4,
                          batch_size=32,
                          num_samples=5000):
        """从随机抽样的实例中计算均值和标准值，并写入到文件中

        :param manifest_path: 数据列表文件路径
        :param num_workers: 计算的线程数量
        :param batch_size: 计算的批量大小
        :param num_samples: 用于计算均值和标准值的音频数量
        """
        manifest = read_manifest(manifest_path)
        if num_samples < 0 or num_samples > len(manifest):
            sampled_manifest = manifest
        else:
            sampled_manifest = random.sample(manifest, num_samples)
        logger.info('开始抽取{}条数据计算均值和标准值...'.format(len(sampled_manifest)))
        dataset = NormalizerDataset(sampled_manifest)
        test_loader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)
        with paddle.no_grad():
            # 求总和
            std, means = None, None
            number = 0
            for std1, means1, number1 in tqdm(test_loader()):
                number += number1
                if means is None:
                    means = means1
                else:
                    means += means1
                if std is None:
                    std = std1
                else:
                    std += std1
            # 求总和的均值和标准值
            for i in range(len(means)):
                means[i] /= number
                std[i] = std[i] / number - means[i] * means[i]
                if std[i] < 1.0e-20:
                    std[i] = 1.0e-20
                std[i] = math.sqrt(std[i])
        istd = 1.0 / std
        # 写入到文件中
        data = {'mean': means.tolist(),
                'istd': istd.tolist()}
        with open(self.mean_std_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f)


class NormalizerDataset(Dataset):
    def __init__(self, sampled_manifest):
        super(NormalizerDataset, self).__init__()
        self.audio_featurizer = AudioFeaturizer()
        self.sampled_manifest = sampled_manifest

    def __getitem__(self, idx):
        instance = self.sampled_manifest[idx]
        # 获取音频特征
        audio = AudioSegment.from_file(instance["audio_filepath"])
        feature = self.audio_featurizer.featurize(audio)
        return feature.astype(np.float32), 0

    def __len__(self):
        return len(self.sampled_manifest)


def collate_fn(features):
    std, means = None, None
    number = 0
    for feature, _ in features:
        number += feature.shape[0]
        sums = np.sum(feature, axis=0)
        if means is None:
            means = sums
        else:
            means += sums
        square_sums = np.sum(np.square(feature), axis=0)
        if std is None:
            std = square_sums
        else:
            std += square_sums
    return std, means, number
