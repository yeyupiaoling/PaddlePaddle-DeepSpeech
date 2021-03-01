"""从数据列表中创建数据字典

每一个字符都有一个对应的ID
"""

import argparse
import codecs
import functools
from collections import Counter

from data_utils.utility import read_manifest
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('count_threshold',  int,    0,  "Truncation threshold for char counts.")
add_arg('vocab_path',       str,    './dataset/zh_vocab.txt', "Filepath to write the vocabulary.")
add_arg('manifest_paths',   str,    './dataset/manifest.train,./dataset/manifest.dev', "Filepaths of manifests for building vocabulary. You can provide multiple manifest files.")
args = parser.parse_args()


# 统计字符
def count_manifest(counter, manifest_path):
    manifest_jsons = read_manifest(manifest_path)
    for line_json in manifest_jsons:
        for char in line_json['text']:
            counter.update(char)


def main():
    print_arguments(args)

    counter = Counter()
    # 获取全部数据列表
    manifest_paths = [path for path in args.manifest_paths.split(',')]
    # 获取全部数据列表中的标签字符
    for manifest_path in manifest_paths:
        count_manifest(counter, manifest_path)
    # 为每一个字符都生成一个ID
    count_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    with codecs.open(args.vocab_path, 'w', 'utf-8') as fout:
        for char, count in count_sorted:
            # 跳过指定的字符阈值，超过这大小的字符都忽略
            if count < args.count_threshold: break
            fout.write(char + '\n')


if __name__ == '__main__':
    main()
