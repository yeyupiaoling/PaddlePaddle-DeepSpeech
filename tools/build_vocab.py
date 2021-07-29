"""从数据列表中创建数据词汇表

每一个字符都有一个对应的ID
"""
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import argparse
import functools
from collections import Counter

from data_utils.utility import read_manifest
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('count_threshold',  int,    0,  "字符计数的截断阈值，0为不做限制")
add_arg('vocab_path',       str,    './dataset/zh_vocab.txt',                           "生成的数据词汇表文件")
add_arg('manifest_paths',   str,    './dataset/manifest.train,./dataset/manifest.test', "数据列表路径")
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
    with open(args.vocab_path, 'w', encoding='utf-8') as fout:
        for char, count in count_sorted:
            # 跳过指定的字符阈值，超过这大小的字符都忽略
            if count < args.count_threshold: break
            fout.write(char + '\n')
    print('数据词汇表已生成完成，保存与：%s' % args.vocab_path)


if __name__ == '__main__':
    main()
