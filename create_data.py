import argparse
import functools
import json
import os
import wave
from collections import Counter
from zhconv import convert

import numpy as np
import soundfile
from tqdm import tqdm

from data_utils.normalizer import FeatureNormalizer
from utils.utility import add_arguments, print_arguments, read_manifest, change_rate

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('annotation_path',      str,  'dataset/annotation/',      '标注文件的路径，如果annotation_path包含了test.txt，就全部使用test.txt的数据作为测试数据')
add_arg('manifest_prefix',      str,  'dataset/',                 '训练数据清单，包括音频路径和标注信息')
add_arg('is_change_frame_rate', bool, True,                       '是否统一改变音频为16000Hz，这会消耗大量的时间')
add_arg('max_test_manifest',    int,  10000,                      '最大的测试数据数量')
add_arg('count_threshold',      int,  2,                          '字符计数的截断阈值，0为不做限制')
add_arg('vocab_path',           str,  'dataset/zh_vocab.txt',     '生成的数据字典文件')
add_arg('num_workers',          int,   8,                         '读取数据的线程数量')
add_arg('manifest_paths',       str,  'dataset/manifest.train',   '数据列表路径')
add_arg('num_samples',          int,  1000000,                    '用于计算均值和标准值得音频数量，当为-1使用全部数据')
add_arg('output_path',          str,  './dataset/mean_std.npz',   '保存均值和标准值得numpy文件路径，后缀 (.npz).')
args = parser.parse_args()


# 创建数据列表
def create_manifest(annotation_path, manifest_path_prefix):
    data_list = []
    test_list = []
    durations_all = []
    duration_0_10 = 0
    duration_10_20 = 0
    duration_20 = 0
    # 获取全部的标注文件
    for annotation_text in os.listdir(annotation_path):
        durations = []
        print('正在创建%s的数据列表，请等待 ...' % annotation_text)
        annotation_text_path = os.path.join(annotation_path, annotation_text)
        # 读取标注文件
        with open(annotation_text_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            audio_path = line.split('\t')[0]
            try:
                # 过滤非法的字符
                text = is_ustr(line.split('\t')[1].replace('\n', '').replace('\r', ''))
                # 保证全部都是简体
                text = convert(text, 'zh-cn')
                # 重新调整音频格式并保存
                if args.is_change_frame_rate:
                    change_rate(audio_path)
                # 获取音频的长度
                f_wave = wave.open(audio_path, "rb")
                duration = f_wave.getnframes() / f_wave.getframerate()
                if duration <= 10:
                    duration_0_10 += 1
                elif 10 < duration <= 20:
                    duration_10_20 += 1
                else:
                    duration_20 += 1
                durations.append(duration)
                d = json.dumps(
                        {
                            'audio_filepath': audio_path.replace('\\', '/'),
                            'duration': duration,
                            'text': text
                        },
                        ensure_ascii=False)

                if annotation_text == 'test.txt':
                    test_list.append(d)
                else:
                    data_list.append(d)
            except Exception as e:
                print(e)
                continue
        durations_all.append(sum(durations))
        print("%s数据一共[%d]小时!" % (annotation_text, int(sum(durations) / 3600)))
        print("0-10秒的数量：%d，10-20秒的数量：%d，大于20秒的数量：%d" % (duration_0_10, duration_10_20, duration_20))

    # 将音频的路径，长度和标签写入到数据列表中
    f_train = open(os.path.join(manifest_path_prefix, 'manifest.train'), 'w', encoding='utf-8')
    f_test = open(os.path.join(manifest_path_prefix, 'manifest.test'), 'w', encoding='utf-8')
    for line in test_list:
        f_test.write(line + '\n')
    interval = 500
    if len(data_list) / 500 > args.max_test_manifest:
        interval = len(data_list) // args.max_test_manifest
    for i, line in enumerate(data_list):
        if i % interval == 0 and i != 0:
            if len(test_list) == 0:
                f_test.write(line + '\n')
            else:
                f_train.write(line + '\n')
        else:
            f_train.write(line + '\n')
    f_train.close()
    f_test.close()
    print("创建数量列表完成，全部数据一共[%d]小时!" % int(sum(durations_all) / 3600))


# 过滤非文字的字符
def is_ustr(in_str):
    out_str = ''
    for i in range(len(in_str)):
        if is_uchar(in_str[i]):
            out_str = out_str + in_str[i]
        else:
            out_str = out_str + ' '
    return ''.join(out_str.split())


# 判断是否为文字字符
def is_uchar(uchar):
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    if u'\u0030' <= uchar <= u'\u0039':
        return False
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
        return False
    if uchar in ('-', ',', '.', '>', '?'):
        return False
    return False


# 生成噪声的数据列表
def create_noise(path='dataset/audio/noise', min_duration=30):
    if not os.path.exists(path):
        print('噪声音频文件为空，已跳过！')
        return
    json_lines = []
    print('正在创建噪声数据列表，路径：%s，请等待 ...' % path)
    for file in tqdm(os.listdir(path)):
        audio_path = os.path.join(path, file)
        try:
            # 噪声的标签可以标记为空
            text = ""
            # 重新调整音频格式并保存
            if args.is_change_frame_rate:
                change_rate(audio_path)
            f_wave = wave.open(audio_path, "rb")
            duration = f_wave.getnframes() / f_wave.getframerate()
            # 拼接音频
            if duration < min_duration:
                wav = soundfile.read(audio_path)[0]
                data = wav
                for i in range(int(min_duration / duration) + 1):
                    data = np.hstack([data, wav])
                soundfile.write(audio_path, data, samplerate=16000)
                f_wave = wave.open(audio_path, "rb")
                duration = f_wave.getnframes() / f_wave.getframerate()
            json_lines.append(
                json.dumps(
                    {
                        'audio_filepath': audio_path.replace('\\', '/'),
                        'duration': duration,
                        'text': text
                    },
                    ensure_ascii=False))
        except Exception as e:
            continue
    with open(os.path.join(args.manifest_prefix, 'manifest.noise'), 'w', encoding='utf-8') as f_noise:
        for json_line in json_lines:
            f_noise.write(json_line + '\n')


# 获取全部字符
def count_manifest(counter, manifest_path):
    manifest_jsons = read_manifest(manifest_path)
    for line_json in manifest_jsons:
        for char in line_json['text']:
            counter.update(char)


# 计算数据集的均值和标准值
def compute_mean_std(manifest_path, num_samples, output_path):
    # 随机取指定的数量计算平均值归一化
    normalizer = FeatureNormalizer(mean_std_filepath=None,
                                   manifest_path=manifest_path,
                                   num_samples=num_samples,
                                   num_workers=args.num_workers)
    # 将计算的结果保存的文件中
    normalizer.write_to_file(output_path)
    print('计算的均值和标准值已保存在 %s！' % output_path)


def main():
    print_arguments(args)
    print('开始生成数据列表...')
    create_manifest(annotation_path=args.annotation_path,
                    manifest_path_prefix=args.manifest_prefix)
    print('='*70)
    print('开始生成噪声数据列表...')
    create_noise(path='dataset/audio/noise')
    print('='*70)

    print('开始生成数据字典...')
    counter = Counter()
    # 获取全部数据列表中的标签字符
    count_manifest(counter, args.manifest_paths)
    # 为每一个字符都生成一个ID
    count_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    with open(args.vocab_path, 'w', encoding='utf-8') as fout:
        fout.write('<blank>\t-1\n')
        for char, count in count_sorted:
            # 跳过指定的字符阈值，超过这大小的字符都忽略
            if count < args.count_threshold: break
            fout.write('%s\t%d\n' % (char, count))
    print('数据词汇表已生成完成，保存与：%s' % args.vocab_path)
    print('='*70)

    print('开始抽取%s条数据计算均值和标准值...' % args.num_samples)
    compute_mean_std(args.manifest_paths, args.num_samples, args.output_path)
    print('='*70)


if __name__ == '__main__':
    main()
