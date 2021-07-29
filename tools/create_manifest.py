"""根据标注文件创建数据列表"""
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import functools
import wave
from tqdm import tqdm
import json
import argparse
from utils.utility import add_arguments, print_arguments, change_rate

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('annotation_path',      str,  'dataset/annotation/',   '标注文件的路径')
add_arg('manifest_prefix',      str,  'dataset/',              '训练数据清单，包括音频路径和标注信息')
add_arg('is_change_frame_rate', bool, True,                    '是否统一改变音频为16000Hz，这会消耗大量的时间')
args = parser.parse_args()


# 创建数据列表
def create_manifest(annotation_path, manifest_path_prefix):
    json_lines = []
    durations = []
    # 获取全部的标注文件
    for annotation_text in os.listdir(annotation_path):
        print('正在创建%s的数据列表，请等待 ...' % annotation_text)
        annotation_text = os.path.join(annotation_path, annotation_text)
        # 读取标注文件
        with open(annotation_text, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            audio_path = line.split('\t')[0]
            try:
                # 过滤非法的字符
                text = is_ustr(line.split('\t')[1].replace('\n', '').replace('\r', ''))
                # 重新调整音频格式并保存
                if args.is_change_frame_rate:
                    change_rate(audio_path)
                # 获取音频的长度
                f_wave = wave.open(audio_path, "rb")
                duration = f_wave.getnframes() / f_wave.getframerate()
                durations.append(duration)
                json_lines.append(
                    json.dumps(
                        {
                            'audio_filepath': audio_path,
                            'duration': duration,
                            'text': text
                        },
                        ensure_ascii=False))
            except Exception as e:
                print(e)
                continue

    # 将音频的路径，长度和标签写入到数据列表中
    f_train = open(os.path.join(manifest_path_prefix, 'manifest.train'), 'w', encoding='utf-8')
    f_test = open(os.path.join(manifest_path_prefix, 'manifest.test'), 'w', encoding='utf-8')
    for i, line in enumerate(json_lines):
        if i % 500 == 0:
            f_test.write(line + '\n')
        else:
            f_train.write(line + '\n')
    f_train.close()
    f_test.close()
    print("创建数量列表完成，全部数据一共[%d]小时!" % int(sum(durations) / 3600))


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
def create_noise(path='dataset/audio/noise'):
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
            json_lines.append(
                json.dumps(
                    {
                        'audio_filepath': audio_path,
                        'duration': duration,
                        'text': text
                    },
                    ensure_ascii=False))
        except:
            continue
    with open(os.path.join(args.manifest_prefix, 'manifest.noise'), 'w', encoding='utf-8') as f_noise:
        for json_line in json_lines:
            f_noise.write(json_line + '\n')


def main():
    print_arguments(args)
    create_manifest(annotation_path=args.annotation_path,
                    manifest_path_prefix=args.manifest_prefix)


if __name__ == '__main__':
    # 生成噪声的数据列表
    create_noise()
    # 生成训练数据列表
    main()
