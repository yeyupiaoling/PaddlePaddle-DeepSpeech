from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import codecs
import functools
import soundfile
import json
import argparse
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
parser.add_argument("--annotation_path",
                    default="./dataset/annotation/",
                    type=str,
                    help="Sound annotation text save path. (default: %(default)s)")
parser.add_argument("--manifest_prefix",
                    default="./dataset/",
                    type=str,
                    help="Filepath prefix for output manifests. (default: %(default)s)")
args = parser.parse_args()


def create_manifest(annotation_path, manifest_path_prefix):
    json_lines = []
    for annotation_text in os.listdir(annotation_path):
        print('The %s manifest takes a long time to create. Please wait ...' % annotation_text)
        annotation_text = os.path.join(annotation_path, annotation_text)
        with codecs.open(annotation_text, 'r', 'utf-8') as f:
            lines = f.readlines()
        for line in lines:
            audio_path = line.split('\t')[0]

            text = is_ustr(line.split('\t')[1].replace('\n', '').replace('\r', ''))
            audio_data, samplerate = soundfile.read(audio_path)
            duration = float(len(audio_data) / samplerate)
            json_lines.append(
                json.dumps(
                    {
                        'audio_filepath': audio_path,
                        'duration': duration,
                        'text': text
                    },
                    ensure_ascii=False))

    f_train = codecs.open(os.path.join(manifest_path_prefix, 'manifest.train'), 'w', 'utf-8')
    f_dev = codecs.open(os.path.join(manifest_path_prefix, 'manifest.dev'), 'w', 'utf-8')
    f_test = codecs.open(os.path.join(manifest_path_prefix, 'manifest.test'), 'w', 'utf-8')
    for i, line in enumerate(json_lines):
        if i % 50 == 0:
            if i % 100 == 0:
                f_dev.write(line + '\n')
            else:
                f_test.write(line + '\n')
        else:
            f_train.write(line + '\n')
    f_train.close()
    f_dev.close()
    f_test.close()


def is_ustr(in_str):
    out_str = ''
    for i in range(len(in_str)):
        if is_uchar(in_str[i]):
            out_str = out_str + in_str[i]
        else:
            out_str = out_str + ' '
    return ''.join(out_str.split())


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


def main():
    print_arguments(args)
    create_manifest(annotation_path=args.annotation_path,
                    manifest_path_prefix=args.manifest_prefix)


if __name__ == '__main__':
    main()
