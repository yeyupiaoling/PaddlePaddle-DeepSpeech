"""Prepare THCHS-30 dataset

Download, unpack and create data list.
i.e.
cd ../
PYTHONPATH=.:$PYTHONPATH python data/thchs_30.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import codecs
import os

from data_utils.utility import download, unpack

# URL_ROOT = 'http://www.openslr.org/resources/18'
URL_ROOT = 'http://192.168.1.119:55000'
DATA_URL = URL_ROOT + '/data_thchs30.tgz'
MD5_DATA = '2d2252bde5c8429929e1841d4cb95e90'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target_dir",
    default="./dataset/audio/",
    type=str,
    help="Directory to save the dataset. (default: %(default)s)")
parser.add_argument(
    "--annotation_text",
    default="./dataset/annotation/",
    type=str,
    help="Sound annotation text save path. (default: %(default)s)")
args = parser.parse_args()


def create_annotation_text(data_dir, annotation_path):
    print('Create THCHS-30 annotation text ...')
    f_a = codecs.open(os.path.join(annotation_path, 'thchs_30.txt'), 'w', 'utf-8')
    data_path = 'data'
    for file in os.listdir(os.path.join(data_dir, data_path)):
        if '.trn' in file:
            file = os.path.join(data_dir, data_path, file)
            with codecs.open(file, 'r', 'utf-8') as f:
                line = f.readline()
                line = ''.join(line.split())
            f_a.write(file[:-4] + '\t' + line + '\n')
    f_a.close()


def prepare_dataset(url, md5sum, target_dir, annotation_path):
    """Download, unpack and create manifest file."""
    data_dir = os.path.join(target_dir, 'data_thchs30')
    if not os.path.exists(data_dir):
        filepath = download(url, md5sum, target_dir)
        unpack(filepath, target_dir)
        os.remove(filepath)
    else:
        print("Skip downloading and unpacking. THCHS-30 data already exists in %s." % target_dir)
    create_annotation_text(data_dir, annotation_path)


def main():
    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)

    prepare_dataset(
        url=DATA_URL,
        md5sum=MD5_DATA,
        target_dir=args.target_dir,
        annotation_path=args.annotation_text)


if __name__ == '__main__':
    main()
