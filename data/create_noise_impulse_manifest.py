"""Create manifest data

Convert the sound and annotation into manifest data
i.e.
cd ../
PYTHONPATH=.:$PYTHONPATH python data/create_noise_impulse_manifest.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import codecs
import os

import soundfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--noise_path",
    default="./dataset/audio/noise",
    type=str,
    help="Noises path, these noise duration in the 20 or so.  (default: %(default)s)")
parser.add_argument(
    "--impulse_path",
    default="./dataset/audio/impulse",
    type=str,
    help="Impulse path, these impulse duration in the 20 or so.  (default: %(default)s)")
parser.add_argument(
    "--manifest_prefix",
    default="./dataset/",
    type=str,
    help="Filepath prefix for output manifests. (default: %(default)s)")
args = parser.parse_args()


def write_manifest(audio_path, manifest_path_prefix):
    json_lines = []
    lines = os.listdir(audio_path)
    for line in lines:
        audio_data, samplerate = soundfile.read(os.path.join(audio_path, line))
        duration = float(len(audio_data) / samplerate)
        json_lines.append(
            json.dumps(
                {
                    'audio_filepath': os.path.join(audio_path, line),
                    'duration': duration
                },
                ensure_ascii=False))

    f = codecs.open(manifest_path_prefix, 'w', 'utf-8')
    for i, line in enumerate(json_lines):
        f.write(line + '\n')


if __name__ == '__main__':
    write_manifest(
        audio_path=args.noise_path,
        manifest_path_prefix=os.path.join(args.manifest_prefix, 'manifest.noise'))

    write_manifest(
        audio_path=args.impulse_path,
        manifest_path_prefix=os.path.join(args.manifest_prefix, 'manifest.impulse'))