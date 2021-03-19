"""Contains common utility functions."""

import distutils.util

import librosa
import soundfile

from data_utils.utility import read_manifest


def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


# 获取训练数据长度
def get_data_len(manifest_path, max_duration, min_duration):
    manifest = read_manifest(manifest_path=manifest_path,
                             max_duration=max_duration,
                             min_duration=min_duration)
    return len(manifest)


# 改变音频采样率为16000Hz
def change_rate(audio_path):
    audio_path = audio_path.replace('\\', '/')
    data, sr = soundfile.read(audio_path)
    if sr != 16000:
        data, sr = librosa.load(audio_path, sr=16000)
        soundfile.write(audio_path, data, samplerate=16000)
