import distutils.util
import json
import urllib.request

from loguru import logger
from tqdm import tqdm


def print_arguments(args=None, configs=None, title=None):
    if args:
        logger.info("----------- 额外配置参数 -----------")
        for arg, value in sorted(vars(args).items()):
            logger.info(f"{arg}: {value}")
        logger.info("------------------------------------------------")
    if configs:
        title = title if title else "配置文件参数"
        logger.info(f"----------- {title} -----------")
        for arg, value in sorted(configs.items()):
            if arg == 'vocabulary': value = str(value)[:30] + ' ......'
            if isinstance(value, dict):
                logger.info(f"{arg}:")
                for a, v in sorted(value.items()):
                    if isinstance(v, dict):
                        logger.info(f"\t{a}:")
                        for a1, v1 in sorted(v.items()):
                            logger.info(f"\t\t{a1}: {v1}")
                    else:
                        logger.info(f"\t{a}: {v}")
            else:
                logger.info(f"{arg}: {value}")
        logger.info("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument("--" + argname,
                           default=default,
                           type=type,
                           help=help + ' 默认: %(default)s.',
                           **kwargs)


def download(url: str, download_target: str):
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True,
                  unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))


class DictObject(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict_to_object(dict_obj):
    if not isinstance(dict_obj, dict):
        return dict_obj
    inst = DictObject()
    for k, v in dict_obj.items():
        inst[k] = dict_to_object(v)
    return inst


def labels_to_string(label, vocabulary, blank_index=0):
    labels = []
    for l in label:
        index_list = [index for index in l if index != blank_index and index != -1]
        labels.append(
            (''.join([vocabulary[index] for index in index_list])).replace('<space>', ' ').replace('<unk>', ''))
    return labels


def read_manifest(manifest_path, max_duration=float('inf'), min_duration=0.5):
    """解析数据列表
    持续时间在[min_duration, max_duration]之外的实例将被过滤。

    :param manifest_path: 数据列表的路径
    :type manifest_path: str
    :param max_duration: 过滤的最长音频长度
    :type max_duration: float
    :param min_duration: 过滤的最短音频长度
    :type min_duration: float
    :return: 数据列表，JSON格式
    :rtype: list
    :raises IOError: If failed to parse the manifest.
    """
    manifest = []
    for json_line in open(manifest_path, 'r', encoding='utf-8'):
        try:
            json_data = json.loads(json_line)
        except Exception as e:
            raise IOError("Error reading manifest: %s" % str(e))
        if max_duration >= json_data["duration"] >= min_duration:
            manifest.append(json_data)
    return manifest
