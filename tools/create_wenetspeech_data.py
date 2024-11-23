import argparse
import json
import os

import ijson
from loguru import logger
from tqdm import tqdm

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--wenetspeech_json',  type=str,    default='/media/WenetSpeech.json',  help="WenetSpeech的标注json文件路径")
parser.add_argument('--dataset_dir',       type=str,    default='dataset/',    help="存放数量列表的文件夹路径")
parser.add_argument('--num_workers',       type=int,    default=8,             help="把opus格式转换为wav格式的线程数量")
args = parser.parse_args()


# 处理WenetSpeech数据
def process_wenetspeech(long_audio_path, segments_lists):
    if not os.path.exists(long_audio_path):
        return None
    data = []
    for segment_file in segments_lists:
        try:
            start_time = float(segment_file['begin_time'])
            end_time = float(segment_file['end_time'])
            text = segment_file['text']
            confidence = segment_file['confidence']
            if confidence < 0.95: continue
        except Exception:
            logger.warning(f'{segment_file} something is wrong, skipped')
            continue
        else:
            line = dict(audio_filepath=long_audio_path,
                        text=text,
                        duration=round(end_time - start_time, 3),
                        start_time=round(start_time, 3),
                        end_time=round(end_time, 3))
            data.append(line)
    data_type = long_audio_path.split('/')[-4]
    return {"data": data, "data_type": data_type}


# 获取标注信息
def get_data(wenetspeech_json):
    data_list = []
    input_dir = os.path.dirname(wenetspeech_json)
    i = 0
    # 开始读取数据，因为文件太大，无法获取进度
    with open(wenetspeech_json, 'r', encoding='utf-8') as f:
        objects = ijson.items(f, 'audios.item')
        logger.info("开始读取数据")
        while True:
            try:
                long_audio = objects.__next__()
                i += 1
                try:
                    long_audio_path = os.path.realpath(os.path.join(input_dir, long_audio['path']))
                    aid = long_audio['aid']
                    segments_lists = long_audio['segments']
                    assert (os.path.exists(long_audio_path))
                except AssertionError:
                    logger.warning(f'{long_audio_path} 不存在或者已经处理过自动删除了，跳过')
                    continue
                except Exception:
                    logger.warning(f'{aid} 数据读取错误，跳过')
                    continue
                else:
                    data_list.append((long_audio_path.replace('\\', '/'), segments_lists))
            except StopIteration:
                logger.info("数据读取完成")
                break
    return data_list


def main():
    all_data = get_data(args.wenetspeech_json)
    logger.info(f'总数据量为：{len(all_data)}')
    # 写入到数据列表
    train_data, test_net_data, test_meeting_data = [], [], []
    for long_audio_path, segments_lists in tqdm(all_data):
        data_type = long_audio_path.split('/')[-4]
        result = process_wenetspeech(long_audio_path, segments_lists)
        if result is None:
            logger.error(f'获取不到{long_audio_path}，已跳过')
            continue
        result_data = result['data']
        if data_type == 'train':
            for line in result_data:
                train_data.append(line)
        elif data_type == 'test_net' or data_type == 'test_meeting':
            for line in result_data:
                if data_type == 'test_net':
                    test_net_data.append(line)
                elif data_type == 'test_meeting':
                    test_meeting_data.append(line)
    os.makedirs(args.dataset_dir, exist_ok=True)
    # 训练数据列表
    with open(os.path.join(args.dataset_dir, 'wenetspeech.json'), 'w', encoding='utf-8') as f:
        train_data.sort(key=lambda x: x["duration"], reverse=False)
        for line in train_data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    # 测试数据列表
    with open(os.path.join(args.dataset_dir, 'test_net.json'), 'w', encoding='utf-8') as f:
        test_net_data.sort(key=lambda x: x["duration"], reverse=False)
        for line in test_net_data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    with open(os.path.join(args.dataset_dir, 'test_meeting.json'), 'w', encoding='utf-8') as f:
        test_meeting_data.sort(key=lambda x: x["duration"], reverse=False)
        for line in test_meeting_data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    test_data = test_net_data + test_meeting_data
    with open(os.path.join(args.dataset_dir, 'test.json'), 'w', encoding='utf-8') as f:
        test_data.sort(key=lambda x: x["duration"], reverse=False)
        for line in test_data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
