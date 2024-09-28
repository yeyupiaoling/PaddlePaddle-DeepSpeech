import json
import os
import shutil

import paddle

from loguru import logger


def load_pretrained(model, pretrained_model):
    """加载预训练模型

    :param model: 使用的模型
    :param pretrained_model: 预训练模型路径
    """
    # 加载预训练模型
    if pretrained_model is None: return model
    if os.path.isdir(pretrained_model):
        pretrained_model = os.path.join(pretrained_model, 'model.pdparams')
    assert os.path.exists(pretrained_model), f"{pretrained_model} 模型不存在！"
    model_dict = model.state_dict()
    model_state_dict = paddle.load(pretrained_model)
    # 过滤不存在的参数
    for name, weight in model_dict.items():
        if name in model_state_dict.keys():
            if list(weight.shape) != list(model_state_dict[name].shape):
                logger.warning('{} not used, shape {} unmatched with {} in model.'.
                               format(name, list(model_state_dict[name].shape), list(weight.shape)))
                model_state_dict.pop(name, None)
        else:
            logger.warning('Lack weight: {}'.format(name))
    # 加载权重
    missing_keys, unexpected_keys = model.set_state_dict(model_state_dict)
    if len(unexpected_keys) > 0:
        logger.warning('Unexpected key(s) in state_dict: {}. '
                       .format(', '.join('"{}"'.format(k) for k in unexpected_keys)))
    if len(missing_keys) > 0:
        logger.warning('Missing key(s) in state_dict: {}. '
                       .format(', '.join('"{}"'.format(k) for k in missing_keys)))
    logger.info('成功加载预训练模型：{}'.format(pretrained_model))
    return model


def load_checkpoint(resume_model, model, optimizer):
    """加载模型

    :param resume_model: 恢复训练的模型路径
    :param model: 使用的模型
    :param optimizer: 使用的优化方法
    """
    last_epoch1 = 0
    error_rate1 = 1.0

    def load_model(model_path):
        assert os.path.exists(os.path.join(model_path, 'model.pdparams')), "模型参数文件不存在！"
        assert os.path.exists(os.path.join(model_path, 'optimizer.pdopt')), "优化方法参数文件不存在！"
        state_dict = paddle.load(os.path.join(model_path, 'model.pdparams'))
        missing_keys, unexpected_keys = model.set_state_dict(state_dict)
        assert len(missing_keys) == len(unexpected_keys) == 0, "模型参数加载失败，参数权重不匹配，请可以考虑当做预训练模型！"
        optimizer.set_state_dict(paddle.load(os.path.join(model_path, 'optimizer.pdopt')))
        # 自动混合精度参数
        with open(os.path.join(model_path, 'model.state'), 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            last_epoch = json_data['last_epoch']
            if 'cer' in json_data.keys():
                error_rate = abs(json_data['cer'])
            if 'wer' in json_data.keys():
                error_rate = abs(json_data['wer'])
        logger.info('成功恢复模型参数和优化方法参数：{}'.format(model_path))
        optimizer.step()
        return last_epoch, error_rate

    # 获取最后一个保存的模型
    if resume_model is not None:
        last_epoch1, error_rate1 = load_model(resume_model)
    return model, optimizer, last_epoch1, error_rate1


# 保存模型
def save_checkpoint(model, optimizer, epoch_id, save_model_path,
                    error_rate, metrics_type, best_model=False):
    """保存模型

    :param model: 使用的模型
    :type model: paddle.nn.Layer
    :param optimizer: 使用的优化方法
    :type optimizer: paddle.optim.Optimizer
    :param save_model_path: 模型保存路径
    :type save_model_path: str
    :param epoch_id: 当前epoch
    :type epoch_id: int
    :param error_rate: 当前的错误率
    :type error_rate: float
    :param metrics_type: 当前使用的评估指标
    :type metrics_type: str
    :param best_model: 是否为最佳模型
    :type best_model: bool
    """
    model_path = os.path.join(save_model_path, f'epoch_{epoch_id}')
    os.makedirs(model_path, exist_ok=True)
    # 保存模型参数
    paddle.save(optimizer.state_dict(), os.path.join(model_path, 'optimizer.pdopt'))
    paddle.save(model.state_dict(), os.path.join(model_path, 'model.pdparams'))
    with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
        data = {"last_epoch": epoch_id, metrics_type: error_rate}
        f.write(json.dumps(data, indent=4, ensure_ascii=False))
    if not best_model:
        # 删除旧的模型
        old_model_path = os.path.join(save_model_path, f'epoch_{epoch_id - 3}')
        if os.path.exists(old_model_path):
            shutil.rmtree(old_model_path)
    logger.info('已保存模型：{}'.format(model_path))
