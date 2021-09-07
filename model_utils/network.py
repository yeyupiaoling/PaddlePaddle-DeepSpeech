import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.static.nn as nn


def conv_bn_layer(input, filter_size, num_channels_out, stride, padding, act, masks):
    """卷积层与批处理归一化

    :param input: Input layer.
    :type input: Variable
    :param filter_size: 卷积核大小
    :type filter_size: int|tuple|list
    :param num_channels_out: 输出通道数
    :type num_channels_out: int
    :param stride: 步幅大小
    :type stride: int|tuple|list
    :param padding: 填充大小
    :type padding: int|tuple|list
    :param act: 激活函数类型
    :type act: string
    :param masks:掩码层，用于填充
    :type masks: Variable
    :return: 批处理范数层后卷积层
    :rtype: Variable
    """
    conv_layer = nn.conv2d(input=input,
                           num_filters=num_channels_out,
                           filter_size=filter_size,
                           stride=stride,
                           padding=padding,
                           param_attr=paddle.ParamAttr(),
                           bias_attr=False)

    batch_norm = nn.batch_norm(input=conv_layer, act=act, param_attr=paddle.ParamAttr(), bias_attr=paddle.ParamAttr())

    # 将填充部分重置为0
    padding_reset = paddle.multiply(batch_norm, masks)
    return padding_reset


def bidirectional_gru_bn_layer(input, size, act):
    """双向gru层与顺序批处理归一化，批处理规范化只在输入状态权值上执行。

    :param input: Input layer.
    :type input: Variable
    :param h_size: GRU的cell的大小
    :type h_size: int
    :param act: 激活函数类型
    :type act: string
    :return: 双向GRU层
    :rtype: Variable
    """
    input_proj_forward = nn.fc(x=input, size=size * 3, weight_attr=paddle.ParamAttr())
    input_proj_reverse = nn.fc(x=input, size=size * 3, weight_attr=paddle.ParamAttr())
    # 批标准只在与输入相关的预测上执行
    input_proj_bn_forward = nn.batch_norm(input=input_proj_forward,
                                          act=None,
                                          param_attr=paddle.ParamAttr(),
                                          bias_attr=paddle.ParamAttr())
    input_proj_bn_reverse = nn.batch_norm(input=input_proj_reverse,
                                          act=None,
                                          param_attr=paddle.ParamAttr(),
                                          bias_attr=paddle.ParamAttr())
    # forward and backward in time
    forward_gru = fluid.layers.dynamic_gru(input=input_proj_bn_forward,
                                           size=size,
                                           gate_activation='sigmoid',
                                           candidate_activation=act,
                                           param_attr=paddle.ParamAttr(),
                                           bias_attr=paddle.ParamAttr(),
                                           is_reverse=False)
    reverse_gru = fluid.layers.dynamic_gru(input=input_proj_bn_reverse,
                                           size=size,
                                           gate_activation='sigmoid',
                                           candidate_activation=act,
                                           param_attr=paddle.ParamAttr(),
                                           bias_attr=paddle.ParamAttr(),
                                           is_reverse=True)
    return paddle.concat(x=[forward_gru, reverse_gru], axis=1)


def conv_group(input, num_stacks, seq_len_data, masks):
    """具有堆叠卷积层的卷积组

    :param input: Input layer.
    :type input: Variable
    :param num_stacks: 堆叠的卷积层数
    :type num_stacks: int
    :param seq_len_data: 有效序列长度数据层
    :type seq_len_data:Variable
    :param masks: 掩码数据层以重置填充
    :type masks: Variable
    :return: 卷积组的输出层
    :rtype: Variable
    """
    filter_size = (41, 11)
    stride = (2, 3)
    padding = (20, 5)
    conv = conv_bn_layer(input=input,
                         filter_size=filter_size,
                         num_channels_out=32,
                         stride=stride,
                         padding=padding,
                         act="brelu",
                         masks=masks)

    seq_len_data = (np.array(seq_len_data) - filter_size[1] + 2 * padding[1]) // stride[1] + 1

    output_height = (161 - 1) // 2 + 1

    for i in range(num_stacks - 1):
        # reshape masks
        output_height = (output_height - 1) // 2 + 1
        masks = paddle.slice(masks, axes=[2], starts=[0], ends=[output_height])
        conv = conv_bn_layer(input=conv,
                             filter_size=(21, 11),
                             num_channels_out=32,
                             stride=(2, 1),
                             padding=(10, 5),
                             act="brelu",
                             masks=masks)

    output_num_channels = 32
    return conv, output_num_channels, output_height, seq_len_data


def rnn_group(input, size, num_stacks):
    """RNN组具有堆叠的双向GRU层

    :param input: Input layer.
    :type input: Variable
    :param size:每层RNN的cell大小
    :type size: int
    :param num_stacks: 堆叠RNN层数
    :type num_stacks: int
    :return: RNN组的输出层
    :rtype: Variable
    """
    output = input
    for i in range(num_stacks):
        output = bidirectional_gru_bn_layer(input=output, size=size, act="relu")
    return output


def deep_speech_v2_network(audio_data,
                           text_data,
                           seq_len_data,
                           masks,
                           dict_size,
                           num_conv_layers=2,
                           num_rnn_layers=3,
                           rnn_size=256,
                           blank=0):
    """DeepSpeech2网络结构

    :param audio_data: 音频输入层
    :type audio_data: Variable
    :param text_data: 标签输入层
    :type text_data: Variable
    :param seq_len_data: 输出长度输入层
    :type seq_len_data: Variable
    :param masks: 掩码数据输入层
    :type masks: Variable
    :param dict_size: 字典大小
    :type dict_size: int
    :param num_conv_layers: 叠加卷积层数
    :type num_conv_layers: int
    :param num_rnn_layers: 叠加RNN层数
    :type num_rnn_layers: int
    :param rnn_size: RNN层隐层的大小
    :type rnn_size: int
    :return: 模型概率分布和ctc损失
    :rtype: tuple of LayerOutput
    """
    audio_data = paddle.unsqueeze(audio_data, axis=[1])

    # 卷积组
    conv_group_output, conv_group_num_channels, conv_group_height, seq_len_data = conv_group(input=audio_data,
                                                                                             num_stacks=num_conv_layers,
                                                                                             seq_len_data=seq_len_data,
                                                                                             masks=masks)

    # 转换数据形式卷积特征映射到向量序列
    transpose = paddle.transpose(conv_group_output, perm=[0, 3, 1, 2])
    reshape_conv_output = paddle.reshape(x=transpose, shape=[0, -1, conv_group_height * conv_group_num_channels])
    # 删除padding部分
    seq_len_data = paddle.reshape(seq_len_data, [-1])
    sequence = nn.sequence_unpad(x=reshape_conv_output, length=seq_len_data)
    # RNN组
    rnn_group_output = rnn_group(input=sequence, size=rnn_size, num_stacks=num_rnn_layers)
    fc = nn.fc(x=rnn_group_output, size=dict_size, weight_attr=paddle.ParamAttr(), bias_attr=paddle.ParamAttr())
    # 输出模型概率分布
    log_probs = paddle.nn.functional.softmax(fc)
    if not text_data:
        return log_probs, None
    else:
        # 计算CTCLoss
        ctc_loss = paddle.nn.functional.ctc_loss(log_probs=fc, labels=text_data, blank=blank, norm_by_times=True,
                                                 reduction='sum', input_lengths=None, label_lengths=None)
        ctc_loss = paddle.sum(ctc_loss)
        return log_probs, ctc_loss
