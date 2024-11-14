import paddle
from paddle import nn
import paddle.nn.functional as F

from data_utils.normalizer import FeatureNormalizer
from model_utils.cmvn import GlobalCMVN
from model_utils.conv import ConvStack
from model_utils.rnn import RNNStack

__all__ = ['DeepSpeech2Model']


class DeepSpeech2Model(nn.Layer):
    """DeepSpeech2模型结构

    :param input_dim: 输入的特征大小
    :type input_dim: int
    :param vocab_size: 字典的大小，用来分类输出
    :type vocab_size: int
    :param mean_istd_path: 均值和标准差文件路径
    :type mean_istd_path: str
    :param num_conv_layers: 堆叠卷积层数
    :type num_conv_layers: int
    :param num_rnn_layers: 堆叠RNN层数
    :type num_rnn_layers: int
    :param rnn_layer_size: RNN层大小
    :type rnn_layer_size: int

    :return: DeepSpeech2模型
    :rtype: nn.Layer
    """

    def __init__(self, input_dim, vocab_size, mean_istd_path, num_conv_layers=2, num_rnn_layers=3, rnn_layer_size=1024):
        super().__init__()
        feature_normalizer = FeatureNormalizer(mean_istd_filepath=mean_istd_path)
        self.global_cmvn = GlobalCMVN(paddle.to_tensor(feature_normalizer.mean, dtype=paddle.float32),
                                      paddle.to_tensor(feature_normalizer.istd, dtype=paddle.float32))
        # 卷积层堆
        self.conv = ConvStack(input_dim, num_conv_layers)
        # RNN层堆
        i_size = self.conv.output_height
        self.rnn = RNNStack(i_size=i_size, h_size=rnn_layer_size, num_stacks=num_rnn_layers)
        # 分类输入层
        self.bn = nn.BatchNorm1D(rnn_layer_size * 2, data_format='NLC')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(rnn_layer_size * 2, vocab_size)

    def forward(self, speech, speech_lengths):
        """
        Args:
            speech (Tensor): [B, Tmax, D]
            speech_lengths (Tensor): [B]
        Returns:
            x (Tensor): [B, T, D]
            x_lens (Tensor): [B]
        """
        x = self.global_cmvn(speech)
        # [B, T, D] -> [B, C=1, D, T]
        x = x.transpose([0, 2, 1])
        x = x.unsqueeze(1)

        x, x_lens = self.conv(x, speech_lengths)

        # 将数据从卷积特征映射转换为向量序列
        x = x.transpose([0, 3, 1, 2])  # [B, T, C, D]
        x = x.reshape([0, 0, -1])  # [B, T, C*D]
        # 删除填充部分
        x = self.rnn(x, x_lens)  # [B, T, D]

        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.transpose([1, 0, 2])
        return x, x_lens

    def predict(self, speech, speech_lengths):
        """
        Args:
            speech (Tensor): [B, Tmax, D]
            speech_lengths (Tensor): [B]
        Returns:
            ctc_probs (Tensor): [B, T, D]
            x_lens (Tensor): [B]
        """
        x = self.global_cmvn(speech)
        # [B, T, D] -> [B, C=1, D, T]
        x = x.transpose([0, 2, 1])
        x = x.unsqueeze(1)

        x, x_lens = self.conv(x, speech_lengths)

        # 将数据从卷积特征映射转换为向量序列
        x = x.transpose([0, 3, 1, 2])  # [B, T, C, D]
        x = x.reshape([0, 0, -1])  # [B, T, C*D]
        # 删除填充部分
        x = self.rnn(x, x_lens)  # [B, T, D]

        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        ctc_probs = F.softmax(x, axis=2)
        return ctc_probs, x_lens

    def export(self):
        static_model = paddle.jit.to_static(
            self.predict,
            input_spec=[
                paddle.static.InputSpec(shape=[None, None, self.input_dim], dtype=paddle.float32),  # speech [B, T, D]
                paddle.static.InputSpec(shape=[None], dtype=paddle.int64),  # speech_lengths [B]
            ])
        return static_model
