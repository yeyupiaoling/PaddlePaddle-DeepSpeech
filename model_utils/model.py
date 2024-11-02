import paddle
import paddle.nn.functional as F
from paddle import nn

from data_utils.normalizer import FeatureNormalizer
from model_utils.cmvn import GlobalCMVN


class Conv2dSubsampling4Pure(nn.Layer):
    def __init__(self, idim: int, odim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(1, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2D(odim, odim, 3, 2),
            nn.ReLU(), )
        self.subsampling_rate = 4
        self.output_dim = ((idim - 1) // 2 - 1) // 2 * odim

    def forward(self, x: paddle.Tensor, x_len: paddle.Tensor):
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        x = x.transpose([0, 2, 1, 3]).reshape([0, 0, -1])
        x_len = ((x_len - 1) // 2 - 1) // 2
        return x, x_len


class DeepSpeech2Model(nn.Layer):
    """The DeepSpeech2 network structure.

    :param input_dim: feature size for audio.
    :type input_dim: int
    :param vocab_size: Dictionary size for tokenized transcription.
    :type vocab_size: int
    :return: A tuple of an output unnormalized log probability layer (
             before softmax) and a ctc cost layer.
    :rtype: tuple of LayerOutput
    """

    def __init__(self,
                 input_dim,
                 vocab_size,
                 mean_istd_path: str,
                 num_rnn_layers=4,
                 rnn_layer_size=1024):
        super().__init__()
        self.input_dim = input_dim
        self.num_rnn_layers = num_rnn_layers
        feature_normalizer = FeatureNormalizer(mean_istd_filepath=mean_istd_path)
        self.global_cmvn = GlobalCMVN(paddle.to_tensor(feature_normalizer.mean, dtype=paddle.float32),
                                      paddle.to_tensor(feature_normalizer.istd, dtype=paddle.float32))
        self.conv = Conv2dSubsampling4Pure(input_dim, 32)
        i_size = self.conv.output_dim
        layernorm_size = 2 * rnn_layer_size
        self.rnn = nn.LayerList()
        self.layernorm_list = nn.LayerList()
        for i in range(0, num_rnn_layers):
            if i == 0:
                rnn_input_size = i_size
            else:
                rnn_input_size = layernorm_size
            self.rnn.append(nn.GRU(input_size=rnn_input_size,
                                   hidden_size=rnn_layer_size,
                                   num_layers=1,
                                   direction="bidirectional"))
            self.layernorm_list.append(nn.LayerNorm(layernorm_size))
        self.output_dim = layernorm_size
        self.output = paddle.nn.Linear(layernorm_size, vocab_size)

    def forward(self, speech, speech_lengths):
        """Compute Model loss

        Args:
            speech (Tensor): [B, T, D]
            speech_lengths (Tensor): [B]

        Returns:
            loss (Tensor): [1]
        """
        x = self.global_cmvn(speech)
        x, x_lens = self.conv(x, speech_lengths)
        for i in range(0, self.num_rnn_layers):
            x, final_state = self.rnn[i](x, sequence_length=x_lens)  # [B, T, D]
            x = self.layernorm_list[i](x)
        x = self.output(x)
        x = x.transpose([1, 0, 2])
        return x, x_lens

    def predict(self, speech, speech_lengths):
        x = self.global_cmvn(speech)
        x, x_lens = self.conv(x, speech_lengths)
        for i in range(0, self.num_rnn_layers):
            x, final_state = self.rnn[i](x, sequence_length=x_lens)  # [B, T, D]
            x = self.layernorm_list[i](x)
        x = self.output(x)
        ctc_probs = F.softmax(x, axis=2)
        return ctc_probs, x_lens

    def export(self):
        static_model = paddle.jit.to_static(
            self.predict,
            input_spec=[
                paddle.static.InputSpec(shape=[None, None, self.input_dim], dtype=paddle.float32),  # [B, T, D]
                paddle.static.InputSpec(shape=[None], dtype=paddle.int64),  # audio_length, [B]
            ])
        return static_model
