"""Contains DeepSpeech2 model."""

import logging
import os
import shutil
import time
import paddle
from datetime import datetime
from distutils.dir_util import mkpath
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler
from visualdl import LogWriter
from utils.error_rate import char_errors, word_errors
from decoders.ctc_greedy_decoder import greedy_decoder_batch
from model_utils.network import deep_speech_v2_network

logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')
paddle.enable_static()


class DeepSpeech2Model(object):
    """DeepSpeech2Model class.

    :param vocab_size: Decoding vocabulary size.
    :type vocab_size: int
    :param num_conv_layers: Number of stacking convolution layers.
    :type num_conv_layers: int
    :param num_rnn_layers: Number of stacking RNN layers.
    :type num_rnn_layers: int
    :param rnn_layer_size: RNN layer size (number of RNN cells).
    :type rnn_layer_size: int
    :param use_gru: Use gru if set True. Use simple rnn if set False.
    :type use_gru: bool
    :param share_rnn_weights: Whether to share input-hidden weights between
                              forward and backward directional RNNs.Notice that
                              for GRU, weight sharing is not supported.
    :type share_rnn_weights: bool
    :param place: Program running place.
    :type place: CPUPlace or CUDAPlace
    :param init_from_pretrained_model: Pretrained model path. If None, will train
                                  from stratch.
    :type init_from_pretrained_model: string|None
    :param output_model_dir: Output model directory. If None, output to current directory.
    :type output_model_dir: string|None
    """

    def __init__(self,
                 vocab_size,
                 num_conv_layers,
                 num_rnn_layers,
                 rnn_layer_size,
                 use_gru=False,
                 share_rnn_weights=True,
                 place=fluid.CPUPlace(),
                 init_from_pretrained_model=None,
                 output_model_dir=None,
                 is_infer=False,
                 error_rate_type='cer',
                 vocab_list=None):
        self._vocab_size = vocab_size
        self._num_conv_layers = num_conv_layers
        self._num_rnn_layers = num_rnn_layers
        self._rnn_layer_size = rnn_layer_size
        self._use_gru = use_gru
        self._share_rnn_weights = share_rnn_weights
        self._place = place
        self._init_from_pretrained_model = init_from_pretrained_model
        self._output_model_dir = output_model_dir
        self._ext_scorer = None
        self.logger = logging.getLogger("")
        self.logger.setLevel(level=logging.INFO)
        if not is_infer:
            shutil.rmtree('log', ignore_errors=True)
            self.writer = LogWriter(logdir='log')
        self.error_rate_type = error_rate_type
        self.vocab_list = vocab_list
        self.save_model_path = ''
        # 预测相关的参数
        self.infer_program = None
        self.infer_compiled_prog = None
        self.infer_feeder = None
        self.infer_log_probs = None
        self.infer_exe = None
        if is_infer:
            self.init_infer_program()

    def create_network(self, is_infer=False):
        """Create data layers and model network.
        :param is_training: Whether to create a network for training.
        :type is_training: bool
        :return reader: Reader for input.
        :rtype reader: read generater
        :return log_probs: An output unnormalized log probability layer.
        :rtype lig_probs: Varable
        :return loss: A ctc loss layer.
        :rtype loss: Variable
        """

        if not is_infer:
            input_fields = {
                'names': ['audio_data', 'text_data', 'seq_len_data', 'masks'],
                'shapes': [[None, 161, None], [None, 1], [None, 1], [None, 32, 81, None]],
                'dtypes': ['float32', 'int32', 'int64', 'float32'],
                'lod_levels': [0, 1, 0, 0]
            }

            inputs = [
                fluid.data(name=input_fields['names'][i],
                           shape=input_fields['shapes'][i],
                           dtype=input_fields['dtypes'][i],
                           lod_level=input_fields['lod_levels'][i])
                for i in range(len(input_fields['names']))
            ]

            reader = fluid.io.DataLoader.from_generator(feed_list=inputs,
                                                        capacity=128,
                                                        iterable=False,
                                                        use_double_buffer=True)

            (audio_data, text_data, seq_len_data, masks) = inputs
        else:
            audio_data = fluid.data(name='audio_data',
                                    shape=[None, 161, None],
                                    dtype='float32',
                                    lod_level=0)
            seq_len_data = fluid.data(name='seq_len_data',
                                      shape=[None, 1],
                                      dtype='int64',
                                      lod_level=0)
            masks = fluid.data(name='masks',
                               shape=[None, 32, 81, None],
                               dtype='float32',
                               lod_level=0)
            text_data = None
            reader = fluid.DataFeeder([audio_data, seq_len_data, masks], self._place)

        log_probs, loss = deep_speech_v2_network(audio_data=audio_data,
                                                 text_data=text_data,
                                                 seq_len_data=seq_len_data,
                                                 masks=masks,
                                                 dict_size=self._vocab_size,
                                                 num_conv_layers=self._num_conv_layers,
                                                 num_rnn_layers=self._num_rnn_layers,
                                                 rnn_size=self._rnn_layer_size,
                                                 use_gru=self._use_gru,
                                                 share_rnn_weights=self._share_rnn_weights)
        return reader, log_probs, loss

    def init_from_pretrained_model(self, exe, program):
        '''Init params from pretrain model. '''

        assert isinstance(self._init_from_pretrained_model, str)

        if not os.path.exists(self._init_from_pretrained_model):
            print(self._init_from_pretrained_model)
            raise Warning("The pretrained params do not exist.")
        fluid.io.load_params(executor=exe,
                             dirname=self._init_from_pretrained_model,
                             main_program=program,
                             filename="params.pdparams")

        print("成功加载了预训练模型：%s" % self._init_from_pretrained_model)

        pre_epoch = 0
        dir_name = self._init_from_pretrained_model.split('_')
        if len(dir_name) >= 2 and dir_name[-2].endswith('epoch') and dir_name[-1].isdigit():
            pre_epoch = int(dir_name[-1])

        return pre_epoch + 1

    def save_param(self, exe, program, dirname):
        '''Save model params to dirname'''

        assert isinstance(self._output_model_dir, str)

        param_dir = os.path.join(self._output_model_dir)

        if not os.path.exists(param_dir):
            os.mkdir(param_dir)

        self.save_model_path = os.path.join(param_dir, dirname)

        fluid.io.save_params(executor=exe,
                             dirname=os.path.join(param_dir, dirname),
                             main_program=program,
                             filename="params.pdparams")
        print("save parameters at %s" % self.save_model_path)

        return True

    def test(self, test_reader):
        '''Test the model.

        :param test_reader: Reader of test.
        :type test_reader: Reader
        :return: Wer/Cer rate.
        :rtype: float
        '''
        errors_sum, len_refs = 0.0, 0
        errors_func = char_errors if self.error_rate_type == 'cer' else word_errors
        # 初始化预测程序
        self.init_infer_program()
        for infer_data in test_reader():
            # 执行预测
            probs_split = self.infer_batch_probs(infer_data=infer_data)
            # 使用最优路径解码
            result_transcripts = greedy_decoder_batch(probs_split=probs_split,
                                                      vocabulary=self.vocab_list,
                                                      blank_index=len(self.vocab_list))
            target_transcripts = infer_data[1]
            # 计算字错率
            for target, result in zip(target_transcripts, result_transcripts):
                errors, len_ref = errors_func(target, result)
                errors_sum += errors
                len_refs += len_ref
        return errors_sum / len_refs

    def train(self,
              train_batch_reader,
              dev_batch_reader,
              learning_rate,
              gradient_clipping,
              num_epoch,
              batch_size,
              num_samples,
              test_off=False):
        """Train the model.

        :param train_batch_reader: Train data reader.
        :type train_batch_reader: callable
        :param dev_batch_reader: Validation data reader.
        :type dev_batch_reader: callable
        :param feeding_dict: Feeding is a map of field name and tuple index
                             of the data that reader returns.
        :type feeding_dict: dict|list
        :param learning_rate: Learning rate for ADAM optimizer.
        :type learning_rate: float
        :param gradient_clipping: Gradient clipping threshold.
        :type gradient_clipping: float
        :param num_epoch: Number of training epochs.
        :type num_epoch: int
        :param batch_size: Number of batch size.
        :type batch_size: int
        :param num_samples: The num of train samples.
        :type num_samples: int
        :param num_iterations_print: Number of training iterations for printing
                                     a training loss.
        :type num_iteratons_print: int
        :param only_train_batch:Every epoch only train only_train_batch batch. Avoid insufficient video memory
        :type only_train_batch:int
        :param test_off: Turn off testing.
        :type test_off: bool
        """
        # prepare model output directory
        if not os.path.exists(self._output_model_dir):
            mkpath(self._output_model_dir)

        if isinstance(self._place, fluid.CUDAPlace):
            dev_count = fluid.core.get_cuda_device_count()
            learning_rate = learning_rate * dev_count
        else:
            dev_count = int(os.environ.get('CPU_NUM', 1))

        # prepare the network
        train_program = fluid.Program()
        startup_prog = fluid.Program()
        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_reader, _, ctc_loss = self.create_network()
                # 学习率
                learning_rate = fluid.layers.exponential_decay(
                        learning_rate=learning_rate,
                        decay_steps=num_samples / batch_size / dev_count,
                        decay_rate=0.83,
                        staircase=True)
                # 准备优化器
                optimizer = fluid.optimizer.AdamOptimizer(
                    learning_rate=learning_rate,
                    regularization=fluid.regularizer.L2Decay(0.0001),
                    grad_clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=gradient_clipping))
                optimizer.minimize(loss=ctc_loss)

        exe = fluid.Executor(self._place)
        exe.run(startup_prog)

        # init from some pretrain models, to better solve the current task
        pre_epoch = 0
        if self._init_from_pretrained_model:
            pre_epoch = self.init_from_pretrained_model(exe, train_program)

        build_strategy = compiler.BuildStrategy()
        exec_strategy = fluid.ExecutionStrategy()

        # pass the build_strategy to with_data_parallel API
        train_compiled_prog = compiler.CompiledProgram(train_program).with_data_parallel(loss_name=ctc_loss.name,
                                                                                         build_strategy=build_strategy,
                                                                                         exec_strategy=exec_strategy)

        train_reader.set_batch_generator(train_batch_reader)

        train_step = 0
        test_step = 0
        num_batch = -1
        # run train
        for epoch_id in range(num_epoch):
            train_reader.start()
            epoch_loss = []
            time_begin = time.time()
            batch_id = 0
            while True:
                try:
                    fetch_list = [ctc_loss.name, learning_rate.name]
                    if batch_id % 100 == 0:
                        fetch = exe.run(program=train_compiled_prog,
                                        fetch_list=fetch_list,
                                        return_numpy=False)
                        each_loss = fetch[0]
                        each_learning_rate = np.array(fetch[1])[0]
                        epoch_loss.extend(np.array(each_loss[0]) / batch_size)

                        print("Train [%s] epoch: [%d/%d], batch: [%d/%d], learning rate: %f, train loss: %f\n" %
                              (datetime.now(), epoch_id, num_epoch, batch_id, num_batch, each_learning_rate,
                               np.mean(each_loss[0]) / batch_size))
                        # 记录训练损失值
                        self.writer.add_scalar('Train loss', np.mean(each_loss[0]) / batch_size, train_step)
                        self.writer.add_scalar('Learning rate', each_learning_rate, train_step)
                        train_step += 1
                    else:
                        _ = exe.run(program=train_compiled_prog,
                                    fetch_list=[],
                                    return_numpy=False)
                    # 每2000个batch保存一次模型
                    if batch_id % 2000 == 0 and batch_id != 0:
                        self.save_param(exe, train_program, "epoch_" + str(epoch_id + pre_epoch))
                    batch_id = batch_id + 1
                except fluid.core.EOFException:
                    train_reader.reset()
                    break
            num_batch = batch_id
            # 每一个epoch保存一次模型
            self.save_param(exe, train_program, "epoch_" + str(epoch_id + pre_epoch))
            used_time = time.time() - time_begin
            if test_off:
                print('======================last Train=====================')
                print("Train time: %f sec, epoch: %d, train loss: %f\n" %
                      (used_time, epoch_id, np.mean(np.array(epoch_loss))))
                print('======================last Train=====================')
            else:
                print('\n======================Begin test=====================')
                # 设置临时模型的路径
                self._init_from_pretrained_model = self.save_model_path
                # 执行测试
                test_result = self.test(test_reader=dev_batch_reader)
                print("Train time: %f sec, epoch: %d, train loss: %f, test %s: %f"
                      % (used_time, epoch_id + pre_epoch, np.mean(np.array(epoch_loss)), self.error_rate_type, test_result))
                print('======================Stop Test=====================\n')
                # 记录测试结果
                self.writer.add_scalar('Test %s' % self.error_rate_type, test_result, test_step)
                test_step += 1

        self.save_param(exe, train_program, "step_final")

        print("\n------------Training finished!!!-------------")

    # 预测一个batch的音频
    def infer_batch_probs(self, infer_data):
        """Infer the prob matrices for a batch of speech utterances.
        :param infer_data: List of utterances to infer, with each utterance
                           consisting of a tuple of audio features and
                           transcription text (empty string).
        :type infer_data: list
        :return: List of 2-D probability matrix, and each consists of prob
                 vectors for one speech utterancce.
        :rtype: List of matrix
        """
        # define inferer
        infer_results = []
        data = []
        if isinstance(self._place, fluid.CUDAPlace):
            num_places = fluid.core.get_cuda_device_count()
        else:
            num_places = int(os.environ.get('CPU_NUM', 1))
        # 开始预测
        for i in range(infer_data[0].shape[0]):
            # 使用多卡推理
            data.append([[infer_data[0][i], infer_data[2][i], infer_data[3][i]]])
            if len(data) == num_places:
                each_log_probs = self.infer_exe.run(program=self.infer_compiled_prog,
                                                    feed=list(self.infer_feeder.feed_parallel(
                                                        iterable=data, num_places=num_places)),
                                                    fetch_list=[self.infer_log_probs],
                                                    return_numpy=False)
                data = []
                infer_results.extend(np.array(each_log_probs[0]))
        # 如果数据是单数，就获取最后一个计算
        last_data_num = infer_data[0].shape[0] % num_places
        if last_data_num != 0:
            for i in range(infer_data[0].shape[0] - last_data_num, infer_data[0].shape[0]):
                each_log_probs = self.infer_exe.run(program=self.infer_program,
                                                    feed=self.infer_feeder.feed(
                                                        [[infer_data[0][i], infer_data[2][i], infer_data[3][i]]]),
                                                    fetch_list=[self.infer_log_probs],
                                                    return_numpy=False)
                infer_results.extend(np.array(each_log_probs[0]))

        # slice result
        infer_results = np.array(infer_results)
        seq_len = (infer_data[2] - 1) // 3 + 1

        start_pos = [0] * (infer_data[0].shape[0] + 1)
        for i in range(infer_data[0].shape[0]):
            start_pos[i + 1] = start_pos[i] + seq_len[i][0]
        probs_split = [
            infer_results[start_pos[i]:start_pos[i + 1]]
            for i in range(0, infer_data[0].shape[0])
        ]

        return probs_split

    # 初始化预测程序，加预训练模型
    def init_infer_program(self):
        # define inferer
        self.infer_program = fluid.Program()
        startup_prog = fluid.Program()

        # prepare the network
        with fluid.program_guard(self.infer_program, startup_prog):
            with fluid.unique_name.guard():
                self.infer_feeder, self.infer_log_probs, _ = self.create_network(is_infer=True)

        self.infer_program = self.infer_program.clone(for_test=True)
        self.infer_exe = fluid.Executor(self._place)
        self.infer_exe.run(startup_prog)

        # init param from pretrained_model
        if not self._init_from_pretrained_model:
            exit("预训练模型文件不存在！")
        self.init_from_pretrained_model(self.infer_exe, self.infer_program)

        # 支持多卡推理
        build_strategy = compiler.BuildStrategy()
        exec_strategy = fluid.ExecutionStrategy()
        self.infer_compiled_prog = compiler.CompiledProgram(self.infer_program).with_data_parallel(
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

    # 单个音频预测
    def infer(self, feature):
        """Infer the prob matrices for a batch of speech utterances.
        :param feature: DataGenerator.process_utterance get data[0]
        :return: List of 2-D probability matrix, and each consists of prob
                 vectors for one speech utterancce.
        :rtype: List of matrix
        """
        audio_len = feature.shape[1]
        mask_shape0 = (feature.shape[0] - 1) // 2 + 1
        mask_shape1 = (feature.shape[1] - 1) // 3 + 1
        mask_max_len = (audio_len - 1) // 3 + 1
        mask_ones = np.ones((mask_shape0, mask_shape1))
        mask_zeros = np.zeros((mask_shape0, mask_max_len - mask_shape1))
        mask = np.repeat(np.reshape(np.concatenate((mask_ones, mask_zeros), axis=1),
                                    (1, mask_shape0, mask_max_len)), 32, axis=0)
        infer_data = [np.array(feature).astype('float32'),
                      None,
                      np.array(audio_len).astype('int64'),
                      np.array(mask).astype('float32')]
        # run inference
        each_log_probs = self.infer_exe.run(program=self.infer_program,
                                            feed=self.infer_feeder.feed(
                                                [[infer_data[0], infer_data[2], infer_data[3]]]),
                                            fetch_list=[self.infer_log_probs],
                                            return_numpy=False)
        infer_result = np.array(each_log_probs[0])

        # slice result
        seq_len = (infer_data[2] - 1) // 3 + 1
        start_pos = [0, 0]
        start_pos[1] = start_pos[0] + seq_len
        probs_split = infer_result[start_pos[0]:start_pos[1]]
        return probs_split

    # 导出预测模型
    def export_model(self, data_feature, model_path):
        _ = self.infer(data_feature)
        audio_data = fluid.data(name='audio_data',
                                shape=[None, 161, None],
                                dtype='float32',
                                lod_level=0)
        seq_len_data = fluid.data(name='seq_len_data',
                                  shape=[None, 1],
                                  dtype='int64',
                                  lod_level=0)
        masks = fluid.data(name='masks',
                           shape=[None, 32, 81, None],
                           dtype='float32',
                           lod_level=0)
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        fluid.io.save_inference_model(dirname=model_path,
                                      feeded_var_names=[audio_data.name, seq_len_data.name, masks.name],
                                      target_vars=[self.infer_log_probs],
                                      executor=self.infer_exe,
                                      main_program=self.infer_program,
                                      model_filename='model.pdmodel',
                                      params_filename='model.pdiparams')
