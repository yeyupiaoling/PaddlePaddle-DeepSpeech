import logging
import os
import shutil
import time
import paddle
from datetime import datetime, timedelta
from distutils.dir_util import mkpath
import numpy as np
import paddle.fluid as fluid
from visualdl import LogWriter
from utils.error_rate import char_errors, word_errors
from decoders.ctc_greedy_decoder import greedy_decoder_batch
from model_utils.network import deep_speech_v2_network

logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')
paddle.enable_static()


class DeepSpeech2Model(object):
    """DeepSpeech2Model class.

    :param vocab_size: 词汇表大小
    :type vocab_size: int
    :param num_conv_layers: 叠加卷积层数
    :type num_conv_layers: int
    :param num_rnn_layers: 叠加RNN层数
    :type num_rnn_layers: int
    :param rnn_layer_size: RNN层大小
    :type rnn_layer_size: int
    :param place: Program running place.
    :type place: CPUPlace or CUDAPlace
    :param resume_model: 恢复模型路径
    :type resume_model: string|None
    :param pretrained_model: 预训练模型路径
    :type pretrained_model: string|None
    :param output_model_dir: 保存模型的路径
    :type output_model_dir: string|None
    :param error_rate_type: 测试计算错误率的方式
    :type error_rate_type: string|None
    :param vocab_list: 词汇表列表
    :type vocab_list: list|None
    :param blank: 损失函数的空白索引
    :type blank: int
    """

    def __init__(self,
                 vocab_size,
                 num_conv_layers,
                 num_rnn_layers,
                 rnn_layer_size,
                 place=paddle.CPUPlace(),
                 resume_model=None,
                 pretrained_model=None,
                 output_model_dir=None,
                 error_rate_type='cer',
                 vocab_list=None,
                 blank=0):
        self._vocab_size = vocab_size
        self._num_conv_layers = num_conv_layers
        self._num_rnn_layers = num_rnn_layers
        self._rnn_layer_size = rnn_layer_size
        self._place = place
        self._blank = blank
        self._pretrained_model = pretrained_model
        self._resume_model = resume_model
        self._output_model_dir = output_model_dir
        self._ext_scorer = None
        self.logger = logging.getLogger("")
        self.logger.setLevel(level=logging.INFO)
        self.error_rate_type = error_rate_type
        self.vocab_list = vocab_list
        self.save_model_path = ''
        # 预测相关的参数
        self.infer_program = None
        self.infer_compiled_prog = None
        self.infer_feeder = None
        self.infer_log_probs = None
        self.infer_exe = None

    def create_network(self, is_infer=False):
        """Create data layers and model network.
        :param is_infer: Whether to create a network for Inference.
        :type is_infer: bool
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
                paddle.static.data(name=input_fields['names'][i],
                                   shape=input_fields['shapes'][i],
                                   dtype=input_fields['dtypes'][i],
                                   lod_level=input_fields['lod_levels'][i])
                for i in range(len(input_fields['names']))
            ]

            reader = fluid.io.DataLoader.from_generator(feed_list=inputs,
                                                        capacity=64,
                                                        iterable=False,
                                                        use_double_buffer=True)

            (audio_data, text_data, seq_len_data, masks) = inputs
        else:
            audio_data = paddle.static.data(name='audio_data',
                                            shape=[None, 161, None],
                                            dtype='float32',
                                            lod_level=0)
            seq_len_data = paddle.static.data(name='seq_len_data',
                                              shape=[None, 1],
                                              dtype='int64',
                                              lod_level=0)
            masks = paddle.static.data(name='masks',
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
                                                 blank=self._blank)
        return reader, log_probs, loss

    # 加载模型
    def load_param(self, program, model_path, ignore_opt=False):
        if not os.path.exists(model_path):
            raise Warning("The pretrained params [%s] do not exist." % model_path)

        load_state_dict = paddle.load(model_path)
        if ignore_opt:
            for key in program.state_dict(mode='opt').keys():
                load_state_dict.pop(key)
        program.set_state_dict(load_state_dict)
        print('[{}] 成功加载模型：{}'.format(datetime.now(), model_path))

    # 保存模型
    def save_param(self, program, epoch):
        if not os.path.exists(self._output_model_dir):
            os.mkdir(self._output_model_dir)
        model_path = '{}/{}.pdparams'.format(self._output_model_dir, epoch)
        paddle.save(program.state_dict(), model_path)
        old_model_path = '{}/{}.pdparams'.format(self._output_model_dir, epoch - 3)
        if os.path.exists(old_model_path):
            os.remove(old_model_path)
        print("模型已保存在：%s" % model_path)
        return model_path

    def train(self,
              train_batch_reader,
              dev_batch_reader,
              learning_rate,
              gradient_clipping,
              num_epoch,
              batch_size,
              train_num_samples,
              test_num_samples,
              test_off=False):
        """Train the model.

        :param train_batch_reader: Train data reader.
        :type train_batch_reader: callable
        :param dev_batch_reader: Validation data reader.
        :type dev_batch_reader: callable
        :param learning_rate: Learning rate for ADAM optimizer.
        :type learning_rate: float
        :param gradient_clipping: Gradient clipping threshold.
        :type gradient_clipping: float
        :param num_epoch: Number of training epochs.
        :type num_epoch: int
        :param batch_size: Number of batch size.
        :type batch_size: int
        :param train_num_samples: The num of train samples.
        :type train_num_samples: int
        :param test_num_samples: The num of test samples.
        :type test_num_samples: int
        :param test_off: Turn off testing.
        :type test_off: bool
        """
        shutil.rmtree('log', ignore_errors=True)
        writer = LogWriter(logdir='log')
        # prepare model output directory
        if not os.path.exists(self._output_model_dir):
            mkpath(self._output_model_dir)

        if isinstance(self._place, paddle.CUDAPlace):
            dev_count = len(paddle.static.cuda_places())
        else:
            dev_count = int(os.environ.get('CPU_NUM', 1))

        pre_epoch = 0
        if self._resume_model:
            try:
                pre_epoch = os.path.basename(self._resume_model).split('.')[0]
                pre_epoch = int(pre_epoch)
            except:
                print("恢复模型命名不正确，epoch从0开始训练！")

        # prepare the network
        train_program = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(train_program, startup_prog):
            with paddle.utils.unique_name.guard():
                train_reader, _, ctc_loss = self.create_network()
                # 学习率
                scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate=learning_rate, gamma=0.83,
                                                                 last_epoch=pre_epoch - 1)
                # 准备优化器
                optimizer = paddle.optimizer.Adam(
                    learning_rate=scheduler,
                    weight_decay=paddle.regularizer.L2Decay(5e-4),
                    grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=gradient_clipping))
                optimizer.minimize(loss=ctc_loss)

        exe = paddle.static.Executor(self._place)
        exe.run(startup_prog)

        # 加载预训练模型
        if self._resume_model is not None or self._pretrained_model is not None:
            if self._resume_model is not None:
                self.load_param(train_program, self._resume_model)
            else:
                self.load_param(train_program, self._pretrained_model, ignore_opt=True)

        build_strategy = paddle.static.BuildStrategy()
        exec_strategy = paddle.static.ExecutionStrategy()

        # pass the build_strategy to with_data_parallel API
        train_compiled_prog = paddle.static.CompiledProgram(train_program) \
            .with_data_parallel(loss_name=ctc_loss.name,
                                build_strategy=build_strategy,
                                exec_strategy=exec_strategy)

        train_reader.set_batch_generator(train_batch_reader)

        train_step = 0
        test_step = 0
        train_num_batch = train_num_samples // batch_size // dev_count
        test_num_batch = test_num_samples // batch_size
        sum_batch = train_num_batch * (num_epoch - pre_epoch)
        # run train
        for epoch_id in range(pre_epoch, num_epoch):
            epoch_id += 1
            train_reader.start()
            epoch_loss = []
            time_begin = time.time()
            batch_id = 0
            start = time.time()
            while True:
                try:
                    if batch_id % 100 == 0:
                        # 执行训练
                        fetch = exe.run(program=train_compiled_prog, fetch_list=[ctc_loss.name], return_numpy=False)
                        each_loss = fetch[0]
                        epoch_loss.extend(np.array(each_loss[0]) / batch_size)
                        eta_sec = ((time.time() - start) * 1000) * (
                                sum_batch - (epoch_id - pre_epoch - 1) * train_num_batch - batch_id)
                        eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                        print(
                            "Train [%s] epoch: [%d/%d], batch: [%d/%d], learning rate: %.8f, train loss: %f, eta: %s" %
                            (datetime.now(), epoch_id, num_epoch, batch_id, train_num_batch, scheduler.get_lr(),
                             np.mean(each_loss[0]) / batch_size, eta_str))
                        # 记录训练损失值
                        writer.add_scalar('Train loss', np.mean(each_loss[0]) / batch_size, train_step)
                        writer.add_scalar('Learning rate', scheduler.get_lr(), train_step)
                        train_step += 1
                    else:
                        # 执行训练
                        _ = exe.run(program=train_compiled_prog, fetch_list=[], return_numpy=False)
                    # 每10000个batch保存一次模型
                    if batch_id % 10000 == 0 and batch_id != 0:
                        self.save_param(train_program, epoch_id)
                    batch_id = batch_id + 1
                    start = time.time()
                except fluid.core.EOFException:
                    train_reader.reset()
                    break
            scheduler.step()
            # 每一个epoch保存一次模型
            self._resume_model = self.save_param(train_program, epoch_id)
            used_time = time.time() - time_begin
            if test_off:
                print('======================last Train=====================')
                print("Train time: %f sec, epoch: %d, train loss: %f\n" %
                      (used_time, epoch_id, float(np.mean(np.array(epoch_loss)))))
                print('======================last Train=====================')
            else:
                print('\n======================Begin test=====================')
                # 执行测试
                test_result = self.test(test_reader=dev_batch_reader, epoch_id=epoch_id, test_num_batch=test_num_batch)
                print("Test [%s] train time: %s, epoch: %d, train loss: %f, test %s: %f"
                      % (datetime.now(), str(timedelta(seconds=int(used_time))), epoch_id,
                         float(np.mean(np.array(epoch_loss))), self.error_rate_type, test_result))
                print('======================Stop Test=====================\n')
                # 记录测试结果
                writer.add_scalar('Test %s' % self.error_rate_type, test_result, test_step)
                test_step += 1

        self.save_param(train_program, num_epoch)
        print("\n------------Training finished!!!-------------")

    def test(self, test_reader, epoch_id, test_num_batch):
        '''Test the model.

        :param test_reader: Reader of test.
        :type test_reader: Reader
        :param epoch_id: Train epoch id
        :type epoch_id: int
        :param test_num_batch: Test batch number
        :type test_num_batch: int
        :return: Wer/Cer rate.
        :rtype: float
        '''
        # 初始化预测程序
        self.create_infer_program()
        # 加载预训练模型
        self.load_param(self.infer_program, self._resume_model)
        errors_sum, len_refs = 0.0, 0
        errors_func = char_errors if self.error_rate_type == 'cer' else word_errors
        for batch_id, infer_data in enumerate(test_reader()):
            # 执行预测
            probs_split = self.infer_batch_data(infer_data=infer_data)
            # 使用最优路径解码
            result_transcripts = greedy_decoder_batch(probs_split=probs_split, vocabulary=self.vocab_list)
            target_transcripts = infer_data[1]
            # 计算字错率
            for target, result in zip(target_transcripts, result_transcripts):
                errors, len_ref = errors_func(target, result)
                errors_sum += errors
                len_refs += len_ref
            if batch_id % 100 == 0:
                print("Test [%s] epoch: %d, batch: [%d/%d], %s: %f" %
                      (datetime.now(), epoch_id, batch_id, test_num_batch, self.error_rate_type, errors_sum / len_refs))
        return errors_sum / len_refs

    # 预测一个batch的音频
    def infer_batch_data(self, infer_data):
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
        if self.infer_exe is None:
            # 初始化预测程序
            self.create_infer_program()
            # 加载预训练模型
            self.load_param(self.infer_program, self._resume_model)
        infer_results = []
        data = []
        if isinstance(self._place, paddle.CUDAPlace):
            num_places = len(paddle.static.cuda_places())
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
    def create_infer_program(self):
        # define inferer
        self.infer_program = paddle.static.Program()
        startup_prog = paddle.static.Program()

        # prepare the network
        with paddle.static.program_guard(self.infer_program, startup_prog):
            with paddle.utils.unique_name.guard():
                self.infer_feeder, self.infer_log_probs, _ = self.create_network(is_infer=True)

        self.infer_program = self.infer_program.clone(for_test=True)
        self.infer_exe = paddle.static.Executor(self._place)
        self.infer_exe.run(startup_prog)

        # 支持多卡推理
        build_strategy = paddle.static.BuildStrategy()
        exec_strategy = paddle.static.ExecutionStrategy()
        self.infer_compiled_prog = paddle.static.CompiledProgram(self.infer_program) \
            .with_data_parallel(build_strategy=build_strategy,
                                exec_strategy=exec_strategy)

    # 导出预测模型
    def export_model(self, model_path):
        self.create_infer_program()
        # 加载预训练模型
        self.load_param(self.infer_program, self._resume_model)
        audio_data = paddle.static.data(name='audio_data',
                                        shape=[None, 161, None],
                                        dtype='float32',
                                        lod_level=0)
        seq_len_data = paddle.static.data(name='seq_len_data',
                                          shape=[None, 1],
                                          dtype='int64',
                                          lod_level=0)
        masks = paddle.static.data(name='masks',
                                   shape=[None, 32, 81, None],
                                   dtype='float32',
                                   lod_level=0)
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        paddle.static.save_inference_model(path_prefix=model_path + '/inference',
                                           feed_vars=[audio_data, seq_len_data, masks],
                                           fetch_vars=[self.infer_log_probs],
                                           executor=self.infer_exe,
                                           program=self.infer_program)
