import argparse
import functools
import os
import time

import numpy as np
import paddle.fluid as fluid
from flask import request, Flask, render_template
from flask_cors import CORS
from data_utils.data import DataGenerator
from model_utils.model import DeepSpeech2Model
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("host",             str,    "localhost",          "server host. (default: %(default)s)")
add_arg("port",             int,    5000,                 "server port. (default: %(default)s)")
add_arg('save_path',        str,    './dataset/upload/',   "Server's IP port.")
add_arg('beam_size',        int,    500,     "Beam search width.")
add_arg('num_conv_layers',  int,    2,       "# of convolution layers.")
add_arg('num_rnn_layers',   int,    3,       "# of recurrent layers.")
add_arg('rnn_layer_size',   int,    2048,    "# of recurrent cells per layer.")
add_arg('alpha',            float,  1.2,     "Coef of LM for beam search.")
add_arg('beta',             float,  0.35,    "Coef of WC for beam search.")
add_arg('cutoff_prob',      float,  1.0,     "Cutoff probability for pruning.")
add_arg('cutoff_top_n',     int,    40,      "Cutoff number for pruning.")
add_arg('use_gru',          bool,   True,    "Use GRUs instead of simple RNNs.")
add_arg('use_gpu',          bool,   True,    "Use GPU or not.")
add_arg('share_rnn_weights',bool,   False,   "Share input-hidden weights across bi-directional RNNs. Not for GRU.")
add_arg('mean_std_path',    str,    './dataset/mean_std.npz',        "Filepath of normalizer's mean & std.")
add_arg('vocab_path',       str,    './dataset/zh_vocab.txt',        "Filepath of vocabulary.")
add_arg('model_path',       str,    './models/step_final/',          "The pre-trained model.")
add_arg('lang_model_path',  str,    './lm/zh_giga.no_cna_cmn.prune01244.klm',        "Filepath for language model.")
add_arg('decoding_method',  str,    'ctc_beam_search',        "Decoding method. Options: ctc_beam_search, ctc_greedy", choices=['ctc_beam_search', 'ctc_greedy'])
add_arg('specgram_type',    str,    'linear',                 "Audio feature type. Options: linear, mfcc.", choices=['linear', 'mfcc'])
args = parser.parse_args()

app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/static")
# 允许跨越访问
CORS(app)


if args.use_gpu:
    place = fluid.CUDAPlace(0)
else:
    place = fluid.CPUPlace()

data_generator = DataGenerator(vocab_filepath=args.vocab_path,
                               mean_std_filepath=args.mean_std_path,
                               augmentation_config='{}',
                               specgram_type=args.specgram_type,
                               keep_transcription_text=True,
                               place=place,
                               is_training=False)
# prepare ASR model
ds2_model = DeepSpeech2Model(vocab_size=data_generator.vocab_size,
                             num_conv_layers=args.num_conv_layers,
                             num_rnn_layers=args.num_rnn_layers,
                             rnn_layer_size=args.rnn_layer_size,
                             use_gru=args.use_gru,
                             init_from_pretrained_model=args.model_path,
                             place=place,
                             share_rnn_weights=args.share_rnn_weights,
                             is_infer=True)

if args.decoding_method == "ctc_beam_search":
    ds2_model.init_ext_scorer(args.alpha, args.beta, args.lang_model_path, data_generator.vocab_list)


def predict(filename):
    feature = data_generator.process_utterance(filename, "")
    probs_split = ds2_model.infer(feature=feature)

    if args.decoding_method == "ctc_greedy":
        result_transcript = ds2_model.decode_batch_greedy(probs_split=probs_split,
                                                          vocab_list=data_generator.vocab_list)
    else:
        result_transcript = ds2_model.decode_batch_beam_search(probs_split=probs_split,
                                                               beam_alpha=args.alpha,
                                                               beam_beta=args.beta,
                                                               beam_size=args.beam_size,
                                                               cutoff_prob=args.cutoff_prob,
                                                               cutoff_top_n=args.cutoff_top_n,
                                                               vocab_list=data_generator.vocab_list,
                                                               num_processes=16)

    return result_transcript[0]


# 语音识别接口
@app.route("/recognition", methods=['POST'])
def recognition():
    f = request.files['audio']
    if f:
        # 临时保存路径
        file_path = args.save_path + f.filename
        f.save(file_path)
        try:
            start = time.time()
            text = predict(filename=file_path)
            end = time.time()
            print("识别时间：%dms，识别结果：%s" % (round((end - start) * 1000), text))
            result = str({"code": 0, "msg": "success", "result": text}).replace("'", '"')
            return result
        except:
            return str({"error": 1, "msg": "audio read fail!"})
    return str({"error": 3, "msg": "audio is None!"})


@app.route('/')
def home():
    return render_template("index.html")


if __name__ == '__main__':
    print_arguments(args)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    app.run(host=args.host, port=args.port)