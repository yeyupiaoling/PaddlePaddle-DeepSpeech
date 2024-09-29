import argparse
import functools
import os
import time

from flask import request, Flask, render_template
from flask_cors import CORS
from loguru import logger

from utils.predict import Predictor
from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("host",             str,    "0.0.0.0",            "监听主机的IP地址")
add_arg("port",             int,    5000,                 "服务所使用的端口号")
add_arg("save_path",        str,    'dataset/upload/',    "上传音频文件的保存目录")
add_arg('use_gpu',          bool,   True,   "是否使用GPU预测")
add_arg('enable_mkldnn',    bool,   False,  "是否使用mkldnn加速")
add_arg('to_itn',           bool,   False,  "是否逆文本标准化")
add_arg('beam_size',        int,    300,    "集束搜索解码相关参数，搜索大小，范围:[5, 500]")
add_arg('alpha',            float,  1.2,    "集束搜索解码相关参数，LM系数")
add_arg('beta',             float,  0.35,   "集束搜索解码相关参数，WC系数")
add_arg('cutoff_prob',      float,  0.99,   "集束搜索解码相关参数，剪枝的概率")
add_arg('cutoff_top_n',     int,    40,     "集束搜索解码相关参数，剪枝的最大值")
add_arg('mean_std_path',    str,    './dataset/mean_istd.json',    "均值和标准值得json文件路径，后缀 (.json)")
add_arg('vocab_path',       str,    './dataset/vocabulary.txt',    "数据集的词汇表文件路径")
add_arg('model_dir',        str,    './models/infer/',             "导出的预测模型文件夹路径")
add_arg('lang_model_path',  str,    './lm/zh_giga.no_cna_cmn.prune01244.klm',    "集束搜索解码相关参数，语言模型文件路径")
add_arg('decoder',          str,    'ctc_greedy',    "结果解码方法，有集束搜索解码器(ctc_beam_search)、贪心解码器(ctc_greedy)", choices=['ctc_beam_search', 'ctc_greedy'])
args = parser.parse_args()

app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/static")
# 允许跨越访问
CORS(app)

predictor = Predictor(model_dir=args.model_dir,
                      vocab_path=args.vocab_path,
                      decoder=args.decoder,
                      alpha=args.alpha,
                      beta=args.beta,
                      lang_model_path=args.lang_model_path,
                      beam_size=args.beam_size,
                      cutoff_prob=args.cutoff_prob,
                      cutoff_top_n=args.cutoff_top_n,
                      use_gpu=args.use_gpu,
                      enable_mkldnn=args.enable_mkldnn)


# 语音识别接口
@app.route("/recognition", methods=['POST'])
def recognition():
    f = request.files['audio']
    if f:
        # 临时保存路径
        file_path = os.path.join(args.save_path, f.filename)
        f.save(file_path)
        try:
            start = time.time()
            # 执行识别
            text = predictor.predict(audio_path=file_path, to_itn=args.to_itn)
            end = time.time()
            logger.info(f"识别时间：{int((end - start) * 1000)}ms，识别结果：{text}")
            result = str({"code": 0, "msg": "success", "result": text}).replace("'", '"')
            return result
        except Exception as e:
            logger.exception("预测出错！")
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
