import _thread
import argparse
import functools
import os
import time
import tkinter.messagebox
import wave
from tkinter import *
from tkinter.filedialog import *

import pyaudio

from utils.predict import Predictor
from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_gpu',          bool,   True,   "是否使用GPU预测")
add_arg('enable_mkldnn',    bool,   False,  "是否使用mkldnn加速")
add_arg('mean_std_path',    str,    'dataset/mean_istd.json', "均值和标准值得json文件路径，后缀 (.json)")
add_arg('vocab_dir',        str,    'dataset/vocab_model',    "数据字典模型文件夹")
add_arg('model_dir',        str,    'models/inference/',      "导出的预测模型文件夹路径")
add_arg('beam_search_conf', str,    'configs/decoder.yml',    "集束搜索解码相关参数")
add_arg('decoder',          str,    'ctc_greedy',    "结果解码方法，有集束搜索解码器(ctc_beam_search)、贪心解码器(ctc_greedy)", choices=['ctc_beam_search', 'ctc_greedy'])
args = parser.parse_args()
print_arguments(args)


class SpeechRecognitionApp:
    def __init__(self, window: Tk, args):
        self.window = window
        self.wav_path = None
        self.predicting = False
        self.playing = False
        self.recording = False
        self.stream = None
        self.to_itn = False
        # 最大录音时长
        self.max_record = 20
        # 录音保存的路径
        self.output_path = 'dataset/record'
        # 创建一个播放器
        self.p = pyaudio.PyAudio()
        # 指定窗口标题
        self.window.title("夜雨飘零语音识别")
        # 固定窗口大小
        self.window.geometry('870x500')
        self.window.resizable(False, False)
        # 识别语音按钮
        self.short_button = Button(self.window, text="选择语音识别", width=20, command=self.predict_audio_thread)
        self.short_button.place(x=10, y=10)
        # 录音按钮
        self.record_button = Button(self.window, text="录音识别", width=20, command=self.record_audio_thread)
        self.record_button.place(x=170, y=10)
        # 播放音频按钮
        self.play_button = Button(self.window, text="播放音频", width=20, command=self.play_audio_thread)
        self.play_button.place(x=330, y=10)
        # 输出结果文本框
        self.result_label = Label(self.window, text="输出日志：")
        self.result_label.place(x=10, y=70)
        self.result_text = Text(self.window, width=120, height=30)
        self.result_text.place(x=10, y=100)
        # 逆文本标准化控件
        self.an_frame = Frame(self.window)
        self.check_var = BooleanVar()
        self.to_an_check = Checkbutton(self.an_frame, text='逆文本标准化', variable=self.check_var, command=self.to_itn_state)
        self.to_an_check.grid(row=0)
        self.an_frame.grid(row=1)
        self.an_frame.place(x=700, y=10)

        # 获取识别器
        self.predictor = Predictor(model_dir=args.model_dir,
                                   vocab_dir=args.vocab_dir,
                                   decoder=args.decoder,
                                   beam_search_conf=args.beam_search_conf,
                                   use_gpu=args.use_gpu,
                                   enable_mkldnn=args.enable_mkldnn)

    # 是否逆文本标准化
    def to_itn_state(self):
        self.to_itn = self.check_var.get()

    # 预测语音线程
    def predict_audio_thread(self):
        if not self.predicting:
            self.wav_path = askopenfilename(filetypes=[("音频文件", "*.wav"), ("音频文件", "*.mp3")], initialdir='./dataset')
            if self.wav_path == '': return
            self.result_text.delete('1.0', 'end')
            self.result_text.insert(END, "已选择音频文件：%s\n" % self.wav_path)
            self.result_text.insert(END, "正在识别中...\n")
            _thread.start_new_thread(self.predict_audio, (self.wav_path, ))
        else:
            tkinter.messagebox.showwarning('警告', '正在预测，请等待上一轮预测结束！')

    # 预测语音
    def predict_audio(self, wav_path):
        self.predicting = True
        try:
            start = time.time()
            text = self.predictor.predict(audio_path=wav_path, to_itn=self.to_itn)
            self.result_text.insert(END, f"消耗时间：{round((time.time() - start) * 1000)}ms, 识别结果: {text}\n")
        except Exception as e:
            print(e)
        self.predicting = False

    # 录音识别线程
    def record_audio_thread(self):
        if not self.playing and not self.recording:
            self.result_text.delete('1.0', 'end')
            _thread.start_new_thread(self.record_audio, ())
        else:
            if self.playing:
                tkinter.messagebox.showwarning('警告', '正在录音，无法播放音频！')
            else:
                # 停止播放
                self.recording = False

    def record_audio(self):
        self.record_button.configure(text='停止录音')
        self.recording = True
        # 录音参数
        chunk = 1024
        format = pyaudio.paInt16
        channels = 1
        rate = 16000

        # 打开录音
        self.stream = self.p.open(format=format,
                                  channels=channels,
                                  rate=rate,
                                  input=True,
                                  frames_per_buffer=chunk)
        self.result_text.insert(END, "正在录音...\n")
        start = time.time()
        frames = []
        while True:
            if not self.recording:break
            data = self.stream.read(chunk)
            frames.append(data)
            if len(frames) % 15 == 0:
                self.result_text.insert(END, "已录音%.2f秒\n" % (time.time() - start))
            if (time.time() - start) > self.max_record:
                self.result_text.insert(END, "录音已超过最大限制时长，强制停止录音！")
                break

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        save_path = os.path.join(self.output_path, '%s.wav' % str(int(time.time())))
        wf = wave.open(save_path, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(self.p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        self.recording = False
        self.result_text.insert(END, "录音已结束，录音文件保存在：%s\n" % save_path)
        # 识别录音
        self.result_text.insert(END, "正在识别中...\n")
        self.wav_path = save_path
        self.predict_audio(self.wav_path)
        self.record_button.configure(text='录音识别')

    # 播放音频线程
    def play_audio_thread(self):
        if self.wav_path is None or self.wav_path == '':
            tkinter.messagebox.showwarning('警告', '音频路径为空！')
        else:
            if not self.playing and not self.recording:
                _thread.start_new_thread(self.play_audio, ())
            else:
                if self.recording:
                    tkinter.messagebox.showwarning('警告', '正在录音，无法播放音频！')
                else:
                    # 停止播放
                    self.playing = False

    # 播放音频
    def play_audio(self):
        self.play_button.configure(text='停止播放')
        self.playing = True
        CHUNK = 1024
        wf = wave.open(self.wav_path, 'rb')
        # 打开数据流
        self.stream = self.p.open(format=self.p.get_format_from_width(wf.getsampwidth()),
                                  channels=wf.getnchannels(),
                                  rate=wf.getframerate(),
                                  output=True)
        # 读取数据
        data = wf.readframes(CHUNK)
        # 播放
        while data != b'':
            if not self.playing:break
            self.stream.write(data)
            data = wf.readframes(CHUNK)
        # 停止数据流
        self.stream.stop_stream()
        self.stream.close()
        self.playing = False
        self.play_button.configure(text='播放音频')


tk = Tk()
myapp = SpeechRecognitionApp(tk, args)

if __name__ == '__main__':
    tk.mainloop()
