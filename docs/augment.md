# 数据增强

数据增强是用来提升深度学习性能的非常有效的技术。通过在原始音频中添加小的随机扰动（标签不变转换）获得新音频来增强的语音数据。开发者不必自己合成，因为数据增强已经嵌入到数据生成器中并且能够即时完成，在训练模型的每个epoch中随机合成音频。

目前提供五个可选的增强组件供选择，配置并插入处理过程。

- 噪声干扰（需要背景噪音的音频文件）
- 混响（需要混响音频文件）
- 随机采样率增强
- 速度扰动
- 移动扰动
- 音量扰动
- SpecAugment增强方式
- SpecSubAugment增强方式

在项目提供了数据增强的配置参数，路径在`configs/augmentation.yml`，示例配置文件如下所示：
```yaml
# 语速增强
speed:
  # 增强概率
  prob: 0.5

# 音量增强
volume:
  # 增强概率
  prob: 0.5
  # 最小增益
  min_gain_dBFS: -15
  # 最大增益
  max_gain_dBFS: 15

# 位移增强
shift:
  # 增强概率
  prob: 0.5
  # 最小偏移，单位为毫秒
  min_shift_ms: -5
  # 最大偏移，单位为毫秒
  max_shift_ms: 5

# 重采样增强
resample:
  # 增强概率
  prob: 0.0
  # 最小增益
  new_sample_rate: [ 8000, 16000, 24000 ]

# 噪声增强
noise:
  # 增强概率
  prob: 0.5
  # 噪声增强的噪声文件夹
  noise_dir: 'dataset/noise'
  # 针对噪声的最小音量增益
  min_snr_dB: 10
  # 针对噪声的最大音量增益
  max_snr_dB: 50

# 混响增强
reverb:
  # 增强概率
  prob: 0.2
  # 混响增强的混响文件夹
  reverb_dir: 'dataset/reverb'

# Spec增强
spec_aug:
  # 增强概率
  prob: 0.5
  # 频域掩蔽的比例
  freq_mask_ratio: 0.15
  # 频域掩蔽次数
  n_freq_masks: 2
  # 频域掩蔽的比例
  time_mask_ratio: 0.05
  # 频域掩蔽次数
  n_time_masks: 2
  # 最大时间扭曲
  max_time_warp: 5

spec_sub_aug:
  # 增强概率
  prob: 0.5
  # 时间替换的最大宽度
  max_time: 30
  # 时间替换的的次数
  num_time_sub: 3
```

当`train.py`的`--data_augment_configs`参数被设置为上述示例配置文件的路径时，每个epoch中的每个音频片段都将被处理。使用数据增强技术时要小心，由于扩大了训练和测试集的差异，不恰当的增强会对训练模型不利，导致训练和预测的差距增大。
