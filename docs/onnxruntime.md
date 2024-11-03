# 使用onnxruntime推理

导出模型这一步需要在Linux系统上完成，Windows系统上无法导出onnx模型。

1. 安装paddle2onnx库，用于将静态的PaddlePaddle模型导出为onnx模型。
```shell
pip install paddle2onnx -U
```

2. 执行导出命令。
```shell
paddle2onnx --model_dir models/inference/ \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file models/inference/model.onnx \
            --opset_version 16
```

3. 参考`onnx_infer.py`程序使用onnxruntime进行推理。