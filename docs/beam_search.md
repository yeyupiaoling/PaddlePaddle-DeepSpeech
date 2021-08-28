

 - 如果需要使用`ctc_beam_search`集束搜索，需要编译`ctc_decoders`库，该编译只支持Ubuntu，其他Linux版本没测试过。
```shell
cd decoders
sh setup.sh
```


# 语言模型
如果是Windows环境，请忽略。语言模型是集束搜索解码方法使用的，集束搜索解码方法只能在Ubuntu下编译，Windows用户只能使用贪心策略解码方法。下载语言模型并放在lm目录下，下面下载的小语言模型，如何有足够大性能的机器，可以下载70G的超大语言模型，点击下载[Mandarin LM Large](https://deepspeech.bj.bcebos.com/zh_lm/zhidao_giga.klm) ，这个模型会大超多。
```shell script
cd DeepSpeech/
mkdir lm
cd lm
wget https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm
```
