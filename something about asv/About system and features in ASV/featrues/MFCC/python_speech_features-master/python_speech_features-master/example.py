#!/usr/bin/env python

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

(rate, sig) = wav.read("english.wav")
# 读取文件，返回采样率rate和语音信号sig。
mfcc_feat = mfcc(sig, rate)
# 提取13维的MFCC特征
d_mfcc_feat = delta(mfcc_feat, 2)
fbank_feat = logfbank(sig, rate)

print(fbank_feat[1:3, :])
