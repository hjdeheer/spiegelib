import spiegelib as spgl
import numpy as np
from spiegelib.features import MFCCS2S, FFT, Spectrogram, RMS, MFCC
from spiegelib.estimator.conv_s2s import ConvBackBone
#from spiegelib.estimator.sound2synth import Sound2Synth
from spiegelib.estimator.lstm_s2s import LSTMBackBone
import tensorflow as tf
import pandas as pd

synth = spgl.synth.SynthDawDreamer("../vsts/Dexed.dll",
                            note_length_secs=1.0,
                            render_length_secs=1.0)
#load 9 param config
#synth.load_state("../vsts/experiment.json")
synth.load_parameterModel("../data/presets/allParamsUpdatedNew.npy")


feature1 = spgl.features.STFT(fft_size=512, hop_size=256, output='magnitude', time_major=True)
features = [feature1]
generator = spgl.DatasetGenerator(synth, features, output_folder="../data/test2", save_audio=True, scale=[True])
# generator.generate(50000, file_prefix="train_", technique='uniform')
# generator.generate(10000, file_prefix="test_", technique='uniform')
generator.generate(50, file_prefix="test_", technique='uniform')
generator.generate(50, file_prefix="2_", technique='normal')
generator.generate(50, file_prefix="3_", technique='preset')
generator.generate(50, file_prefix="4_", technique='preset')
generator.patch_to_onehot(bins=16)
generator.patch_to_onehot(bins=32)
generator.patch_to_onehot(bins=64)
generator.save_scaler(0, "data_scaler.pkl")
