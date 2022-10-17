import spiegelib as spgl
import numpy as np
from spiegelib.features import MFCCS2S, FFT, Spectrogram, RMS, MFCC
from spiegelib.estimator.conv_s2s import ConvBackBone
#from spiegelib.estimator.sound2synth import Sound2Synth
from spiegelib.estimator.lstm_s2s import LSTMBackBone
import tensorflow as tf


def extendParamModel(parameterModel, synthDescription):
    #Add min-max for missing variables
    for i, (pModel, pSynth) in enumerate(zip(parameterModel, synthDescription)):
        if 0 <= i <= 3:
            pModel['min'] = 0
            pModel['max'] = 99
            pModel['name'] = pSynth['name']
            pModel['isDiscrete'] = False
        elif 'min' not in pModel:
            #These are switches
            pModel['min'] = 0
            pModel['max'] = 1
            pModel['name'] = pSynth['name']
            pModel['isDiscrete'] = True
    for pModel, pSynth in zip(parameterModel, synthDescription):
        pModel['name'] = pSynth['name']
        #If parameter ranges from 0-99 it is not discrete
        if pModel['max'] == 99:
            pModel['isDiscrete'] = False
        else:
            pModel['isDiscrete'] = True




import pandas as pd
synth = spgl.synth.SynthDawDreamer("../vsts/Dexed.dll",
                            note_length_secs=1.0,
                            render_length_secs=1.0)
#load 9 param config
synth.load_state("../vsts/op2_dexed.json")
parameterModel = np.load("../data/presets/allParams.npy", allow_pickle=True)
extendParamModel(parameterModel, synth.parametersDesc)




# Mel-frequency Cepstral Coefficients audio feature extractor.
feature1 = spgl.features.STFT(fft_size=512, hop_size=256, output='magnitude', time_major=True)
#feature2 = spgl.features.MFCC(num_mfccs=100, frame_size=2048, hop_size=512, time_major=True)
#feature3 = spgl.features.MelSpectrogram(n_mels=256, hop_size=512, time_major=True)
#feature4 = spgl.features.Spectrogram(hop_size=512, time_major=True)
# feature5 = spgl.features.CQTChromagram()

features = [feature1]
generator = spgl.DatasetGenerator(synth, features, output_folder="../data/uniform_9", save_audio=True, scale=[True])
#generator.generate(25, file_prefix="", technique='uniform')
generator.patch_to_onehot(parameterModel)



# generator = spgl.DatasetGenerator(synth, features, output_folder="D:\data_uniform", save_audio=True, scale=[True])
# generator.generate(30000, file_prefix="train_", technique='uniform')
# generator.generate(10000, file_prefix="test_", technique='uniform')

#generator.save_scaler(0, "data_scaler.pkl")
# generator.save_scaler(1, "data_scaler.pkl")
# generator.save_scaler(2, "data_scaler.pkl")
# generator.save_scaler(3, "data_scaler.pkl")
# generator.save_scaler(4, "data_scaler.pkl")

# s = np.load("../data/test/STFT/features.npy")
# s1 = np.load("../data/test/MFCC/features.npy")
# s2 = np.load("../data/test/MelSpectrogram/features.npy")
# s3 = np.load("../data/test/Spectrogram/features.npy")
# s4 = np.load("../data/test/CQTChromagram/features.npy")
#
# print("TEST")
# t = np.load("../data/dataset_uniform_scaled/MelSpectrogram/test_features.npy")
# print(s)

#
# print(s)
# r = np.load("../data/dataset_uniform/MFCC/train_features.npy")
# p = np.load("../data/dataset_uniform/MFCCS2S/train_features.npy")

# r = np.load("../data/dataset/RMS/train_features.npy")
# p = np.load("../data/dataset/patch/train_patches.npy")



















# s = np.reshape(s, (s.shape[0], s.shape[1], s.shape[2], 1))
# r = np.reshape(r, (r.shape[0], r.shape[1]))
#
# dataset_12 = tf.data.Dataset.from_tensor_slices((s, r))
# dataset_label = tf.data.Dataset.from_tensor_slices(p)
# dataset = tf.data.Dataset.zip((dataset_12, dataset_label)).batch(16)
#
# for x, y in dataset.take(1):
#   input1, input2 = x
#   print(input1.shape, input2.shape)
#
# datasetxx = tf.data.Dataset.from_tensor_slices(s)
# model = Sound2Synth(output_dim=155)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(dataset, epochs=3)
# model.summary()

#Reshape to 4 dimensions:

#(None, width, height, 1)

# s = np.reshape(s, (s.shape[0], s.shape[1], s.shape[2], 1))
# my_seq = tf.keras.Sequential([ConvBackBone(output_dim=512)])
# my_seq(s)
# my_seq.summary()









# generator.generate(1, file_prefix="test_")
# generator.save_scaler(0, 'data_scaler.pkl')






# synth.render_patch()
# audio = synth.get_audio()
# synth.write_to_wav(audio, "../audio/test2.wav")
# buffer = spgl.AudioBuffer("../audio/test2.wav")
# buffer.plot_spectrogram()
# plt.show()
#
#
# synth.randomize_patch()
#
#
# synth.render_patch()
# audio = synth.get_audio()
# synth.write_to_wav(audio, "../audio/test3.wav")
# buffer = spgl.AudioBuffer("../audio/test3.wav")
# buffer.plot_spectrogram()
# plt.show()
#
# synth.randomize_patch()
#
#
# synth.render_patch()
# audio = synth.get_audio()
# synth.write_to_wav(audio, "../audio/test4.wav")
# buffer = spgl.AudioBuffer("../audio/test4.wav")
# buffer.plot_spectrogram()
# plt.show()

