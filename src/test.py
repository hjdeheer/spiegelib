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

# synth2 = spgl.synth.SynthVST("../vsts/Dexed.dll",
#                             note_length_secs=1.0,
#                             render_length_secs=1.0)

print(synth.parametersDesc)
parameterModel = np.load("../data/presets/allParams.npy", allow_pickle=True)


# Mel-frequency Cepstral Coefficients audio feature extractor.
feature1 = spgl.features.STFT(fft_size=512, hop_size=256, output='magnitude', time_major=True)
feature2 = spgl.features.MFCC(num_mfccs=20, frame_size=2048, hop_size=1024, time_major=True)
feature3 = spgl.features.MelSpectrogram()
#feature3 = FFT(output='power')


# feature4 = Spectrogram()
# feature5 = RMS()
# features = [feature1, feature2, feature3]
# generator = spgl.DatasetGenerator(synth, features, output_folder="../data/dataset_uniform", save_audio=False)
# generator.generate(30000, file_prefix="train_", technique='uniform')
# generator.generate(10000, file_prefix="test_", technique='uniform')
#


s = np.load("../data/dataset_uniform/STFT/test_features.npy")

print(s)
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



# # Mel-frequency Cepstral Coefficients audio feature extractor.
# features = spgl.features.MFCC(num_mfccs=13, frame_size=2048,
#                               hop_size=1024, time_major=True)
#
#
# # %%
# # Setup generator for MFCC output and generate 50000 training examples
# # and 10000 testing examples
# generator = spgl.DatasetGenerator(synth, features, output_folder="../data/data_FM_mfcc", save_audio=True)
#
#generator.generate(1, file_prefix="train_")

