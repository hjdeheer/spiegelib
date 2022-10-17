import numpy as np
import tensorflow as tf
import os

from spiegelib.estimator import ParameterLoss
from matplotlib import pyplot as plt

import spiegelib as spgl

def main():
    synth_path = "../vsts/Dexed.dll"
    model_path = "../data/models/conv6_STFT_uniform_9_50K/model.h5"

    source_path = "../data/evaluation/audio"
    save_path  = "../data/evaluation/predict"


    #load Synth
    synth = spgl.synth.SynthDawDreamer(synth_path,
                                        note_length_secs=1.0,
                                        render_length_secs=1.0)

    synth.load_state("../vsts/op2_dexed.json")
    synth.load_parameterModel("../data/presets/allParamsUpdated.npy")


    #Load network and data
    network = spgl.estimator.TFEstimatorBase.load(model_path)
    extractor = spgl.features.STFT(output='magnitude', fft_size=512, hop_size=256, time_major=True)
    extractor.load_scaler("D:/data_uniform50k/STFT/data_scaler.pkl")
    extractor.add_modifier(lambda data : data.reshape(data.shape[0], data.shape[1], 1), type='output')

    #Instantiate matcher obj
    matcher = spgl.SoundMatch(synth, network, extractor)


    trueParams = np.load("../data/evaluation/patch/patches.npy")
    predictedParams = []

    trueAudio = spgl.AudioBuffer.load_folder(source_path)
    for i in range(len(trueAudio)):
        audio, params = matcher.match(trueAudio[i], onehot=16)

        #Save predicted audio and params
        predictedParams.append(params)
        audio.save(os.path.join(save_path, f'{i}.wav'))

    predictedParams = np.array(predictedParams)
    #Perform evaluation

    predictedAudio = [spgl.AudioBuffer.load_folder(save_path)]
    evaluation = spgl.evaluation.MFCCEval(trueAudio, predictedAudio)
    evaluation.evaluate()
    print(evaluation.get_scores())
    print(evaluation.get_stats())
    bins = np.arange(0, 60, 2.5)
    evaluation.plot_hist([0], 'mean_abs_error', bins)


    plt.title("MFCC distance of Conv6-uniform-9 parameters")
    plt.xlabel("MFCC distance")
    plt.ylabel("Num samples")
    plt.show()



if __name__ == "__main__":
    main()
