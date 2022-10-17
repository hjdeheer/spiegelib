import numpy as np
import tensorflow as tf
import os

from matplotlib import pyplot as plt

import spiegelib as spgl

def main():
    synth_path = "../vsts/Dexed.dll"
    model_path = "../data/models/conv6onehot_STFT_uniform9/model.h5"

    source_path = "../data/evaluation/audio"
    save_path  = "../data/evaluation/predict"

    synth = spgl.synth.SynthDawDreamer(synth_path,
                                        note_length_secs=1.0,
                                        render_length_secs=1.0)

    synth.load_state("../vsts/op2_dexed.json")
    network = spgl.estimator.TFEstimatorBase.load(model_path)

    extractor = spgl.features.STFT(output='magnitude', fft_size=512, hop_size=256, time_major=True)
    extractor.load_scaler("D:/data_uniform/STFT/data_scaler.pkl")
    extractor.add_modifier(lambda data : data.reshape(data.shape[0], data.shape[1], 1), type='output')

    matcher = spgl.SoundMatch(synth, network, extractor)   

    targets = spgl.AudioBuffer.load_folder(source_path)

    for i in range(len(targets)):
        audio = matcher.match(targets[i], onehot=True)
        audio.save(os.path.join(save_path, f'{i}.wav'))


    #Perform evaluation
    estimations  = [spgl.AudioBuffer.load_folder(save_path)]
    evaluation = spgl.evaluation.MFCCEval(targets, estimations)
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
