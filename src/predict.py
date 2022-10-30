import numpy as np
import os

from spiegelib.estimator import ParameterLoss
from matplotlib import pyplot as plt
import seaborn as sns
import spiegelib as spgl

def main():

    #Specify configuration
    bins = 16
    dataset = "uniform"

    #Specify all model data
    synth_path = "../vsts/Dexed.dll"
    model_path = f"../data/models/{dataset}/{dataset}_{bins}.h5"
    source_path = "../data/evaluation/audio"

    #Load Synth - standard = midi note 72 (60 C4)
    synth = spgl.synth.SynthDawDreamer(synth_path,
                                        note_length_secs=1.0,
                                        render_length_secs=1.0)

    #Choose correct state (fixed)
    synth.load_state("../vsts/NewExperiment.json")
    synth.load_parameterModel("../data/presets/allParamsUpdatedNew.npy")
    save_path  = f"../data/evaluation/predict_{dataset}_{bins}"

    predicted_patches = []

    #Load network and data
    network = spgl.estimator.TFEstimatorBase.load(model_path)
    extractor = spgl.features.STFT(output='magnitude', fft_size=512, hop_size=256, time_major=True)

    #LOAD CORRECT SCALER!!
    #extractor.load_scaler("../data/models/uniform_lfo2op48/STFT/data_scaler.pkl")
    extractor.add_modifier(lambda data : data.reshape(data.shape[0], data.shape[1], 1), type='output')

    #Instantiate matcher obj
    matcher = spgl.SoundMatch(synth, network, extractor)

    trueAudio = spgl.AudioBuffer.load_folder(source_path)
    for i in range(len(trueAudio)):
        audio, params = matcher.match(trueAudio[i], onehot=16)

        #Save predicted audio and params
        predicted_patches.append(params)
        audio.save(os.path.join(save_path, f'{i}.wav'))

    predicted_patches = np.array(predicted_patches)

    #Save predicted params
    np.save(f"{save_path}/params", predicted_patches)

if __name__ == "__main__":
    main()
