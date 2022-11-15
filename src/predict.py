import numpy as np
import os

import itertools

from tqdm import tqdm

from spiegelib.estimator import ParameterLoss
from matplotlib import pyplot as plt
import spiegelib as spgl

def main():

    #Run all configurations!
    allBins = [4, 8, 12, 16, 32]
    allDatasets = ["uniform", "normal", "preset"]

    for bins, dataset in tqdm(list(itertools.product(allBins, allDatasets))):
        #Specify all model data
        synth_path = "../vsts/Dexed.dll"
        model_path = f"../data/models/lfo2op_{dataset}/lfo2op_{dataset}_{bins}.h5"
        source_path = "../data/evaluation/audio"

        #Load Synth - standard = midi note 72 (60 C4)
        synth = spgl.synth.SynthDawDreamer(synth_path,
                                            note_length_secs=1.0,
                                            render_length_secs=1.0, midi_note=48)

        #Choose correct state (fixed)
        synth.load_state("../vsts/NewExperiment.json")
        synth.load_parameterModel("../data/presets/allParamsUpdatedNew.npy")
        save_path  = f"../data/evaluation/predict_{dataset}_{bins}"

        predicted_patches = []

        #Load network and data
        network = spgl.estimator.TFEstimatorBase.load(model_path)
        extractor = spgl.features.STFT(output='magnitude', fft_size=512, hop_size=256, time_major=True)

        #LOAD CORRECT SCALER!!
        extractor.load_scaler(f"../data/models/lfo2op_{dataset}/data_scaler.pkl")
        extractor.add_modifier(lambda data : data.reshape(data.shape[0], data.shape[1], 1), type='output')

        #Instantiate matcher obj
        matcher = spgl.SoundMatch(synth, network, extractor)

        trueAudio = spgl.AudioBuffer.load_folder(source_path)
        for i in range(len(trueAudio)):
            audio, params = matcher.match(trueAudio[i], onehot=bins)
            #Save predicted audio and params
            predicted_patches.append(params)
            audio.save(os.path.join(save_path, f'{i}.wav'))

        predicted_patches = np.array(predicted_patches)

        #Save predicted params
        np.save(f"{save_path}/params", predicted_patches)

if __name__ == "__main__":
    main()
