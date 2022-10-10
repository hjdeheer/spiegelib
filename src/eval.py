import numpy as np
import tensorflow as tf
import os

import spiegelib as spgl

def main():
    synth_path = "/home/spiegelib/vsts/dexed-0.9.6-lnx/Dexed.vst3"
    model_path = "/home/spiegelib/data/models/conv8_STFT_uniform/cnn_vgg11_STFT_uniform.h5"

    source_path = "/home/spiegelib/data/evaluation/audio"
    save_path  = "/home/spiegelib/data/eval_test"

    synth = spgl.synth.SynthDawDreamer(synth_path,
                                        note_length_secs=1.0,
                                        render_length_secs=1.0)

    network = spgl.estimator.TFEstimatorBase.load(model_path)

    extractor = spgl.features.STFT(output='magnitude', fft_size=512, hop_size=256, time_major=True)
    extractor.add_modifier(lambda data : data.reshape(data.shape[0], data.shape[1], 1), type='output')

    matcher = spgl.SoundMatch(synth, network, extractor)   

    targets = spgl.AudioBuffer.load_folder(source_path)

    for i in range(len(targets)):
        audio = matcher.match(targets[i])
        audio.save(os.path.join(save_path, f'{i}.wav'))

if __name__ == "__main__":
    main()
