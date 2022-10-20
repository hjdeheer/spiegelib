import numpy as np
from spiegelib.synth import SynthDawDreamer
import os

def generate_overidden_params(override_operators=[]):
    algorithm_number = 1
    alg = (1.0 / 32.0) * float(algorithm_number - 1) + 0.001

    overridden_parameters = [
        (0, 1.0), # Filter Cutoff (Fully open)
        (1, 0.0), # Filter Resonance
        (2, 1.0), # Output Gain
        (3, 0.5), # Master Tuning (Center is 0)
        (4, alg), # Operator configuration
        (5, 0.0), # Feedback
        (6, 1.0), # Key Sync Oscillators
        (7, 0.0), # LFO Speed
        (8, 0.0), # LFO Delay
        (9, 0.0), # LFO Pitch Modulation Depth
        (10, 0.0),# LFO Amplitude Modulation Depth
        (11, 0.0),# LFO Key Sync
        (12, 0.0),# LFO Waveform
        (13, 0.5),# Middle C Tuning
    ]

    # Turn off all pitch modulation parameters
    overridden_parameters.extend([(i, 0.0) for i in range(14, 23)]) 

    for overridden_operator in override_operators:
        start_param = 22 * overridden_operator + 1
        overridden_parameters.extend([(i, 0.0) for i in range(start_param, start_param + 22)])

    return overridden_parameters

def main():
    config_path = ""
    vst_path = "../vsts/Dexed.dll"

    synth = SynthDawDreamer(vst_path, note_length_secs=1.0, render_length_secs=1.0)

    if not os.path.exists(config_path):
        synth.set_overridden_parameters(generate_overidden_params())
        synth.save_state(config_path)
    
    synth.load_state(config_path)



if __name__ == '__main__':
    main()