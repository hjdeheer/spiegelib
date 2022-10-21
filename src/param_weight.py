import numpy as np
from spiegelib.synth import SynthDawDreamer
import os

def generate_default_overidden_params( ):
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

    # Turn Operator 1 into a simple sine wave with no envelope
    overridden_parameters.extend([
        (23, 0.9), # Operator 1 Attack Rate
        (24, 0.9), # Operator 1 Decay Rate
        (25, 0.9), # Operator 1 Sustain Rate
        (26, 0.9), # Operator 1 Release Rate
        (27, 1.0), # Operator 1 Attack Level
        (28, 1.0), # Operator 1 Decay Level
        (29, 1.0), # Operator 1 Sustain Level
        (30, 0.0), # Operator 1 Release Level
        (31, 1.0), # Operator 1 Gain
        (32, 0.0), # Operator 1 Mode (1.0 is Fixed Frequency)
        (33, 0.5), # Operator 1 Coarse Tuning
        (34, 0.0), # Operator 1 Fine Tuning
        (35, 0.5), # Operator 1 Detune
        (36, 0.0), # Operator 1 Env Scaling Param
        (37, 0.0), # Operator 1 Env Scaling Param
        (38, 0.0), # Operator 1 Env Scaling Param
        (39, 0.0), # Operator 1 Env Scaling Param
        (40, 0.0), # Operator 1 Env Scaling Param
        (41, 0.0), # Operator 1 Env Scaling Param
        (42, 0.0), # Operator 1 Mod Sensitivity
        (43, 0.0), # Operator 1 Key Velocity
        (44, 1.0), # Operator 1 On/Off switch
    ])

    # Override some of Operator 2 parameters
    overridden_parameters.extend([
        (45, 0.9), # Operator 2 Attack Rate (No attack on operator 2)
        (49, 1.0), # Operator 2 Attack Level
        (53, 1.0), # Operator 2 Gain (Operator 2 always outputs)
        (54, 0.0), # Operator 1 Mode (1.0 is Fixed Frequency)
        (58, 0.0), # Operator 1 Env Scaling Param
        (59, 0.0), # Operator 1 Env Scaling Param
        (60, 0.0), # Operator 1 Env Scaling Param
        (61, 0.0), # Operator 1 Env Scaling Param
        (62, 0.0), # Operator 1 Env Scaling Param
        (63, 0.0), # Operator 1 Env Scaling Param
        (64, 0.0), # Operator 1 Mod Sensitivity
        (65, 0.0), # Operator 1 Key Velocity
        (66, 1.0), # Operator 1 On/Off switch
    ])

    # Override operators 3 through 6
    overridden_parameters.extend([(i, 0.0) for i in range(67, 155)])
    return overridden_parameters


def generate_configurations(synth, values=np.arange(0, 1, 0.1)):
    config_path = "../data/param_weighting/configs"

    default_overrides = generate_default_overidden_params()
    parameters_to_weight = [parameter for parameter in range(22, 155) 
                                if parameter not in default_overrides]
    parameter_description = synth.parameter_desc
    print(parameters_to_weight)
    for parameter in parameters_to_weight:
        for value in values:
            overridden_parameters = default_overrides.copy()
            overridden_parameters.append((parameter, value))
            synth.set_overridden_parameters(overridden_parameters)
            synth.save_state(os.path.join(config_path, f"{parameter_description[parameter]['name']}_{value}.json"))
    
def main():
    pass
    # synth.load_state(config_path)
    # synth.randomize_patch()



if __name__ == '__main__':
    vst_path = os.path.join("..", "vsts", "Dexed.dll")
    print(vst_path)
    synth = SynthDawDreamer(vst_path, note_length_secs=1.0, render_length_secs=1.0)

    generate_configurations(synth)
    main()