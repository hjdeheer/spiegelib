from distutils.command.config import config
import numpy as np
from tqdm import trange
from spiegelib.synth import SynthDawDreamer
from spiegelib.evaluation import MFCCEval
import json
import os


def generate_default_overidden_params( ):
    """
        Generate the default overridden parameters for the Dexed synthesizer.
        Initial config is taken from the Spiegelib guide.

        :return: List[Tuple[Int, Float]] List of overridden parameters. These parameters will
        not change when randomize_patch is called.
    """

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
    
def generate_parameter_weights(synth, n_samples=5000, values=np.arange(0, 1.1, 0.1).round(1)):
    """
        Generate the parameter weights by calculate the MFCC error over different configurations 
        denoted by values. Each parameter in parameters_to_weight is fixed to values.

        :synth: spiegelib.synth.SynthDawDreamer, the synthesizer used to generate sounds.
        :n_samples: Int, the number of samples to generalize over.
        :values: List[Float], the values with which to fix a parameter to.

        :return: Dict[Int, Float], dictionary containing the weighting for each parameter.
    """
    default_overrides = generate_default_overidden_params()
    parameters_to_weight = [parameter for parameter in range(22, 155) 
                                if parameter not in dict(default_overrides)]
    weight_dict = {}

    # Weight dictionary holds a list of size n_samples per parameter
    for parameter in parameters_to_weight:
        weight_dict[parameter] = []

    for _ in trange(0, n_samples):
        # Set the default overridden parameters and randomize the patch    
        synth.set_overridden_parameters(default_overrides)
        synth.randomize_patch(technique="uniform")

        patch = synth.patch

        for parameter in parameters_to_weight:
            parameter_audio = []

            for value in values:
                # Remove the current parameter from the patch list
                new_patch = [(p, v) for (p, v) in patch if p != parameter]
                # Add the current parameter with the specified value
                new_patch.append((parameter, value))

                # Load the new patch in and render the audio
                synth.patch = new_patch
                synth.load_patch()
                synth.render_patch()
                parameter_audio.append(synth.get_audio())

          
            mfcc_distance = calculate_mean_MFCC_error(parameter_audio)
            # Add the mean MAE score to the dictionary
            weight_dict[parameter].append(mfcc_distance)
    
    weights = {}

    # Average the per sample weight scores for each parameter
    print(weight_dict)
    for (key, value) in weight_dict.items():
        weights[key] =  np.mean(value)

    return weights


def calculate_mean_MFCC_error(audio_list, error_type="mean_abs_error"):
    """
    Calculate the mean MFCC error by pair-wise comparing audio samples

    :audio_list: List[AudioBuffer], the list of generated samples
    :error_type: Str, the error to be returned (See spiegelib.evaluation.EvalaluationBase)
    """
    # Do consecutive MFCC analysis on samples in parameter_audio
    mfcc_distances = []
    # for audio_1, audio_2 in zip(audio_list, audio_list[1:]):
    mfcc_eval = MFCCEval(audio_list[:-1], [audio_list[1:]])
    mfcc_eval.evaluate()

        # Extract the MAE score from the MFCC Eval class
        # mfcc_distances.append(mfcc_eval.get_scores()['target_0']['source_0'][error_type])  

    return mfcc_eval.get_stats()['source_0'][error_type]



if __name__ == '__main__':
    vst_path = os.path.join("..", "vsts", "Dexed.dll")
    synth = SynthDawDreamer(vst_path, note_length_secs=1.0, render_length_secs=1.0)

    config_path = "../data/param_weighting/configs"

    weights = generate_parameter_weights(synth, n_samples=5)

    print(weights)

    np.save("weights.npy", weights, allow_pickle=True)