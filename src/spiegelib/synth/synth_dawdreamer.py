#!/usr/bin/env python

"""
:class:`SynthDawDreamer` is a class for interacting with VST synthesizer plugins using DawDreamer.

This class relies on the dawdreamer package developed by DBraun for interacting
with VSTs. A VST can be loaded, parameters displayed and modified, random patches generated,
and audio rendered for further processing.
"""

from __future__ import print_function
import numpy as np

import dawdreamer as daw
from scipy.io import wavfile as wav
from spiegelib import AudioBuffer
from spiegelib.synth.synth_base import SynthBase


class SynthDawDreamer(SynthBase):
    """
    :param plugin_path: path to vst plugin binary, defaults to None
    :type plugin_path: str, optional
    :param keyword arguments: see :class:`spiegelib.synth.synth_base.SynthBase` for details
    """

    def __init__(self, plugin_path=None, **kwargs):
        super().__init__(**kwargs)

        if plugin_path:
            self.load_plugin(plugin_path)

        else:
            self.engine = None
            self.loaded_plugin = False


    def load_plugin(self, plugin_path):
        """
        Loads a synth VST plugin

        :param plugin_path: path to vst plugin binary
        :type plugin_path: str
        """
        #Initialize the engine and generator
        self.engine = daw.RenderEngine(self.sample_rate, self.buffer_size)
        self.generator = self.engine.make_plugin_processor("Synth", plugin_path)

        #Initialize parameters
        self.parametersDesc = self.generator.get_parameters_description()
        self.parameters = parse_parameters(self.parametersDesc)

        #Initialize patch
        self.patch = [None] * len(self.parameters)
        self.get_curr_patch()

        #Override params if available
        for i in range(len(self.overridden_params)):
            index, value = self.overridden_params[i]
            self.generator.set_parameter(int(index), value)

        #Set first midi note
        self.generator.add_midi_note(self.midi_note, self.midi_velocity, 0, self.note_length_secs)

        self.loaded_plugin = True

    def clear_midi(self):
        self.generator.clear_midi()

    def add_note(self, note, velocity, start, duration):
        self.generator.add_midi_note(note, velocity, start, duration)


    def load_patch(self):
        """
        Update patch parameter in generator. Overridden parameters will not be effected.
        """

        # Check for parameters to include in patch update
        for param in self.patch:
            if not self.is_valid_parameter_setting(param):
                raise Exception(
                    'Parameter %s is invalid. Must be a valid '
                    'parameter number and be in range 0-1. '
                    'Received %s' % param
                )
            else:
                self.generator.set_parameter(param[0], param[1])


    def is_valid_parameter_setting(self, parameter):
        """
        Checks to see if a parameter is valid for the currently loaded synth.

        :param parameter: A parameter tuple with form `(parameter_index, parameter_value)`
        :type parameter: tuple
        """
        return (
            parameter[0] in self.parameters
            and parameter[1] >= 0.0
            and parameter[1] <= 1.0
        )



    def render_patch(self):
        """
        Render the current patch. Uses the values of midi_note, midi_velocity, note_length_secs,
        and render_length_secs to render audio. Plugin must be loaded first.
        """
        if self.loaded_plugin:
            graph = [(self.generator, [])]
            self.engine.load_graph(graph)
            self.engine.render(self.render_length_secs)
            self.rendered_patch = True

        else:
            print("Please load plugin first.")


    def get_audio(self):
        """
        Return monophonic audio from rendered patch

        :return: An audio buffer of the rendered patch
        :rtype: :class:`spiegelib.core.audio_buffer.AudioBuffer`
        """

        if self.rendered_patch:
            audio = self.engine.get_audio()
            return audio

        else:
            raise Exception('Patch must be rendered before audio can be retrieved')


    def write_to_wav(self, audio, path):
        wav.write(path, self.sample_rate, audio.transpose())


    #TODO Implement 3 strategies here! Start with basic uniform
    def randomize_patch(self):
        """
        Randomize the current patch. Overridden parameteres will be unaffected.
        """
        technique = "uniform"

        if self.loaded_plugin:
            random_patch = []
            #First type
            if technique == "uniform":
                for key, value in self.patch:
                    #If cutoff or output param -> turn volume to 1
                    #TODO Determine what to do with volume params
                    if key == 0 or key == 2:
                        random_patch.append((key, 1))
                    #If we can automate this parameter:
                    elif self.parametersDesc[key]["isAutomatable"] and not self.parametersDesc[key]["isDiscrete"]:
                        random_patch.append((key, np.random.uniform(0,1)))
                    elif self.parametersDesc[key]["isDiscrete"]:
                        print(self.parametersDesc[key])

            self.set_patch(random_patch)
        else:
            print("Please load plugin first.")

    def get_curr_patch(self):
        """
        Obtains the current configuration of parameters (patch)
        Sets self.patch to an updatedlist with tuples where [0] is the index i
        and [1] value of parameter i
        """
        for i in range(len(self.patch)):
            self.patch[i] = (i, self.generator.get_parameter(i))



################################################################################


def parse_parameters(param_list):
    """
    Parse parameter list return by dawdreamer into a dictionary keyed on parameter
    index with values being the name / short descriptions for the parameter at that index.

    :param param_str: A parameter description list returned by dawdreamer
    :type param_str: str
    :returns: A dictionary with parameter index as keys and parameter name / description for values
    :rtype: dict
    """

    param_dict = {}
    for d in param_list:
        param_index = d['index']
        param_dict[param_index] = d['name']
    return param_dict


