"""
Tests for RMS class

"""

import pytest
import numpy as np

from spiegelib import AudioBuffer
from spiegelib.features import RMS
import librosa
import torch
import utils

class TestRMS():

    def test_empty_construction(self):
        rms = RMS()

    def test_sound2synth_equal(self):

        sine = utils.make_test_sine(2048, 440, 44100)
        audio = AudioBuffer(sine, 44100)

        rms = RMS(hop_size=512)

        sp_rms = rms(audio)

        basic_args = {
            'hop_length': 512
        }
        s2s_rms = torch.tensor(librosa.feature.rms(sine, **basic_args)).transpose(-1,-2).float()


        mse = ((s2s_rms - sp_rms) ** 2).mean()
        error = 0.000001
        assert mse < error







    