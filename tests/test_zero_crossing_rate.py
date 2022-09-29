"""
Tests for ZeroCrossingRate class

"""

import pytest
import numpy as np

from spiegelib import AudioBuffer
from spiegelib.features import ZeroCrossingRate
import librosa
import torch
import utils

class TestZeroCrossingRate():

    def test_empty_construction(self):
        zcr = ZeroCrossingRate()

    def test_sound2synth_equal(self):

        sine = utils.make_test_sine(2048, 440, 44100)
        audio = AudioBuffer(sine, 44100)

        zcr = ZeroCrossingRate(hop_size=512)

        sp_zcr = zcr(audio)

        basic_args = {
            'hop_length': 512,
        }
        s2s_zcr = torch.tensor(librosa.feature.zero_crossing_rate(sine, **basic_args)).transpose(-1,-2).float()


        mse = ((s2s_zcr - sp_zcr) ** 2).mean()
        error = 0.000001
        assert mse < error







    