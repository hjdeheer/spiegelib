"""
Tests for SpectralFlatness class

"""

import pytest
import numpy as np

from spiegelib import AudioBuffer
from spiegelib.features import SpectralFlatness
import librosa
import torch
import utils

class TestSpectralFlatness():

    def test_empty_construction(self):
        spectral_flatness = SpectralFlatness()

    def test_sound2synth_equal(self):

        sine = utils.make_test_sine(2048, 440, 44100)
        audio = AudioBuffer(sine, 44100)

        spectral_flatness = SpectralFlatness(frame_size=2048, hop_size=512, window_length=None)

        sp_flatness_feature = spectral_flatness(audio)

        fft_args = {
            'n_fft': 2048,
            'win_length': None,
            'hop_length': 512
        }
        s2s_flatness_feature = torch.tensor(librosa.feature.spectral_flatness(sine, **fft_args)).transpose(-1,-2).float()


        mse = ((s2s_flatness_feature - sp_flatness_feature) ** 2).mean()
        error = 0.000001
        assert mse < error







    