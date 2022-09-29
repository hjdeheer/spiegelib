"""
Tests for CQTChromagram Sound2Synth class
"""

import pytest
import numpy as np

from spiegelib import AudioBuffer
from spiegelib.features import CQTChromagram

import torch
import torchaudio
import librosa

import utils

class TestCQTChromagrams():

    def test_empty_construction(self):
        cqt = CQTChromagram()


    def test_sine_chroma(self):

        fmins = [32.703,65.406,130.81,261.63,523.25,1046.5,2093.0,4186.0,8372.0]

        sine = utils.make_test_sine(2048, 440, 44100)
        audio = AudioBuffer(sine, 44100)
        cqt = CQTChromagram(n_octaves=1, fmins=fmins, n_chroma=48, bins_per_octave=48)
        features = cqt(audio)

        assert features.shape == (5, 48 * len(fmins))


    def test_compare_s2s(self):

        fmins = [32.703,65.406,130.81,261.63,523.25,1046.5,2093.0,4186.0,8372.0]

        sine = utils.make_test_sine(2048, 440, 44100)
        audio = AudioBuffer(sine, 44100)

        cqt = CQTChromagram(n_octaves=1, fmins=fmins, n_chroma=48, bins_per_octave=48)
        sp_chroma = cqt(audio)

        oct_cqt = [
            torch.tensor(librosa.feature.chroma_cqt(sine, sr=44100, n_octaves=1, fmin=fmins[oct-1], n_chroma=48, bins_per_octave=48)).transpose(-1,-2).float() for oct in range(1,10)
        ]
        s2s_chroma = torch.cat(oct_cqt, dim=-1)

        mse = ((s2s_chroma - sp_chroma) ** 2).mean()
        error = 0.000001
        assert mse < error