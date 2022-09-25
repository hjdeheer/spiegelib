"""
Tests for MelSpectrogram class

TODO -- include a test that verifies the output
"""
import pytest

import numpy as np

from spiegelib import AudioBuffer
import spiegelib.features as sf
import torch
import torchaudio

import utils

class TestSpectrogram():

    def test_empty_construction(self):
        mel = sf.Spectrogram()


    def test_librosa_equal_to_torch(self):

        sample_rate = 44100
        n_fft = 2048
        win_length = None
        hop_length = 512
        center =True
        pad_mode ="reflect"
        power = 2.0

        sine = utils.make_test_sine(2048, 440, sample_rate)
        audio = AudioBuffer(sine, sample_rate)
        librosa_spec = sf.Spectrogram(
                             frame_size=n_fft,
                             window_length=win_length,
                             hop_size=hop_length,
                             center=center,
                             pad_mode=pad_mode,
                             power=power
                            )(audio)

        torch_spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=center,
            pad_mode=pad_mode,
            power=power
        )(torch.Tensor(sine))

        print(librosa_spec)

        mse = ((torch_spec - librosa_spec) ** 2).mean()
        error = 0.000001
        assert mse < error

