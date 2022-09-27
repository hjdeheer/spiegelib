"""
Tests for AmplitudeEnvelope class

"""

import pytest
import numpy as np

from spiegelib import AudioBuffer
from spiegelib.features import AmplitudeEnvelope, Spectrogram
import librosa
import torch
import torchaudio
import utils

class TestSpectralFlatness():

    def test_empty_construction(self):

        spectrogram = Spectrogram(
                        frame_size=2048,
                        window_length=None,
                        hop_size=512,
                        center=True,
                        pad_mode="reflect",
                        power=2.0
                    )

        amplitude_envelope = AmplitudeEnvelope(spectrogram)

    def test_sound2synth_equal(self):

        sample_rate = 44100
        n_fft = 2048
        win_length = None
        hop_length = 512
        center =True
        pad_mode ="reflect"
        power = 2.0

        sine = utils.make_test_sine(2048, 440, 44100)
        audio = AudioBuffer(sine, 44100)

        sp_spec = Spectrogram(
                        frame_size=n_fft,
                        window_length=win_length,
                        hop_size=hop_length,
                        center=center,
                        pad_mode=pad_mode,
                        power=power
                    )

        torch_spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=center,
            pad_mode=pad_mode,
            power=power
        )(torch.Tensor(sine))

        amplitude_envelope = AmplitudeEnvelope(sp_spec)
        sp_amplitude_envelope = amplitude_envelope(audio)

        s2s_amplitude_envelope =  torch_spec.max(dim=-1,keepdim=True)[0]


        mse = ((s2s_amplitude_envelope - sp_amplitude_envelope) ** 2).mean()
        error = 0.000001
        assert mse < error







    