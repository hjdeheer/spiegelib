"""
Tests for RMSE class

"""

import pytest
import numpy as np

from spiegelib import AudioBuffer
from spiegelib.features import RMSE, Spectrogram, MelSpectrogram
import librosa
import torch
import torchaudio
import utils

class TestRMSE():

    def test_empty_construction(self):

        spectrogram = Spectrogram(
                        frame_size=2048,
                        window_length=None,
                        hop_size=512,
                        center=True,
                        pad_mode="reflect",
                        power=2.0
                    )

        rms = RMSE(spectrogram)

    def test_sound2synth_spec_equal(self):

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

        rms = RMSE(sp_spec)
        sp_rms = rms(audio)

        s2s_rms = (torch_spec**2).mean(dim=-1,keepdim=True)**0.5


        mse = ((s2s_rms - sp_rms) ** 2).mean()
        error = 0.000001
        assert mse < error


    def test_sound2synth_mel_equal(self):

        sample_rate = 44100
        n_fft = 2048
        win_length = None
        hop_length = 512
        center =True
        pad_mode ="reflect"
        power = 2.0
        norm ='slaney'
        onesided = True
        n_mels = 128

        sine = utils.make_test_sine(2048, 440, 44100)
        audio = AudioBuffer(sine, 44100)

        sp_mel = MelSpectrogram(
                    sample_rate=sample_rate,
                    frame_size=n_fft,
                    window_length=win_length,
                    hop_size=hop_length,
                    center=True,
                    pad_mode="reflect",
                    power=2.0,
                    norm='slaney',
                    n_mels=n_mels
                )

        torch_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm='slaney',
            onesided=True,
            n_mels=n_mels
        )(torch.Tensor(sine))

        rms = RMSE(sp_mel)
        sp_rms = rms(audio)

        s2s_rms = (torch_mel**2).mean(dim=-1,keepdim=True)**0.5


        mse = ((s2s_rms - sp_rms) ** 2).mean()
        error = 0.000001
        assert mse < error






    