#!/usr/bin/env python
"""
Spectrograms as defined by Sound2Synth ()
"""

import numpy as np

import librosa
from spiegelib import AudioBuffer
from spiegelib.features.features_base import FeaturesBase
import spiegelib.core.utils as utils

class Spectrogram(FeaturesBase):

    """
    Args:
        frame_size (int, optional): Size of FFT to use when calculating MFCCs, defaults to 2048
        window_length (int or None, optional): Specifies the size of the  
        hop_size (int, optional): hop length in samples, defaults to 512
        power (float, optional): Exponent for the magnitude 
        center (boolean, optional): True if signal should be padded 
        norm (str, optional): Normalization mode for triangles in the spectrogram
        pad_mode (str, optional): Padding mode to be used at the edges
        kwargs: Keyword arguments, see :class:`spiegelib.features.features_base.FeaturesBase`.
    """

    def __init__(self,
                 frame_size=2048,
                 window_length=None,
                 hop_size=512,
                 power=2.0,
                 center=True,
                 norm='slaney',
                 pad_mode="reflect",
                 **kwargs):
        
        self.frame_size = frame_size
        self.window_length = None
        self.hop_size = hop_size
        self.power = power
        self.center = center
        self.norm = norm
        self.pad_mode = pad_mode
        super().__init__(**kwargs)


    def get_features(self, audio):
        """
        Run Spectrogram extraction on audio buffer.

        Args:
            audio (:ref:`AudioBuffer <audio_buffer>`): input audio

        Returns:
            np.ndarray: Results of Spectrogram extraction. Format depends on output type set during\
                construction.
        """

        if not isinstance(audio, AudioBuffer):
            raise TypeError('audio must be AudioBuffer, recieved %s' % type(audio))

        if audio.get_sample_rate() != self.sample_rate:
            raise ValueError(
                'audio buffer samplerate does not equal feature '
                'extraction rate, %s != %s' % (audio.get_sample_rate(), self.sample_rate)
            )

        features, _ = librosa.core.spectrum._spectrogram(
            y=audio.get_audio(),
            win_length=self.window_length,
            n_fft=self.frame_size,
            hop_length=self.hop_size,
            power=self.power,
            center=self.center,
            pad_mode=self.pad_mode
        )

        if self.time_major:
            features = np.transpose(features)

        return features
