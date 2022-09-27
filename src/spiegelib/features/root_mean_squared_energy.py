#!/usr/bin/env python
"""
Root mean squared energy as defined by Sound2Synth ()
"""

import numpy as np

import librosa
from spiegelib import AudioBuffer
from spiegelib.features.features_base import FeaturesBase
import spiegelib.core.utils as utils

class RMSE(FeaturesBase):

    """
    Args:
        spectrogram (Spectrogram, MelSpectrogram): the spectrogram to extract the amplitudes from
    """

    def __init__(self, spectrogram, **kwargs):
        self.spectrogram = spectrogram
        super().__init__(**kwargs)


    def get_features(self, audio):
        """
        Run Amplitude Envelope extraction on audio buffer.

        Args:
            audio (:ref:`AudioBuffer <audio_buffer>`): input audio

        Returns:
            np.ndarray: Results of Amplitude Envelope extraction. Format depends on the spectrogram used during\
                construction.
        """

        if not isinstance(audio, AudioBuffer):
            raise TypeError('audio must be AudioBuffer, recieved %s' % type(audio))

        if audio.get_sample_rate() != self.sample_rate:
            raise ValueError(
                'audio buffer samplerate does not equal feature '
                'extraction rate, %s != %s' % (audio.get_sample_rate(), self.sample_rate)
            )

        features =  np.mean(self.spectrogram.get_features(audio) ** 2, axis=-1, keepdims=True) ** 0.5

        if self.time_major:
            features = np.transpose(features)

        return features
