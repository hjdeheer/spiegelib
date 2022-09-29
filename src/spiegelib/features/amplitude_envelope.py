#!/usr/bin/env python
"""
Amplitude Envelopes as defined by Sound2Synth ()
"""

import numpy as np

import librosa
from spiegelib import AudioBuffer
from spiegelib.features.features_base import FeaturesBase
import spiegelib.core.utils as utils

class AmplitudeEnvelope(FeaturesBase):

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


        features =  np.ndarray.max(self.spectrogram.get_features(audio), axis=-1, keepdims=True)

        if self.time_major:
            features = np.transpose(features)

        return features
