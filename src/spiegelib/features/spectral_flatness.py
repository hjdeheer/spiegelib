#!/usr/bin/env python
"""
Spectral flatness summarized over time using mean and variance.
"""

import numpy as np
import librosa
from spiegelib import AudioBuffer
from spiegelib.features.features_base import FeaturesBase

class SpectralFlatness(FeaturesBase):
    """
    Args:
        frame_size (int, optional): size of FFT, defaults to 2048
        hop_size (int, optional): size of hop shift in samples, defuault to 512
        window_length (int or None, optional): Specifies the size of the window
        scale_axis (int, tuple, None): When applying scaling, determines which dimensions
            scaling be applied along. Defaults to 0, which scales each feature independently.
        kwargs: Keyword arguments, see :class:`spiegelib.features.features_base.FeaturesBase`.
    """

    def __init__(self, frame_size=2048, hop_size=512, window_length=None, scale_axis=0, **kwargs):
        """
        Constructor
        """

        self.frame_size = frame_size
        self.hop_size = hop_size
        self.window_length = window_length
        super().__init__(scale_axis=scale_axis, **kwargs)


    def get_features(self, audio):
        """
        Extract spectral flatness and return results.

        Args:
            audio (:ref:`AudioBuffer <audio_buffer>`): input audio

        Returns:
            np.ndarray: Results of spectral flatness extraction. Format depends on\
            output type set during construction.
        """

        if not isinstance(audio, AudioBuffer):
            raise TypeError('audio must be AudioBuffer, recieved %s' % type(audio))

        if audio.get_sample_rate() != self.sample_rate:
            raise ValueError(
                'audio buffer samplerate does not equal feature '
                'extraction rate, %s != %s' % (audio.get_sample_rate(), self.sample_rate)
            )

        features = librosa.feature.spectral_flatness(
            y=audio.get_audio(),
            n_fft=self.frame_size,
            hop_length=self.hop_size,
            win_length=self.window_length
        )

        features = np.transpose(features)

        return features
