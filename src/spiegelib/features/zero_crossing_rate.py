#!/usr/bin/env python
"""
Zero Crossing Rate
"""

from symbol import power
import numpy as np
import librosa
from spiegelib import AudioBuffer
from spiegelib.features.features_base import FeaturesBase

class ZeroCrossingRate(FeaturesBase):
    """
    Args:
        hop_size (int, optiona): hop length in samples, defaults to 512
        scale_axis (int, tuple, None): When applying scaling, determines which dimensions
            scaling be applied along. Defaults to 0, which scales each MFCC and time series
            component independently.
        kwargs: Keyword arguments, see :class:`spiegelib.features.features_base.FeaturesBase`.
    """

    def __init__(self, hop_size=512, scale_axis=0, **kwargs):
        """
        Contructor
        """

        self.hop_size = hop_size
        super().__init__(scale_axis=scale_axis, **kwargs)


    def get_features(self, audio):
        """
        Run MFCC extraciton on audio buffer.

        Args:
            audio (:ref:`AudioBuffer <audio_buffer>`): input audio

        Returns:
            np.ndarray: Results of MFCC extraction. Format depends on output type set during\
                construction.
        """

        if not isinstance(audio, AudioBuffer):
            raise TypeError('audio must be AudioBuffer, recieved %s' % type(audio))

        if audio.get_sample_rate() != self.sample_rate:
            raise ValueError(
                'audio buffer samplerate does not equal feature '
                'extraction rate, %s != %s' % (audio.get_sample_rate(), self.sample_rate)
            )


        # Librosa MFCC calls the mel filters with norm="Slaney", thus we do not need to provide this
        features = librosa.feature.zero_crossing_rate(
            y=audio.get_audio(),
            hop_length=self.hop_size
        )

        features = np.transpose(features)

        return features
