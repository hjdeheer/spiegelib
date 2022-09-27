#!/usr/bin/env python
"""
CQT Chromagrams as defined by Sound2Synth
"""

import numpy as np
import librosa
from spiegelib import AudioBuffer
from spiegelib.features.features_base import FeaturesBase

class CQTChromagram(FeaturesBase):

    def __init__(self, 
                 n_octaves=1, 
                 fmins=[32.703,65.406,130.81,261.63,523.25,1046.5,2093.0,4186.0,8372.0],
                 n_chroma=48, 
                 bins_per_octave=48,
                 **kwargs):

        self.n_octaves = n_octaves
        self.fmins = fmins
        self.n_chroma = n_chroma
        self.bins_per_octave = bins_per_octave

        super().__init__(**kwargs)


    def get_features(self, audio):
        """
        Run CQT Chromagram extraction on audio buffer.

        Args:
            audio (:ref:`AudioBuffer <audio_buffer>`): input audio

        Returns:
            np.ndarray: Results of CQT Chromagram extraction. Format depends on output type set during\
                construction.
        """

        if not isinstance(audio, AudioBuffer):
            raise TypeError('audio must be AudioBuffer, recieved %s' % type(audio))

        if audio.get_sample_rate() != self.sample_rate:
            raise ValueError(
                'audio buffer samplerate does not equal feature '
                'extraction rate, %s != %s' % (audio.get_sample_rate(), self.sample_rate)
            )

        features = [np.transpose(librosa.feature.chroma_cqt(
            y=audio.get_audio(),
            sr=self.sample_rate,
            n_octaves=self.n_octaves,
            fmin=self.fmins[oct-1],
            n_chroma=self.n_chroma,
            bins_per_octave=self.bins_per_octave
        ), ) for oct in range(1, 10)]

        features = np.concatenate(features, axis=1)

        if self.time_major:
            features = np.transpose(features)

        return features
