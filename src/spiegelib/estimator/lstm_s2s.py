#!/usr/bin/env python
"""
LSTM backbone for processing features.
From Sound2Synth (Chen, Ying et al, 2022)
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class LSTMBackBone(layers.Layer):

    # Sound2Synth uses a 2 layered Bidirectional LSTM and feeds the output to a Fully Connected layer
    def __init__(self, hidden_dim=256, output_dim=128):
        super(LSTMBackBone, self).__init__()
        self.lstm1 = layers.Bidirectional(layers.LSTM(units=hidden_dim, return_sequences=True))
        self.lstm2 = layers.Bidirectional(layers.LSTM(units=2*hidden_dim))
        self.dense1 = layers.Dense(output_dim, use_bias=True)
        self.act1 = layers.LeakyReLU(alpha=1e-2)
        
        
    
    def call(self, inputs): 
        y = inputs
        t = self.lstm1(y)
        t = self.lstm2(t)
        x = self.dense1(t)
        x = self.act1(x)

        return x