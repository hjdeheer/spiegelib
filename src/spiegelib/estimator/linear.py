#!/usr/bin/python
"""
A Linear backbone for processing features. From Sound2Synth (Chen, Ying et al, 2022)
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class LinearBackBone(layers.Layer):

    def __init__(self, input_dim, hidden_dim=2048, output_dim=2048):
        super(LinearBackBone, self).__init__()
        
        self.model = tf.keras.Sequential()
        self.model.add(layers.Dense(hidden_dim, use_bias=True))
        self.model.add(layers.LeakyReLU(alpha=1e-2))
        self.model.add(layers.Dense(output_dim, use_bias=True))
        self.model.add(layers.LeakyReLU(alpha=1e-2))

    
    def call(self, inputs):
        y = inputs
        x = tf.reshape(y, [y.shape[0], -1])
        return self.model(x)