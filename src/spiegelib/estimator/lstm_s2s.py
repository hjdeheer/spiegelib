#!/usr/bin/env python
"""
LSTM backbone for processing features.
From Sound2Synth (Chen, Ying et al, 2022)
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class LSTMBackBone(layers.Layer):

    def __init__(self, input_dim, output_dim=2048):
        pass
        
    
    def call(self, inputs): 
        pass