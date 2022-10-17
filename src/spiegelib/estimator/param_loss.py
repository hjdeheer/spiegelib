#!/usr/bin/env python
"""
Parameter loss function
"""

import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy

class ParameterLoss(tf.keras.losses.Loss):

    def __init__(self, synth, num_bins, weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.synth = synth
        self.num_bins = num_bins
        self.weight = weight
        self.cce = CategoricalCrossentropy()
    

    def call(self, Y_true, Y_pred):
        losses = []
        pointer = 0

        for parameter in self.synth.get_automatable_keys():
            max_bins = self.num_bins
            
            if parameter['isDiscrete']:
                max_bins = parameter['max']
            
            losses.append(self.weight * cce(Y_true[pointer:pointer+max_bins], Y_pred[pointer:pointer+max_bins]).numpy() / max_bins)
            pointer += max_bins
        
        return tf.math.reduce_mean(losses)

    def get_config(self):
        parent_config = super().get_config()
        return {
                **parent_config,
                "synth":self.synth,
                "num_bins":self.num_bins
                }
