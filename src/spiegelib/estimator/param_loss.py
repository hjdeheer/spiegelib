#!/usr/bin/env python
"""
Parameter loss function
"""

import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError


class ParameterLoss(tf.keras.losses.Loss):

    def __init__(self, automatable_keys, num_bins, weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.automatable_keys = automatable_keys
        self.num_bins = num_bins
        self.weight = weight
        self.cce = CategoricalCrossentropy()

    def call(self, Y_true, Y_pred):
        losses = []
        pointer = 0

        for parameter in self.automatable_keys:
            max_bins = self.num_bins

            if parameter['isDiscrete']:
                max_bins = parameter['max'] + 1

            losses.append(self.weight * self.cce(Y_true[pointer:pointer + max_bins],
                                                 Y_pred[pointer:pointer + max_bins]) / max_bins)
            pointer += max_bins

        return tf.math.reduce_mean(losses)

    def get_config(self):
        parent_config = super().get_config()
        return {
            **parent_config,
            "automatable_keys": self.automatable_keys,
            "num_bins": self.num_bins
        }
