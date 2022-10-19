#!/usr/bin/env python
"""
Parameter loss function
"""

import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
import numpy as np

class ParameterLoss(tf.keras.losses.Loss):

    def __init__(self, automatable_keys, num_bins, weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.automatable_keys = automatable_keys
        self.num_bins = num_bins
        self.weight = weight
        self.cce = CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, Y_true, Y_pred):
        losses = []
        pointer = 0

        for parameter in self.automatable_keys:
            max_bins = self.num_bins
            
            if parameter['isDiscrete']:
                max_bins = parameter['max'] + 1

            pred_slice = Y_pred[:, pointer : pointer + max_bins]
            true_slice = Y_true[:, pointer : pointer + max_bins]

            pred_slice = tf.nn.softmax(pred_slice)
            
            loss = self.cce(true_slice, pred_slice) / max_bins
            losses.append(loss)
            pointer += max_bins

        # Returns a [64] tensor, the loss for each sample in the batch
        # Check https://stackoverflow.com/questions/63390725/should-the-custom-loss-function-in-keras-return-a-single-loss-value-for-the-batc
        final_loss = tf.math.reduce_sum(losses, axis=0)
        return final_loss

    def get_config(self):
        parent_config = super().get_config()
        return {
            **parent_config,
            "automatable_keys": self.automatable_keys,
            "num_bins": self.num_bins
        }
