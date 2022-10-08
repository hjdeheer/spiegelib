#!/usr/bin/env python
"""
Convolutional Neural Network based on the 6-layer deep model proposed by
Barkan et al. [1]_
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

from spiegelib.estimator.tf_estimator_base import TFEstimatorBase
from .conv_s2s import ConvBackBone

class Conv8(TFEstimatorBase):
    """
    :param input_shape: Shape of matrix that will be passed to model input
    :type input_shape: tuple
    :param num_outputs: Number of outputs the model has
    :type numOuputs: int
    :param kwargs: optional keyword arguments to pass to
        :class:`spiegelib.estimator.TFEstimatorBase`
    """

    def __init__(self, input_shape, num_outputs, **kwargs):
        """
        Constructor
        """

        super().__init__(input_shape, num_outputs, **kwargs)


    def build_model(self):
        """
        Construct 8-layer CNN Model
        """
        self.model = tf.keras.Sequential()
        self.model.add(ConvBackBone(input_dim=self.input_shape, output_dim=self.num_outputs))
        dim = (None, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        self.model.build(input_shape=dim)
        self.model.compile(
            optimizer=tf.optimizers.Adam(),
            loss=TFEstimatorBase.rms_error,
            metrics=['accuracy']
        )
