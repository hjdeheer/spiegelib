"""
Main Conv backbone for processing spectogram feature.
From Sound2Synth (Chen, Ying et al, 2022)
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class ConvBackBone(layers.Layer):

    # Sound2Synth uses VGG 11 network
    def __init__(self, input_dim, output_dim= 2048, **kwargs):
        super(ConvBackBone, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        #Conv2d layer -> batch normalization -> leaky Relu _> max pool2d
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, input_shape=self.input_dim)
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.LeakyReLU(1e-2)
        self.max1 = tf.keras.layers.MaxPool2D()

        self.pad2 = tf.keras.layers.ZeroPadding2D(padding=1)
        self.conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3)
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.LeakyReLU(1e-2)
        self.max2 = tf.keras.layers.MaxPool2D()

        self.pad3 = tf.keras.layers.ZeroPadding2D(padding=1)
        self.conv3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3)
        self.norm3 = tf.keras.layers.BatchNormalization()
        self.act3 = tf.keras.layers.LeakyReLU(1e-2)

        self.pad4 = tf.keras.layers.ZeroPadding2D(padding=1)
        self.conv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=3)
        self.norm4 = tf.keras.layers.BatchNormalization()
        self.act4 = tf.keras.layers.LeakyReLU(1e-2)
        self.max3 = tf.keras.layers.MaxPool2D()

        self.pad5 = tf.keras.layers.ZeroPadding2D(padding=1)
        self.conv5 = tf.keras.layers.Conv2D(filters=512, kernel_size=3)
        self.norm5 = tf.keras.layers.BatchNormalization()
        self.act5 = tf.keras.layers.LeakyReLU(1e-2)

        self.pad6 = tf.keras.layers.ZeroPadding2D(padding=1)
        self.conv6 = tf.keras.layers.Conv2D(filters=512, kernel_size=3)
        self.norm6 = tf.keras.layers.BatchNormalization()
        self.act6 = tf.keras.layers.LeakyReLU(1e-2)
        self.max4 = tf.keras.layers.MaxPool2D()

        self.pad7 = tf.keras.layers.ZeroPadding2D(padding=1)
        self.conv7 = tf.keras.layers.Conv2D(filters=512, kernel_size=3)
        self.norm7 = tf.keras.layers.BatchNormalization()
        self.act7 = tf.keras.layers.LeakyReLU(1e-2)

        self.pad8 = tf.keras.layers.ZeroPadding2D(padding=1)
        self.conv8 = tf.keras.layers.Conv2D(filters=512, kernel_size=3)
        self.norm8 = tf.keras.layers.BatchNormalization()
        self.act8 = tf.keras.layers.LeakyReLU(1e-2)
        self.max5 = tf.keras.layers.MaxPool2D()

        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(output_dim, use_bias=True)
        self.act9 = layers.LeakyReLU(alpha=1e-2)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        })
        return config

    def call(self, inputs):
        y = inputs
        x = self.max1(self.act1(self.norm1(self.conv1(y))))
        x = self.max2(self.act2(self.norm2(self.conv2(self.pad2(x)))))
        x = self.act3(self.norm3(self.conv3(self.pad3(x))))
        x = self.max3(self.act4(self.norm4(self.conv4(self.pad4(x)))))
        x = self.act5(self.norm5(self.conv5(self.pad5(x))))
        x = self.max4(self.act6(self.norm6(self.conv6(self.pad6(x)))))
        x = self.act7(self.norm7(self.conv7(self.pad7(x))))
        x = self.max5(self.act8(self.norm8(self.conv8(self.pad8(x)))))
        x = self.flatten(x)
        x = self.act9(self.dense1(x))
        return x
