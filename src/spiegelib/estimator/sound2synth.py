import tensorflow as tf
from .conv_s2s import ConvBackBone
from .linear import LinearBackBone

class Sound2Synth(tf.keras.Model):
  #Add all components here:

  def __init__(self, output_dim= 2048):
    super(Sound2Synth, self).__init__()
    self.conv1 = ConvBackBone(in_channels=1, output_dim=512)
    self.lin1 = LinearBackBone(hidden_dim=256, output_dim=128)
    self.concat = tf.keras.layers.Concatenate()

    self.lin2 = tf.keras.layers.Dense(output_dim, use_bias=True)
    self.act1 = tf.keras.layers.LeakyReLU(alpha=1e-2)



  #Inputs is an array of features
  def call(self, inputs):
    conv = self.conv1(inputs[0])
    lin = self.lin1(inputs[1])
    concatted = self.concat([conv, lin])
    x = self.lin2(concatted)
    x = self.act1(x)
    return x

