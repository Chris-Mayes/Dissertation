import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import (Dense, 
                                     BatchNormalization, 
                                     LeakyReLU, 
                                     Reshape, 
                                     Conv2DTranspose,
                                     Conv2D,
                                     Dropout,
                                     Flatten)

import tensorflow_addons as tfa

from tensorflow.keras.applications import EfficientNetB3
from keras.models import Model
import efficientnet.tfkeras
import efficientnet.keras as efn 

def get_resnet_generator(
    filters=64,
    num_downsampling_blocks=2,
    num_residual_blocks=5,
    num_upsample_blocks=2,
    name=None,
    ):
    
    #inputs = tf.keras.layers.Input(shape=[256,256,3])

    x = efn.EfficientNetB3(include_top=False, weights=None, input_shape=(256,256,3), pooling=None)
    
    model = keras.models.Model(x, name=name)

    model.compile()
    model.summary()
    return model

get_resnet_generator(
    filters=64,
    num_downsampling_blocks=2,
    num_residual_blocks=5,
    num_upsample_blocks=2,
    name=None,
    )