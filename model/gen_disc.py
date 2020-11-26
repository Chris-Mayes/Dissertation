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


class ReflectionPadding2D(layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """


    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")


def residual_block(
    x,
    activation,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
    use_bias=False,
):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.add([input_tensor, x])
    return x

def efficientnet_block(
    x,
    activation,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
    use_bias=False,
):
    
    return x
    

def downsample(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization()(x)
    if activation:
        x = activation(x)
    return x


def upsample(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    use_bias=False,
):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization()(x)
    if activation:
        x = activation(x)
    return x

def efficientnet_generator():
    model = efn.EfficientNetB3(include_top=False, weights="imagenet", input_shape=(256, 256, 3))
    
    # add new classifier layers
    x = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(4, 4), use_bias=False)(model.layers[-36].output)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(4, 4), use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(3, (8, 8), padding="valid")(x)
    x = layers.Activation("tanh")(x)

    # define new model
    model = Model(inputs=model.inputs, outputs=x)

    #model.compile()
    #model.summary()
    return model


def get_resnet_generator(
    filters=64,
    num_downsampling_blocks=2,
    num_residual_blocks=5,
    num_upsample_blocks=2,
    name=None,
):
    inputs = tf.keras.layers.Input(shape=[256,256,3])
    x = ReflectionPadding2D(padding=(3, 3))(inputs)
    x = layers.Conv2D(filters, (7, 7), use_bias=False)(
        x
    )
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Activation("relu")(x)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters, activation=layers.Activation("relu"))

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"))

    # Upsampling
    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, filters, activation=layers.Activation("relu"))

    # Final block
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(3, (7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(inputs, x, name=name)
    
    model.compile()
    model.summary()
    return model


def get_discriminator(
    filters=64, 
    num_downsampling=3, 
    name=None
    ):
    
    img_input = tf.keras.layers.Input(shape=[256,256,3], name=name + "_img_input")
    x = layers.Conv2D(
        filters,
        (4, 4),
        strides=(2, 2),
        padding="same",
    )(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(3):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(2, 2),
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(1, 1),
            )

    x = layers.Conv2D(
        1, (4, 4), strides=(1, 1), 
        padding="same"
    )(x)

    model = keras.models.Model(inputs=img_input, outputs=x, name=name)
    #model.compile()
    #model.summary()
    return model

  