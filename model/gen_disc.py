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
    x = layers.Conv2D(dim,kernel_size,strides=strides,padding=padding,use_bias=use_bias,)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(dim,kernel_size,strides=strides,padding=padding,use_bias=use_bias,)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.add([input_tensor, x])
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

"""

def efficientnet_generator():
    model = efn.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(256, 256, 3))
    
    ""
    add pooling between conv2d and increase kernel_size(increases output for layer)
    increase strides?
    size of output (size of input - 1) * (stride) + kernel size
    ""

    # add new classifier layers
    x = layers.Conv2DTranspose(filters=640, kernel_size=(2, 2), strides=(2, 2), use_bias=False)(model.output)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
    x = tfa.layers.InstanceNormalization()(x)

    x = layers.Conv2DTranspose(filters=160, kernel_size=(8, 8), strides=(4, 4), use_bias=False)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tfa.layers.InstanceNormalization()(x)

    x = layers.Conv2DTranspose(filters=40, kernel_size=(4, 4), strides=(4, 4), use_bias=False)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tfa.layers.InstanceNormalization()(x)

    x = layers.Conv2DTranspose(filters=3, kernel_size=(4, 4), strides=(4, 4), use_bias=False)(x)
    #x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
    x = tfa.layers.InstanceNormalization()(x)
    
    # define new model
    model = Model(inputs=model.inputs, outputs=x)

    model.compile()
    model.summary()
    return model
"""

def efficient_block(x, name):

    efficientnet_model = efn.EfficientNetB3(
            include_top=False,
            weights="imagenet",
            input_shape=(256, 256, 3)
            )

    x
    
    for layer in efficientnet_model.layers:
        layer._name = layer.name + str(name)
    
    #efficientnet_model.trainable = False
    
    x = efficientnet_model.layers[30](x)
   
    x = efficientnet_model.layers[31](x)
    #x1 = efficientnet_model.layers[31](x)
    #x = efficientnet_model.layers[32](x1)
    #x = efficientnet_model.layers[33](x)
    #x = efficientnet_model.layers[34](x)
    #x2 = efficientnet_model.layers[35](x)
    #x = tf.keras.layers.Multiply()([x1, x2])

    x = efficientnet_model.layers[37](x)
    x5 = efficientnet_model.layers[38](x)
    x = efficientnet_model.layers[39](x5)
    x = efficientnet_model.layers[40](x)
    x = efficientnet_model.layers[41](x)
    x = efficientnet_model.layers[42](x)
    x = efficientnet_model.layers[43](x)
    
    x3 = efficientnet_model.layers[44](x)
    #x = efficientnet_model.layers[45](x3)
    #x = efficientnet_model.layers[46](x)
    #x = efficientnet_model.layers[47](x)
    #x4 = efficientnet_model.layers[48](x)
    #x = tf.keras.layers.Multiply()([x3, x4])
    
    x = efficientnet_model.layers[50](x3)
    x = efficientnet_model.layers[51](x)

    x6 = efficientnet_model.layers[52](x)
    x9 = tf.keras.layers.Add()([x5, x6])
    
    x = efficientnet_model.layers[54](x)
    #x = efficientnet_model.layers[54](x9)
    x = efficientnet_model.layers[55](x)
    x = efficientnet_model.layers[56](x)
    x = efficientnet_model.layers[57](x)
    x = efficientnet_model.layers[58](x)
    
    x7 = efficientnet_model.layers[59](x)
    #x = efficientnet_model.layers[60](x7)
    #x = efficientnet_model.layers[61](x)
    #x = efficientnet_model.layers[62](x)
    #x8 = efficientnet_model.layers[63](x)
    #x = tf.keras.layers.Multiply()([x7, x8])
    
    x = efficientnet_model.layers[65](x7)
    x = efficientnet_model.layers[66](x)
    x10 = efficientnet_model.layers[67](x)
    x = tf.keras.layers.Add()([x9, x10])
    
    x = efficientnet_model.layers[69](x)
    x = efficientnet_model.layers[70](x)
    x = efficientnet_model.layers[71](x)
    x = layers.Conv2DTranspose(144, kernel_size=(1,1), strides=(1,1), padding="same", use_bias=False)(x)
    
    #model = keras.models.Model(inputs, x, name=name)
    """
    x=efficientnet_model.layers[29](x)
    for layer in efficientnet_model.layers[30:60]:
        x=layer(x)
    """
    return x

def efficientnet_generator(
    filters = 64,
    name=None,
):

    inputs = tf.keras.layers.Input(shape=[256,256,3])

    x = ReflectionPadding2D(padding=(3, 3))(inputs)
    x = layers.Conv2D(filters, (7, 7), use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Activation("relu")(x)

    #downsample
    x = layers.Conv2D(64, kernel_size=(3,3), strides=(2,2), padding="same", use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(144, kernel_size=(3,3), strides=(2,2), padding="same", use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Activation("relu")(x)

    #efficient blocks

    #for _ in range(num_efficient_blocks):
    x = efficient_block(x, 'block1')
    x = efficient_block(x, 'block2')
    x = efficient_block(x, 'block3')
    x = efficient_block(x, 'block4')
    x = efficient_block(x, 'block5')
    #x = efficient_block(x, 'block6')
    #x = efficient_block(x, 'block7')
    #x = efficient_block(x, 'block8')
    #x = efficient_block(x, 'block9')


    # Upsampling
    x = layers.Conv2DTranspose(64, kernel_size=(3,3), strides=(2,2), padding="same", use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2DTranspose(3, kernel_size=(3,3), strides=(2,2), padding="same", use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Activation("relu")(x)

    # Final block
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(3, (7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(inputs, x, name=name)
    
    model.compile()
    model.summary()
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
    x = layers.Conv2D(64, kernel_size=(3,3), strides=(2,2), padding="same", use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, kernel_size=(3,3), strides=(2,2), padding="same", use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Activation("relu")(x)

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"))

    # Upsampling
    x = layers.Conv2DTranspose(64, kernel_size=(3,3), strides=(2,2), padding="same", use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2DTranspose(3, kernel_size=(3,3), strides=(2,2), padding="same", use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Activation("relu")(x)

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

  