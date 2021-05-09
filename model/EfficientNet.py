import os
import random
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
from keras.layers import Activation
import efficientnet.tfkeras
import efficientnet.keras as efn 

from keras.utils.vis_utils import plot_model



class ReflectionPad2D(layers.Layer):
    """Implements Reflection Padding as a layer. Keras does not have a function that achieves this on its own so is 
        implemented through a class that can be called as a layer. Reflection padding uses the contents of the image
        matrices for the padding values. It 'reflects' the row into the padding and therefore ensures that the outputs
        will transition smoothly into the padding rather than using values that may create a harsher edge. Inspired by
        keras solution.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPad2D, self).__init__(**kwargs)

    def call(self, matrix_in):
        w, h = self.padding
        additional_padding = [[0, 0], [h, h], [w, w], [0, 0]]
        return tf.pad(matrix_in, additional_padding, mode="REFLECT")

    
def resnet_block(input_layer):
    """Residual block created for a baseline test using the guidance from the 'Unpaired Image-to-image Translation using Cycle-
        consistent Adversarial Networks' paper."""

    features = input_layer.shape[-1]
    func_input = input_layer

    res_layer = ReflectionPad2D()(input_layer)
    res_layer = layers.Conv2D(features, kernel_size=(3, 3), strides=(1, 1), padding="valid", use_bias=False)(res_layer)
    res_layer = tfa.layers.InstanceNormalization()(res_layer)
    res_layer = layers.Activation("relu")(res_layer)
    res_layer = ReflectionPad2D()(res_layer)      
    res_layer = layers.Conv2D(features, kernel_size=(3, 3), strides=(1, 1), padding="valid", use_bias=False)(res_layer)
    res_layer = tfa.layers.InstanceNormalization()(res_layer)
    output = layers.add([input_layer, res_layer])
    
    return output




def complete_efficientnet_generator():
    model = efn.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(256, 256, 3))

    """Complete EfficientNet model with upsampling to return dimensions to 256,256,3"""

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
    x = tfa.layers.InstanceNormalization()(x)
    
    # define new model
    model = Model(inputs=model.inputs, outputs=x)

    model.compile()
    model.summary()
    return model


def efficient_block_v1(input, name):
    """EfficientNet taking layers 30-71 (all 64x64) and manually including the multiply and add layers"""

    efficientnet_model = efn.EfficientNetB3(
            include_top=False,
            weights="imagenet",
            input_shape=(256, 256, 3)
            )
    
    for layer in efficientnet_model.layers:
        layer._name = layer.name + str(name)

    x = efficientnet_model.layers[30](input)
   
    x1 = efficientnet_model.layers[31](x)
    x = efficientnet_model.layers[32](x1)
    x = efficientnet_model.layers[33](x)
    x = efficientnet_model.layers[34](x)
    x2 = efficientnet_model.layers[35](x)
    x = tf.keras.layers.Multiply()([x1, x2])

    x = efficientnet_model.layers[37](x)
    x5 = efficientnet_model.layers[38](x)
    x = efficientnet_model.layers[39](x5)
    x = efficientnet_model.layers[40](x)
    x = efficientnet_model.layers[41](x)
    x = efficientnet_model.layers[42](x)
    x = efficientnet_model.layers[43](x)
    
    x3 = efficientnet_model.layers[44](x)
    x = efficientnet_model.layers[45](x3)
    x = efficientnet_model.layers[46](x)
    x = efficientnet_model.layers[47](x)
    x4 = efficientnet_model.layers[48](x)
    x = tf.keras.layers.Multiply()([x3, x4])
    
    x = efficientnet_model.layers[50](x)
    x = efficientnet_model.layers[51](x)

    x6 = efficientnet_model.layers[52](x)
    x9 = tf.keras.layers.Add()([x5, x6])
    
    x = efficientnet_model.layers[54](x9)
    x = efficientnet_model.layers[55](x)
    x = efficientnet_model.layers[56](x)
    x = efficientnet_model.layers[57](x)
    x = efficientnet_model.layers[58](x)
    
    x7 = efficientnet_model.layers[59](x)
    x = efficientnet_model.layers[60](x7)
    x = efficientnet_model.layers[61](x)
    x = efficientnet_model.layers[62](x)
    x8 = efficientnet_model.layers[63](x)
    x = tf.keras.layers.Multiply()([x7, x8])
    
    x = efficientnet_model.layers[65](x)
    x = efficientnet_model.layers[66](x)
    x10 = efficientnet_model.layers[67](x)
    x = tf.keras.layers.Add()([x9, x10])
    
    x = efficientnet_model.layers[69](x)
    x = efficientnet_model.layers[70](x)
    x = efficientnet_model.layers[71](x)
    x = layers.Conv2DTranspose(144, kernel_size=(1,1), strides=(1,1), padding="same", use_bias=False)(x)
    
    return x


def efficient_block_v2(input, name):
    """EfficientNet layers 30-71 (all 64x64) without using the multiply and add layers"""

    efficientnet_model = efn.EfficientNetB3(
            include_top=False,
            weights="imagenet",
            input_shape=(256, 256, 3)
            )
    
    for layer in efficientnet_model.layers:
        layer._name = layer.name + str(name)
        

    x = efficientnet_model.layers[30](input)
    x = efficientnet_model.layers[31](x)

    x = efficientnet_model.layers[37](x)
    x1 = efficientnet_model.layers[38](x)
    x = efficientnet_model.layers[39](x1)
    x = efficientnet_model.layers[40](x)
    x = efficientnet_model.layers[41](x)
    x = efficientnet_model.layers[42](x)
    x = efficientnet_model.layers[43](x)
    x = efficientnet_model.layers[44](x)  

    x = efficientnet_model.layers[50](x)
    x = layers.Dropout(0.5)(x) #try 0.5
    x = efficientnet_model.layers[51](x)
    x2 = efficientnet_model.layers[52](x)
    x3 = tf.keras.layers.Add()([x1, x2]) #[53]
    
    x = efficientnet_model.layers[54](x3)
 
    x = efficientnet_model.layers[55](x)
    x = efficientnet_model.layers[56](x)
    x = efficientnet_model.layers[57](x)
    x = efficientnet_model.layers[58](x)
    x = efficientnet_model.layers[59](x)

    x = efficientnet_model.layers[65](x)
    #x = layers.Dropout(0.3)(x)
    x = efficientnet_model.layers[66](x)
    x4 = efficientnet_model.layers[67](x)
    x = tf.keras.layers.Add()([x3, x4]) #[68]
    x = efficientnet_model.layers[69](x)
    x = efficientnet_model.layers[70](x)
    x = efficientnet_model.layers[71](x)
    x = layers.Conv2DTranspose(144, kernel_size=(1,1), strides=(1,1), padding="same", use_bias=False)(x)
    
    return x


def split_model(net_in, first_layer, last_layer, name):
    """Splits the EfficientNet model based on a range defined by the start and end parameters. Allows all the features of the model remain 
        while only keeping a small portion. The previous implementations relied on manually separating the layers, possibly reducing
        performance"""  

    l_used = set()
    structure = net_in.get_config()
    for i, layer in enumerate(structure['layers']):
        if i == 0:
            structure['layers'][0]['config']['batch_input_shape'] = net_in.layers[first_layer].input_shape
            if i != first_layer:
                structure['layers'][0]['name']
                structure['layers'][0]['config']['name'] = structure['layers'][0]['name']
        elif i < first_layer:
            continue
        elif i > last_layer:
            continue
        l_used.add(layer['name'])

    
    # extract layers to be used
    l_extracted=[]
    for l in structure['layers']:
        if l['name'] in l_used:
            l_extracted.append(l)

    l_extracted[1]['inbound_nodes'][0][0][0] = l_extracted[0]['name']

    # set model structure
    structure['layers'] = l_extracted
    structure['input_layers'][0][0] = l_extracted[0]['name']
    structure['output_layers'][0][0] = l_extracted[-1]['name']

    # create new model
    modelsplit = Model.from_config(structure)
        
    for layer in modelsplit.layers:
        layer._name = layer.name + str(name)

    modelsplit._name = 'EfficientNet_' + str(name)
    return modelsplit


def efficient_block_v3(inputs, name):
    """The efficient block used for the split function. Takes a layer as an input and returns a usable block of efficientnet"""

    model = efn.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(256, 256, 3))
    
    block = split_model(model, 30, 71, name)(inputs)
    x = tf.keras.layers.Dense(144)
    x = layers.Conv2D(144, kernel_size=(1,1), strides=(1,1), padding="same", use_bias=False)(block)

    return x


def efficientnet_generator(filters=64, name=None):
    """Full EfficientNet model used for image generation. Downsamples the 256,256,3 input image into 64x64x144. Run through the specified
        number of EfficientNet blocks, and then upsampled back to 256,256,3"""

    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    #downsample
    inputs = tf.keras.layers.Input(shape=[256,256,3])
    x = ReflectionPad2D(padding=(3, 3))(inputs)
    x = layers.Conv2D(filters, kernel_size=(7, 7))(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, kernel_size=(3,3),strides=(2,2), padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(144, kernel_size=(3,3), strides=(2,2), padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Activation("relu")(x)

    eff_1 = efficient_block_v3(x, 'block1')
    eff_2 = efficient_block_v3(eff_1, 'block2')
    eff_3 = efficient_block_v3(eff_2, 'block3')
    eff_4 = efficient_block_v3(eff_3, 'block4')
    eff_5 = efficient_block_v3(eff_4, 'block5')
    #x = efficient_block_v1(x, 'block6')
    #x = efficient_block_v1(x, 'block7')
    #x = efficient_block_v1(x, 'block8')
    #x = efficient_block_v1(x, 'block9')

    # Upsampling
    x = layers.Conv2DTranspose(64,kernel_size=(3,3),strides=(2,2),padding='same')(eff_5)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2DTranspose(32,kernel_size=(3,3),strides=(2,2),padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Activation("relu")(x)

    x = ReflectionPad2D(padding=(3, 3))(x)
    x = layers.Conv2D(3, kernel_size=(7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)


    model = keras.models.Model(inputs, x, name=name)
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    model.compile()
    model.summary()
    return model


def resnet_generator(
    filters=64,
    num_resnet_blocks=5):
    """Standard ResNet implementation that is used in the CycleGAN paper for a baseline test"""

    inputs = tf.keras.layers.Input(shape=[256,256,3])
    x = ReflectionPad2D(padding=(3, 3))(inputs)
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
    for i in range(num_resnet_blocks):
        x = resnet_block(x)

    # Upsampling
    x = layers.Conv2DTranspose(64, kernel_size=(3,3), strides=(2,2), padding="same", use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2DTranspose(3, kernel_size=(3,3), strides=(2,2), padding="same", use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Activation("relu")(x)

    # Final block
    x = ReflectionPad2D(padding=(3, 3))(x)
    x = layers.Conv2D(3, (7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)
    

    model = keras.models.Model(inputs, x)
    
    model.compile()
    model.summary()
    return model


def discriminator(
    filters=64,  
    name=None):
    """Discriminator network based on the CycleGAN paper. Simply an image classifier that downsamples an image input
    and returns a decision"""
    
    model_input = tf.keras.layers.Input(shape=[256,256,3])
    x = layers.Conv2D(filters, kernel_size=(4, 4),strides=(2, 2),padding="same")(model_input)
    x = layers.LeakyReLU(0.2)(x)

    for i in range(3):
        filters = filters * 2
        if i < 2:
            x = layers.Conv2D(filters, (4, 4), strides=(2, 2))(x)
            x = tfa.layers.InstanceNormalization()(x)
            x = layers.LeakyReLU(0.2)(x)
        else:
            x = layers.Conv2D(filters, (4, 4), strides=(1, 1))(x)
            x = tfa.layers.InstanceNormalization()(x)
            x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(1, (4, 4), strides=(1, 1), padding="same")(x)

    model = keras.models.Model(inputs=model_input, outputs=x, name=name)
    model.compile()
    model.summary()
    return model


