"""
UNet architecture in Keras TensorFlow
"""
import os
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class Unet:
    def __init__(self, input_size=256):
        self.input_size = input_size

    def build_model(self):
        def conv_block(x, n_filter, pool=True):
            x = Conv2D(n_filter, (3, 3), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(n_filter, (3, 3), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            c = x

            if pool == True:
                x = MaxPooling2D((2, 2), (2, 2))(x)
                return c, x
            else:
                return c

        n_filters = [16, 32, 64, 128, 256]
        inputs = Input((self.input_size, self.input_size, 3))

        c0 = inputs
        ## Encoder
        c1, p1 = conv_block(c0, n_filters[0])
        c2, p2 = conv_block(p1, n_filters[1])
        c3, p3 = conv_block(p2, n_filters[2])
        c4, p4 = conv_block(p3, n_filters[3])

        ## Bridge
        b1 = conv_block(p4, n_filters[4], pool=False)
        b2 = conv_block(b1, n_filters[4], pool=False)

        ## Decoder
        d1 = Conv2DTranspose(n_filters[3], (3, 3), padding="same", strides=(2, 2))(b2)
        d1 = Concatenate()([d1, c4])
        d1 = conv_block(d1, n_filters[3], pool=False)

        d2 = Conv2DTranspose(n_filters[3], (3, 3), padding="same", strides=(2, 2))(d1)
        d2 = Concatenate()([d2, c3])
        d2 = conv_block(d2, n_filters[2], pool=False)

        d3 = Conv2DTranspose(n_filters[3], (3, 3), padding="same", strides=(2, 2))(d2)
        d3 = Concatenate()([d3, c2])
        d3 = conv_block(d3, n_filters[1], pool=False)

        d4 = Conv2DTranspose(n_filters[3], (3, 3), padding="same", strides=(2, 2))(d3)
        d4 = Concatenate()([d4, c1])
        d4 = conv_block(d4, n_filters[0], pool=False)

        ## output
        outputs = Conv2D(1, (1, 1), padding="same")(d4)
        outputs = BatchNormalization()(outputs)
        outputs = Activation("sigmoid")(outputs)

        ## Model
        model = Model(inputs, outputs)
        return model
