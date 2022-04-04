"""
reference from https://d2l.ai/chapter_convolutional-modern/nin.html
"""

import tensorflow.keras as keras


class MlpConvBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.block = [
            keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, activation='relu'),
            keras.layers.Conv2D(filters, 1, activation='relu'),
            keras.layers.Conv2D(filters, 1, activation='relu')
        ]

    def call(self, inputs, *args, **kwargs):
        for layer in self.block:
            inputs = layer(inputs)
        return inputs


class NIN(keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [
            MlpConvBlock(96, 11, 4),
            keras.layers.MaxPool2D(3, 2),
            MlpConvBlock(256, 5),
            keras.layers.MaxPool2D(3, 2),
            MlpConvBlock(10, 3),
            keras.layers.GlobalAvgPool2D(),
            keras.layers.Flatten()
        ]

    def call(self, inputs, training=None, mask=None):
        for block in self.blocks:
            inputs = block(inputs)
        return inputs
