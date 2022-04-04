from tensorflow import concat
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAvgPool2D


class InceptionModule(Layer):
    def __init__(self, f1, f2x1, f2x3, f3x1, f3x5, fm, **kwargs):
        super().__init__(**kwargs)

        self.conv_1 = [Conv2D(filters=f1, kernel_size=1, padding='same', activation='relu')]

        self.conv_3 = [
            Conv2D(filters=f2x1, kernel_size=1, padding='same', activation='relu'),
            Conv2D(filters=f2x3, kernel_size=3, padding='same', activation='relu')
        ]

        self.conv_5 = [
            Conv2D(filters=f3x1, kernel_size=1, padding='same', activation='relu'),
            Conv2D(filters=f3x5, kernel_size=5, padding='same', activation='relu')
        ]

        self.max_pool_3 = [
            MaxPool2D(pool_size=(3, 3), padding='same'),
            Conv2D(filters=fm, kernel_size=1, padding='same', activation='relu')
        ]

    @staticmethod
    def _pass_through(inputs, path):
        for p in path:
            inputs = p(inputs)
        return inputs

    def call(self, inputs, *args, **kwargs):
        i1, i3, i5, im = map(lambda path: self._pass_through(inputs, path), [self.conv_1, self.conv_3, self.conv_5, self.max_pool_3])
        o = concat([i1, i3, i5, im], axis=-1)
        return o


class AuxiliaryClassifier(Layer):
    def __init__(self, f1, fc1, fc2, **kwargs):
        super().__init__(**kwargs)
        self.layers = [
            AveragePooling2D(pool_size=(5, 5), padding='same'),
            Conv2D(filters=f1, kernel_size=1, padding='same', activation='relu'),
            Flatten(),
            Dense(fc1, activation='relu'),
            Dense(fc2, activation='softmax')
        ]

    def call(self, inputs, *args, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


class GoogLeNet(Model):
    pass
