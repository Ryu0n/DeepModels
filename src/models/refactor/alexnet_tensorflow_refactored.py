import tensorflow.keras as keras


class ConvolutionalBlock(keras.layers.Layer):
    def __init__(self, filters: int, kernel_length: int, stride: int, max_pool_length=None, max_pool_stride=None, padding='valid', **kwargs):
        super().__init__(**kwargs)
        self.block = [
            keras.layers.Conv2D(filters=filters,
                                kernel_size=(kernel_length, kernel_length),
                                strides=(stride, stride),
                                activation='relu',
                                padding=padding),
            keras.layers.BatchNormalization()
        ]
        assert [max_pool_length, max_pool_stride].count(None) % 2 != 1, 'max pool parameter is empty.'
        if max_pool_length:
            self.block.append(keras.layers.MaxPool2D(pool_size=(max_pool_length, max_pool_length),
                                                     strides=(max_pool_stride, max_pool_stride)))

    def call(self, inputs, **kwargs):
        for layer in self.block:
            inputs = layer(inputs)
        return inputs


class AlexNetRefactored(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [
            ConvolutionalBlock(96, 11, 4, 3, 2),
            ConvolutionalBlock(256, 5, 1, 3, 2, 'same'),
            ConvolutionalBlock(384, 3, 1, padding='same'),
            ConvolutionalBlock(384, 3, 1, padding='same'),
            ConvolutionalBlock(256, 3, 1, 3, 2, 'same'),
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='sigmoid')
        ]

    def call(self, inputs, training=None, mask=None):
        for block in self.blocks:
            inputs = block(inputs)
        return inputs


class AlexNetRefactoredLight(keras.Model):
    """
    for 10 categories classification
    e.g. cifar10, mnist, fashion_mnist
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [
            ConvolutionalBlock(48, 3, 1, 2, 2),
            ConvolutionalBlock(128, 3, 1, 2, 2, 'same'),
            ConvolutionalBlock(192, 3, 1, padding='same'),
            ConvolutionalBlock(192, 3, 1, padding='same'),
            ConvolutionalBlock(128, 3, 1, 2, 2, 'same'),
            keras.layers.Flatten(),
            keras.layers.Dense(2048, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(2048, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='sigmoid')
        ]

    def call(self, inputs, training=None, mask=None):
        for block in self.blocks:
            inputs = block(inputs)
        return inputs
