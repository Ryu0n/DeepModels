import tensorflow as tf
import tensorflow.keras as keras


class AlexNet(keras.models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.i = keras.layers.InputLayer((None, 227, 227, 3))

        # First Layer
        self.conv1_1 = keras.layers.Conv2D(filters=48, kernel_size=11, strides=4, activation='relu')
        self.maxpool1_1 = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))

        self.conv1_2 = keras.layers.Conv2D(filters=48, kernel_size=11, strides=4, activation='relu')
        self.maxpool1_2 = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))

        # Second Layer
        self.conv2_1 = keras.layers.Conv2D(filters=128, kernel_size=5, padding='same', activation='relu')
        self.maxpool2_1 = keras.layers.MaxPool2D()

        self.conv2_2 = keras.layers.Conv2D(filters=128, kernel_size=5, padding='same', activation='relu')
        self.maxpool2_2 = keras.layers.MaxPool2D()

        # Third Layer
        self.conv3_1 = keras.layers.Conv2D(filters=192, kernel_size=3, padding='same', activation='relu')
        self.conv3_2 = keras.layers.Conv2D(filters=192, kernel_size=3, padding='same', activation='relu')

        # Fourth Layer
        self.conv4_1 = keras.layers.Conv2D(filters=192, kernel_size=3, padding='same', activation='relu')
        self.conv4_2 = keras.layers.Conv2D(filters=192, kernel_size=3, padding='same', activation='relu')

        # Fifth Layer
        self.conv5_1 = keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')
        self.maxpool5_1 = keras.layers.MaxPool2D()

        self.conv5_2 = keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')
        self.maxpool5_2 = keras.layers.MaxPool2D()

        # Sixth Layer
        self.dense6_1 = keras.layers.Dense(2048)
        self.dense6_2 = keras.layers.Dense(2048)
        self.flatten = keras.layers.Flatten()

        # Seventh Layer
        self.dense7_1 = keras.layers.Dense(2048)
        self.dense7_2 = keras.layers.Dense(2048)

        # Eighth Layer
        self.dense8 = keras.layers.Dense(1000, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        inputs = self.i(inputs)

        outputs1 = self.conv1_1(inputs)
        outputs1 = self.maxpool1_1(outputs1)
        outputs2 = self.conv1_1(inputs)
        outputs2 = self.maxpool1_1(outputs2)

        outputs1 = self.conv2_1(outputs1)
        outputs1 = self.maxpool2_1(outputs1)
        outputs2 = self.conv2_2(outputs2)
        outputs2 = self.maxpool2_2(outputs2)

        outputs1_ = tf.concat([outputs1, outputs2], axis=-1)
        outputs2_ = tf.concat([outputs2, outputs1], axis=-1)

        outputs1 = self.conv3_1(outputs1_)
        outputs2 = self.conv3_2(outputs2_)

        outputs1 = self.conv4_1(outputs1)
        outputs2 = self.conv4_2(outputs2)

        outputs1 = self.conv5_1(outputs1)
        outputs1 = self.maxpool5_1(outputs1)
        outputs2 = self.conv5_2(outputs2)
        outputs2 = self.maxpool5_2(outputs2)

        outputs1_ = self.flatten(tf.concat([outputs1, outputs2], axis=-1))
        outputs2_ = self.flatten(tf.concat([outputs2, outputs1], axis=-1))

        outputs1 = self.dense6_1(outputs1_)
        outputs2 = self.dense6_2(outputs2_)

        outputs1_ = tf.concat([outputs1, outputs2], axis=-1)
        outputs2_ = tf.concat([outputs2, outputs1], axis=-1)

        outputs1 = self.dense7_1(outputs1_)
        outputs2 = self.dense7_1(outputs2_)

        output_ = tf.concat([outputs1, outputs2], axis=-1)

        output = self.dense8(output_)

        return output
