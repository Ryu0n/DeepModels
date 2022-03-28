import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100


class DatasetLoader:
    dataset = {'mnist': mnist,
               'fashion_mnist': fashion_mnist,
               'cifar10': cifar10,
               'cifar100': cifar100}

    @staticmethod
    def _split_dataset(x_train, y_train, x_test, y_test, valid_ratio=0.33, random_state=42):
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=valid_ratio, random_state=random_state)
        return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

    @staticmethod
    def _cast(x_train: np.ndarray, x_test: np.ndarray):
        return map(lambda array: array.astype(np.float32), [x_train, x_test])

    @staticmethod
    def _unsqueeze(x_train: np.ndarray, x_test: np.ndarray):
        return map(lambda array: array[..., np.newaxis] if len(array.shape) else array, [x_train, x_test])

    @staticmethod
    def _to_dataset(xy: tuple, batch_size):
        return tf.data.Dataset.from_tensor_slices(xy).shuffle(1000).batch(batch_size)

    @staticmethod
    def load_dataset(dataset_name: str, batch_size: int, valid_ratio=0.33, random_state=42):
        """

        :param dataset_name:
        :param batch_size:
        :param valid_ratio:
        :param random_state:
        :return: train_ds, valid_ds, test_ds
        """
        (x_train, y_train), (x_test, y_test) = DatasetLoader.dataset.get(dataset_name).load_data()
        x_train, x_test = DatasetLoader._cast(x_train, x_test)
        x_train, x_test = DatasetLoader._unsqueeze(x_train, x_test)
        y_train, y_test = map(to_categorical, (y_train, y_test))
        train, valid, test = DatasetLoader._split_dataset(x_train, y_train, x_test, y_test, valid_ratio, random_state)
        return map(lambda t: DatasetLoader._to_dataset(t, batch_size), [train, valid, test])
