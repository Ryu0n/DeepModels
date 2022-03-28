from src.datasets.datasetloader import DatasetLoader
from src.models.original.alexnet_tensorflow import AlexNet
from src.models.refactor.alexnet_tensorflow_refactored import AlexNetRefactored, AlexNetRefactoredLight


class ModelTrainer:
    models = {'AlexNetRefactoredLight': AlexNetRefactoredLight}

    @staticmethod
    def _compile_model(model_name, optimizer='adam', loss='categorical_crossentropy'):
        model = ModelTrainer.models.get(model_name)()
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        return model

    def train(self, model_name: str, dataset_name: str, valid_ratio=0.33, random_state=42, batch_size=1, epochs=1):
        """
        ease to train models.
        :param model_name: model name that has to train
        models = {'AlexNetRefactoredLight': AlexNetRefactoredLight}

        :param dataset_name: dataset name
        dataset = {'mnist': mnist,
           'fashion_mnist': fashion_mnist,
           'cifar10': cifar10,
           'cifar100': cifar100}

        :param valid_ratio: ratio between train and test datasets

        :param random_state:
        :param batch_size:
        :param epochs:
        :return:
        """
        model = self._compile_model(model_name)
        train_ds, valid_ds, test_ds = DatasetLoader.load_dataset(dataset_name, batch_size, valid_ratio, random_state)
        history = model.fit(train_ds, batch_size=batch_size, epochs=epochs, validation_data=valid_ds)
        print(model.summary())
        print(model.evaluate(test_ds))
        return model, history
