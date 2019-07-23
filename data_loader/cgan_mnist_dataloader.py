import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from base.base_data_loader import BaseDataLoader


class CGanMnistDataLoader(BaseDataLoader):
    def __init__(self, config):
        BaseDataLoader.__init__(self, config)
        train_data, test_data = mnist.load_data()
        X_data, labels = train_data
        labels = to_categorical(labels, num_classes=10, dtype=np.float64)
        self.train_data = (X_data, labels)
        self.test_data = test_data
        self.train_data_size = len(train_data[0])
        self.test_data_size = len(test_data[0])

    def get_train_data(self):
        images = []
        labels = []
        while True:
            for (image, label) in zip(self.train_data[0], self.train_data[1]):
                images.append(image)
                labels.append(label)
                if len(images) >= self.config.trainer.batch_size:
                    x_data = np.array(images) / 127.5 - 1
                    x_data = np.expand_dims(x_data, -1)
                    y_data = np.array(labels, dtype=np.float64)
                    images.clear()
                    labels.clear()
                    yield x_data, y_data

    def get_test_data(self):
        images = []
        labels = []
        while True:
            for (image, label) in zip(self.test_data[0], self.test_data[1]):
                images.append(image)
                labels.append(label)
                if len(images) >= self.config.trainer.batch_size:
                    x_data = np.array(images) / 127.5 - 1
                    x_data = np.expand_dims(x_data, -1)
                    y_data = np.array(labels)
                    images.clear()
                    yield x_data, y_data

    def get_train_data_generator(self):
        return self.get_train_data()

    def get_validation_data_generator(self):
        raise NotImplementedError

    def get_test_data_generator(self):
        return self.get_test_data()

    def get_train_data_size(self):
        return self.train_data_size

    def get_validation_data_size(self):
        raise NotImplementedError

    def get_test_data_size(self):
        return self.test_data_size
