import numpy as np
from keras.datasets import mnist

from base.base_data_loader import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    def __init__(self, config):
        BaseDataLoader.__init__(self, config)
        train_data, test_data = mnist.load_data()
        self.train_data = train_data
        self.test_data = test_data
        self.train_data_size = len(train_data[0])
        self.test_data_size = len(test_data[0])

    def get_train_data(self):
        images = []
        while True:
            for (image, label) in zip(self.train_data[0], self.train_data[1]):
                images.append(image)
                if len(images) >= self.config.trainer.batch_size:
                    data = np.array(images) / 127.5 - 1
                    data = np.expand_dims(data, -1)
                    images.clear()
                    yield data

    def get_test_data(self):
        images = []
        while True:
            for (image, label) in zip(self.test_data[0], self.test_data[1]):
                images.append(image)
                if len(images) >= self.config.trainer.batch_size:
                    data = np.array(images) / 127.5 - 1
                    data = np.expand_dims(data, -1)
                    images.clear()
                    yield data

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
