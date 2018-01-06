import math

import numpy as np
from PIL import Image
from keras.utils import Sequence

from image_utils import transform_im, read_and_resize, resize_shape, crop


class TrainFileSequence(Sequence):

    def __init__(self, paths, y_train, batch_size):
        self.x, self.y = paths, y_train
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(1.0 * len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([read_and_resize(p) for p in batch_x]), np.array(batch_y)


class TrainFileSequenceOnFly(Sequence):

    def __init__(self, paths, y_train, batch_size):
        self.x, self.y = paths, y_train
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(1.0 * len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = []
        y = []
        for _p, _y in zip(batch_x, batch_y):
            img = Image.open(_p)
            _as = ['resize05', 'resize08', 'resize15', 'resize20', 'gamma08', 'gamma12', 'q70', 'q90']
            rotates = [90, 180, 270, 0]
            for a_type in _as:
                for crop_type in [0, 1]:
                    for angle in rotates:
                        image_copy = img.copy()
                        image_copy = transform_im(image_copy, crop_type, a_type, angle)
                        image_copy = np.array(image_copy.resize((256, 256)))
                        image_copy = image_copy / 255.0
                        X.append(image_copy)
                        y.append(_y)

        return np.array(X), np.array(y)


class PredictFileSequence(Sequence):
    def __init__(self, paths, batch_size):
        self.x = paths
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(1.0 * len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([read_and_resize(p) for p in batch_x])


class DataSequence(Sequence):

    def __init__(self, image_data, batch_size):
        self.x = image_data
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(1.0 * len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        raise NotImplementedError()

    @staticmethod
    def x_to_array(batch_x):
        return np.array([np.array(resize_shape(p, 256, 256)) / 255.0 for p in batch_x])


class TrainDataSequence(DataSequence):

    def __init__(self, image_data, y_train, batch_size):
        super(TrainDataSequence, self).__init__(image_data, batch_size)
        self.y = y_train

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.x_to_array(batch_x), np.array(batch_y)


class PredictDataSequence(DataSequence):

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.x_to_array(batch_x)