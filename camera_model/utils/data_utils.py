import math

import numpy as np
import pandas as pd
from PIL import Image
from keras.utils import Sequence
import random
from sklearn.model_selection import train_test_split

from image_utils import transform_im, read_prediction_crop, read_and_crop


def image_augmentations(path, label, shape):
    _X = []
    _y = []
    with Image.open(path) as img:
        _as1 = ['resize05', 'resize08', 'resize15', 'resize20', 'gamma08', 'gamma12', 'q70', 'q90']
        _as2 = ['gamma08', 'gamma12', None]
        rotates = [90, 180, 270, 0]
        crops = [0, 1, 2, 3, 4]
        for a_type in random.sample(_as1, 1) + [None]:
            for crop_type in random.sample(crops, 1):
                for angle in random.sample(rotates, 1):
                    image_copy = img.copy()
                    image_copy = transform_im(image_copy, crop_type, a_type, angle, shape)
                    image_copy = np.array(image_copy)
                    image_copy = image_copy / 255.0
                    _X.append(image_copy)
                    _y.append(label)
    return _X, _y


class TrainFileSequence(Sequence):

    def __init__(self, paths, y_train, batch_size):
        self.x, self.y = paths, y_train
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(1.0 * len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([read_and_crop(p) for p in batch_x]), np.array(batch_y)


class TrainFileSequenceOnFly(Sequence):

    def __init__(self, paths, y_train, batch_size, shape):
        self.x, self.y = paths, y_train
        self.batch_size = batch_size
        self.shape = shape

    def __len__(self):
        return math.ceil(1.0 * len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = []
        y = []
        for _p, _y in zip(batch_x, batch_y):
            _X, _y = image_augmentations(_p, _y, self.shape)
            X += _X
            y += _y

        return np.array(X), np.array(y)


class PredictFileSequence(Sequence):
    def __init__(self, paths, batch_size, shape):
        self.x = paths
        self.batch_size = batch_size
        self.shape = shape

    def __len__(self):
        return math.ceil(1.0 * len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([read_prediction_crop(p, self.shape) for p in batch_x])


def label_transform(labels):
    labels = pd.get_dummies(pd.Series(labels))
    index = labels.columns.values
    return labels, index


def train_val_gen(X, y, batch_size, shape):
    train_train, train_validate, y_train, y_validate = train_test_split(X, y, test_size=0.1, random_state=777)

    X_y_train = TrainFileSequenceOnFly(train_train, y_train, batch_size, shape)
    X_y_validate = TrainFileSequenceOnFly(train_validate, y_validate, batch_size, shape)

    return X_y_train, X_y_validate
