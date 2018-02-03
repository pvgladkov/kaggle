import math
import random

import numpy as np
from PIL import Image
from keras.utils import Sequence

from camera_model.utils.data_utils import logger, ALL_AUGMENTATIONS, CROPS, ROTATES
from camera_model.utils.image_utils import read_prediction_crop, crop, transform_im


class _TrainSequenceOnFly(Sequence):

    def __init__(self, x_train, y_train, batch_size, shape):
        self.x, self.y = x_train, y_train
        self.batch_size = batch_size
        self.shape = shape

    def __len__(self):
        return math.ceil(1.0 * len(self.x) / self.batch_size)

    def image_augmentations(self, x, y, shape):
        raise NotImplementedError()

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        X1 = []
        X2 = []
        y = []
        for _p, _y in zip(batch_x, batch_y):
            _X1, _X2, _y = self.image_augmentations(_p, _y, self.shape)
            if _X1 is not None:
                X1 += _X1
                X2 += _X2
                y += _y

        return [np.array(X1), np.array(X2)], np.array(y)


class TrainFileSequenceOnFly(_TrainSequenceOnFly):

    def image_augmentations(self, x, y, shape):
        return image_augmentations(x, y, shape)


class ValidateFileSequenceOnFly(_TrainSequenceOnFly):

    def image_augmentations(self, x, y, shape):
        return image_augmentations(x, y, shape, validate=True)


class PredictFileSequence(Sequence):
    def __init__(self, paths, batch_size, shape):
        self.x = paths
        self.batch_size = batch_size
        self.shape = shape

    def __len__(self):
        return math.ceil(1.0 * len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

        X1 = []
        X2 = []

        for path in batch_x:
            X1.append(read_prediction_crop(path, self.shape))
            manipulated = np.float32([int('manip' in path)])
            X2.append(manipulated)

        return [np.array(X1), np.array(X2)]


def train_val_gen(X, y, batch_size, shape):
    train_train, train_validate, y_train, y_validate = X[:-275], X[-275:], y[:-275], y[-275:]

    assert train_validate.shape == (275, )
    assert y_validate.shape == (275, 10)

    X_y_train = TrainFileSequenceOnFly(train_train, y_train, batch_size, shape)
    X_y_validate = ValidateFileSequenceOnFly(train_validate, y_validate, batch_size, shape)

    return X_y_train, X_y_validate


def image_augmentations(path, label, shape, validate=False):
    with Image.open(path) as img:
        try:
            _X1, _X2, _y = _augmentations(img, label, shape, validate)
        except Exception as e:
            logger.info(path)
            _X1, _X2, _y = None, None, None
    return _X1, _X2, _y


def _augmentations(img, label, shape, validate=False):
    _X1 = []
    _X2 = []
    _y = []

    img = crop(img, 0, 512)

    if validate:
        __as = random.sample(ALL_AUGMENTATIONS, 1) + [None, None]
    else:
        __as = random.sample(ALL_AUGMENTATIONS, 1) + [None, None]

    for a_type in __as:

        if a_type is None:
            manipulated = np.float32([0])
        else:
            manipulated = np.float32([1])

        for crop_type in random.sample(CROPS, 1):
            for angle in random.sample(ROTATES, 2):
                image_copy = img.copy()
                image_copy = transform_im(image_copy, crop_type, a_type, angle, shape)
                image_copy = np.array(image_copy)
                image_copy = image_copy / 255.0
                _X1.append(image_copy)
                _X2.append(manipulated)
                _y.append(label)
    return _X1, _X2, _y
