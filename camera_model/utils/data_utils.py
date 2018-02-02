import math

import numpy as np
import pandas as pd
from PIL import Image
from keras.utils import Sequence
import random
from sklearn.model_selection import train_test_split
from kaggle.utils import create_logger
from image_utils import transform_im, read_prediction_crop, read_and_crop, crop, demosaicing_error

logger = create_logger('data_utils')


def _augmentations(img, label, shape, validate=False, mosaic=False):
    _X = []
    _y = []
    _as1 = ['resize05', 'resize08', 'resize15', 'resize20', 'gamma08', 'gamma12', 'q70', 'q90']
    _as2 = ['gamma08', 'gamma12']
    _as3 = ['resize05', 'resize08', 'resize15', 'resize20']
    _as4 = ['q70', 'q90']
    _as5 = ['resize05', 'resize08', 'resize15', 'resize20', 'gamma08', 'gamma12']
    rotates = [90, 180, 270, 0]
    crops = [0, 1, 2, 3, 4]

    img = crop(img, 0, 512)

    if validate:
        __as = random.sample(_as1, 1) + [None, None]
    else:
        __as = random.sample(_as1, 1) + [None, None]

    for a_type in __as:
        for crop_type in random.sample(crops, 1):
            for angle in random.sample(rotates, 2):
                image_copy = img.copy()
                image_copy = transform_im(image_copy, crop_type, a_type, angle, shape)

                if mosaic:
                    error = ((demosaicing_error(image_copy) / 2.0) > 3).astype(np.int)
                    _X.append(error)
                else:
                    image_copy = np.array(image_copy)
                    image_copy = image_copy / 255.0
                    _X.append(image_copy)
                _y.append(label)
    return _X, _y


def _demosaicing_errors(img, label):
    error = ((demosaicing_error(img) / 2.0) > 3).astype(np.int)
    return [error], [label]


def image_augmentations(path, label, shape, validate=False):
    with Image.open(path) as img:
        try:
            _X, _y = _augmentations(img, label, shape, validate, True)
        except Exception as e:
            logger.info(path)
            _X, _y = None, None
    return _X, _y


def image_errors(path, label, shape, validate=False):
    with Image.open(path) as img:
        try:
            _X, _y = _demosaicing_errors(img, label)
        except Exception as e:
            _X, _y = None, None
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
        X = []
        y = []
        for _p, _y in zip(batch_x, batch_y):
            _X, _y = self.image_augmentations(_p, _y, self.shape)
            if _X is not None:
                X += _X
                y += _y

        return np.array(X), np.array(y)


class TrainFileSequenceOnFly(_TrainSequenceOnFly):

    def image_augmentations(self, x, y, shape):
        return image_augmentations(x, y, shape)


class ValidateFileSequenceOnFly(_TrainSequenceOnFly):

    def image_augmentations(self, x, y, shape):
        return image_augmentations(x, y, shape, validate=True)


class TrainDataSequenceOnFly(_TrainSequenceOnFly):

    def image_augmentations(self, x, y, shape):
        return _augmentations(x, y, shape)


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


class PredictValidateFileSequence(PredictFileSequence):

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([image_augmentations(p, 0, self.shape, True)[0][0] for p in batch_x])


def label_transform(labels):
    labels = pd.get_dummies(pd.Series(labels))
    index = labels.columns.values
    return labels, index


def train_val_gen(X, y, batch_size, shape):
    train_train, train_validate, y_train, y_validate = X[:-275], X[-275:], y[:-275], y[-275:]

    # assert train_train.shape == (9175, )
    # assert train_validate.shape == (275, )
    # assert y_train.shape == (9175, 10)
    # assert y_validate.shape == (275, 10)

    X_y_train = TrainFileSequenceOnFly(train_train, y_train, batch_size, shape)
    X_y_validate = ValidateFileSequenceOnFly(train_validate, y_validate, batch_size, shape)

    return X_y_train, X_y_validate
