import math
import random

import numpy as np
from PIL import Image
from keras.utils import Sequence

from camera_model.utils.data_utils import logger, ALL_AUGMENTATIONS, CROPS, ROTATES
from camera_model.utils.image_utils import read_and_crop, read_prediction_crop, crop, transform_im, demosaicing_error


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
            _X, _y = _augmentations(img, label, shape, validate, True)
        except Exception as e:
            logger.info(path)
            _X, _y = None, None
    return _X, _y


def _augmentations(img, label, shape, validate=False, mosaic=False):
    _X = []
    _y = []

    img = crop(img, 0, 512)

    if validate:
        __as = random.sample(ALL_AUGMENTATIONS, 1) + [None, None]
    else:
        __as = random.sample(ALL_AUGMENTATIONS, 1) + [None, None]

    for a_type in __as:
        for crop_type in random.sample(CROPS, 2):
            for angle in [None]:
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