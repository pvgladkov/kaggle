# coding: utf-8

import os
import numpy as np
import pandas as pd
from random import shuffle
from datetime import datetime

from keras.callbacks import (ModelCheckpoint, EarlyStopping)
from PIL import Image

from camera_model.data_utils import TrainSequence, PredictSequence
from settings import train_path, test_path, submissions_path, weights_path
from models import CNN1
from kaggle.utils import create_logger

logger = create_logger('simple_cnn')


def read_and_resize(f_path):
    im_array = np.array(Image.open(f_path), dtype="uint8")
    pil_im = Image.fromarray(im_array)
    new_array = np.array(pil_im.resize((256, 256)))
    return new_array / 255.0


def label_transform(labels):
    labels = pd.get_dummies(pd.Series(labels))
    index = labels.columns.values
    return labels, index


def train_df(use_crop=False, use_original=False):
    cameras = os.listdir(train_path)

    if not use_crop and not use_original:
        raise Exception('invalid train data')

    train_images = []
    for camera in cameras:
        for _fname in sorted(os.listdir(train_path + '/' + camera)):
            if not use_crop and 'crop' in _fname:
                continue

            if not use_original and 'crop' not in _fname:
                continue

            _path = '{}/{}/{}'.format(train_path, camera, _fname)
            train_images.append((camera, _fname, _path))

    shuffle(train_images)
    return pd.DataFrame(train_images, columns=['camera', 'fname', 'path'])


def test_df():
    test_images = []
    for _fname in sorted(os.listdir(test_path)):
        _path = '{}/{}'.format(test_path, _fname)
        test_images.append((_fname, _path))

    return pd.DataFrame(test_images, columns=['fname', 'path'])


if __name__ == '__main__':

    batch_size = 128

    train = train_df(use_crop=False, use_original=True)
    logger.info('train.shape {}'.format(train.shape))

    test = test_df()
    logger.info('test.shape {}'.format(test.shape))

    y, label_index = label_transform(train['camera'].values)
    y = np.array(y)
    logger.info('y.shape {}'.format(y.shape))

    logger.info('train model')

    model = CNN1.model(len(label_index), batch_size)
    file_path = "{}/{}.best.hdf5".format(weights_path, CNN1.name)
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=1)
    callbacks_list = [checkpoint, early]

    X_train = TrainSequence(train['path'].values, y, batch_size)

    history = model.fit_generator(X_train, epochs=20, verbose=2, use_multiprocessing=True,
                                  workers=20, callbacks=callbacks_list)

    model.load_weights(file_path)

    logger.info('make prediction')
    X_test = PredictSequence(test['path'].values, batch_size)

    predicts = model.predict_generator(X_test, use_multiprocessing=True, workers=20)
    predicts = np.argmax(predicts, axis=1)
    predicts = [label_index[p] for p in predicts]

    df = pd.DataFrame(columns=['fname', 'camera'])
    df['fname'] = test['fname'].values
    df['camera'] = predicts

    now = datetime.now()
    s_name = '{}/submission-{}-{}.csv'.format(submissions_path, CNN1.name, str(now.strftime("%Y%m%d-%H%M")))
    df.to_csv(s_name, index=False)
    logger.info('save to {}'.format(s_name))
