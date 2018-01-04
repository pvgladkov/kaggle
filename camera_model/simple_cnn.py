# coding: utf-8

import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from random import shuffle
from datetime import datetime

from keras.callbacks import (ModelCheckpoint, EarlyStopping)
from PIL import Image

from settings import train_path, test_path, submissions_path, weights_path
from models import CNN1
from kaggle.utils import create_logger

logger = create_logger('simple_cnn')


def read_and_resize_job(q):
    while True:
        f_path = q.get()
        if f_path is None:
            break
        images[f_path] = read_and_resize(f_path)


def read_and_resize(f_path):
    im_array = np.array(Image.open(f_path), dtype="uint8")
    pil_im = Image.fromarray(im_array)
    new_array = np.array(pil_im.resize((256, 256)))
    return new_array / 255.0


def label_transform(labels):
    labels = pd.get_dummies(pd.Series(labels))
    index = labels.columns.values

    return labels, index


def train_df(use_crop=False):
    cameras = os.listdir(train_path)

    train_images = []
    for camera in cameras:
        for _fname in sorted(os.listdir(train_path + '/' + camera)):
            if not use_crop and 'crop' in _fname:
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

    train = train_df(use_crop=False)
    logger.info('train.shape {}'.format(train.shape))

    test = test_df()
    logger.info('test.shape {}'.format(test.shape))

    logger.info('read images')
    n_core = 20
    images = mp.Manager().dict()

    q = mp.Queue(maxsize=n_core)
    iolock = mp.Lock()
    pool = mp.Pool(n_core, initializer=read_and_resize_job, initargs=(q,))

    for i in train['path'].values:
        q.put(i)
    for i in test['path'].values:
        q.put(i)

    for _ in range(n_core):
        q.put(None)
    pool.close()
    pool.join()

    images = dict(images)

    X_train = np.array([images[p] for p in train['path'].values])
    X_test = np.array([images[p] for p in test['path'].values])

    logger.info('X_train.shape {}'.format(X_train.shape))
    logger.info('X_test.shape {}'.format(X_test.shape))

    y, label_index = label_transform(train['camera'].values)
    y = np.array(y)
    logger.info('y.shape {}'.format(y.shape))

    logger.info('train model')
    model = CNN1.model(len(label_index))
    file_path = "{}/{}.best.hdf5".format(weights_path, CNN1.name)

    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=1)
    callbacks_list = [checkpoint, early]

    history = model.fit(X_train, y, validation_split=0.1, epochs=20, shuffle=True, verbose=2,
                        callbacks=callbacks_list)

    model.load_weights(file_path)

    logger.info('make prediction')
    predicts = model.predict(X_test)
    predicts = np.argmax(predicts, axis=1)
    predicts = [label_index[p] for p in predicts]

    df = pd.DataFrame(columns=['fname', 'camera'])
    df['fname'] = test['fname'].values
    df['camera'] = predicts

    now = datetime.now()
    s_name = '{}/submission-{}.csv'.format(submissions_path, str(now.strftime("%Y%m%d-%H%M")))
    df.to_csv(s_name, index=False)
