# coding: utf-8

import os
from sklearn.utils import shuffle
import pandas as pd
from PIL import Image
import numpy as np

from multiprocessing import Queue, Lock, Manager, Pool

from kaggle.utils import create_logger

logger = create_logger('simple_cnn')


def get_train_df(train_path, use_crop=False, use_original=False):
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

    shuffle(train_images, random_state=777)
    return pd.DataFrame(train_images, columns=['camera', 'fname', 'path'])


def get_test_df(test_path):
    test_images = []
    for _fname in sorted(os.listdir(test_path)):
        _path = '{}/{}'.format(test_path, _fname)
        test_images.append((_fname, _path))

    return pd.DataFrame(test_images, columns=['fname', 'path'])


def load_files(train_path, test_path):

    def read_f(q):
        while True:
            img_path = q.get()
            logger.info(img_path)
            if img_path is None:
                break
            img = Image.open(img_path)
            image_copy = np.array(img.resize((256, 256)))
            image_copy = image_copy / 255.0
            data[img_path] = image_copy

    train_df = get_train_df(train_path, use_original=True, use_crop=False)
    test_df = get_test_df(test_path)

    data = Manager().dict()
    n_core = 20
    q = Queue(maxsize=n_core)
    iolock = Lock()
    pool = Pool(n_core, initializer=read_f, initargs=(q,))

    for i in train_df['path'].values:
        q.put(i)

    for i in test_df['path'].values:
        q.put(i)

    for _ in range(n_core):
        q.put(None)
    pool.close()
    pool.join()

    data = dict(data)
    return data
