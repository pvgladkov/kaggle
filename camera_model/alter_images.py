# coding: utf-8

import os
from multiprocessing import Pool
from kaggle.utils import create_logger
import pandas as pd
from PIL import Image
from settings import train_path, features_path
import random
from camera_model.utils.image_utils import transform_im

logger = create_logger('alter_images')


def make_alter(path):

    logger.info(path)
    p = []
    _as = ['resize05', 'resize08', 'resize15', 'resize20', 'gamma08', 'gamma12', 'q70', 'q90']
    rotates = [90, 180, 270, 0]
    for a_type in _as:
        for crop_type in [0, 1]:
            angle = random.choice(rotates)
            p.append(transform(path, crop_type, a_type, angle))
    return path, p


def transform(path, crop_type=None, alter_type=None, rotate_angle=None):

    new_path = path + '_crop{}_{}_r{}.jpg'.format(crop_type, alter_type, rotate_angle)
    if os.path.exists(new_path):
        return new_path

    img = Image.open(path)
    img = transform_im(img, crop_type, alter_type, rotate_angle)
    img.save(new_path)
    return new_path


def alter(rows, func):
    imf_d = {}
    p = Pool(20)
    ret = p.map(func, rows)
    for i in range(len(ret)):
        imf_d[ret[i][0]] = ret[i][1]

    fdata = [imf_d[f] for f in rows]
    return fdata


VERSION = 1

if __name__ == '__main__':

    logger.info('start')

    train = pd.read_csv('{}/train_pca_features_v{}.csv'.format(features_path, 1))
    train['path'] = train.apply(lambda x: '{}/{}/{}'.format(train_path, x['camera'], x['fname']), axis=1)

    logger.info('train.shape {}'.format(train.shape))
    logger.info('train make alter')

    alter(train['path'].values, make_alter)
    logger.info('finish')