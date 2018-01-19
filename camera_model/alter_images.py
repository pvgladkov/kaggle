# coding: utf-8

import os
from multiprocessing import Pool
from kaggle.utils import create_logger
import pandas as pd
from PIL import Image
from settings import train_path, features_path
import random
from utils.image_utils import transform_im, crop

logger = create_logger('alter_images')


def make_alter(path):

    logger.info(path)
    p = []
    _as = ['resize05', 'resize08', 'resize15', 'resize20', 'gamma08', 'gamma12', 'q70', 'q90']
    p.append(transform(path))
    return path, p


def transform(path, crop_type=None, alter_type=None, rotate_angle=None):

    new_path = path + '_crop0.jpg'
    if os.path.exists(new_path):
        return new_path

    img = Image.open(path)
    img = crop(img, 0, 512)
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