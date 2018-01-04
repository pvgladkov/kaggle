# coding: utf-8

import os
from multiprocessing import Pool
from kaggle.utils import create_logger
import pandas as pd
from PIL import Image, ImageEnhance
from settings import train_path, features_path
import random
from skimage import exposure

logger = create_logger('alter_images')


def make_alter(path):

    logger.info(path)
    p = []
    _as = ['resize05', 'resize08', 'resize15', 'resize20', 'gamma08', 'gamma12', 'q70', 'q90']
    rotates = [90, 180, 270, 0]
    for a_type in _as:
        for crop_type in [0, 1]:
            rotate = random.choice(rotates)
            p += transform(path, crop_type, a_type, rotate)
    return path, p


def transform(path, crop_type=None, alter_type=None, rotate=None):

    new_path = path + '_crop{}_{}_r{}.jpg'.format(crop_type, alter_type, rotate)
    if os.path.exists(new_path):
        return new_path

    img = Image.open(path)
    w, h = img.size

    crop_types = [
        (w // 2 - 256, h // 2 - 256, w // 2 + 256, h // 2 + 256),
        (0, 0, 512, 512),
        (w - 512, 0, w, 512),
        (0, h - 512, 512, h),
        (w - 512, h - 512, w, h),
    ]

    def _resize(im, scale):
        return im.resize((int(w * scale), int(h * scale)), resample=Image.BICUBIC)

    def _gamma(im, gamma):
        return ImageEnhance.Brightness(im).enhance(gamma)

    def _gamma2(im, gamma):
        return exposure.adjust_gamma(im, gamma=gamma)

    alter_types = {
        'resize05': lambda x: _resize(x, 0.5),
        'resize08': lambda x: _resize(x, 0.8),
        'resize15': lambda x: _resize(x, 1.5),
        'resize20': lambda x: _resize(x, 2.0),
        'gamma08': lambda x: _gamma(x, 0.8),
        'gamma12': lambda x: _gamma(x, 1.2),
        'q70': lambda x: x,
        'q90': lambda x: x
    }

    if crop_type is not None:
        img = img.crop(crop_types[crop_type])

    if rotate > 0:
        img = img.rotate(rotate)

    img = alter_types.get(alter_type, lambda x: x)(img)

    if alter_type == 'q70':
        img.save(new_path, 'JPEG', quality=70)
    elif alter_type == 'q90':
        img.save(new_path, 'JPEG', quality=90)
    else:
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