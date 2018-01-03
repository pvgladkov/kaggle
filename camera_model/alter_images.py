# coding: utf-8

from multiprocessing import Pool
from kaggle.utils import create_logger
import pandas as pd
from PIL import Image, ImageEnhance
import random

home_path = '/var/local/pgladkov/camera_model'
data_path = '/var/local/pgladkov/camera_model/data'
train_path = data_path + '/train'
test_path = data_path + '/test'
features_path = home_path + '/features'

logger = create_logger('alter_images')


def make_alter(path):

    logger.info(path)
    p = []
    for crop_type in range(5):
        _as = ['resize05', 'resize08', 'resize15', 'resize20', 'gamma08', 'gamma12', 'q70', 'q90', None]
        a_type = random.choice(_as)
        p += transform(path, crop_type, a_type)
    return path, p


def transform(path, crop_type=0, alter_type=None):
    img = Image.open(path)
    w, h = img.size

    crop_types = [
        (w // 2 - 256, h // 2 - 256, w // 2 + 256, h // 2 + 256),
        (0, 0, 512, 512),
        (w - 512, 0, w, 512),
        (0, h - 512, 512, h),
        (w - 512, h - 512, w, h),
    ]

    alter_types = {
        'resize05': lambda x: x.resize((int(w*0.5), int(h*0.5)), resample=Image.BICUBIC),
        'resize08': lambda x: x.resize((int(w*0.8), int(h*0.8)), resample=Image.BICUBIC),
        'resize15': lambda x: x.resize((int(w*1.5), int(h*1.5)), resample=Image.BICUBIC),
        'resize20': lambda x: x.resize((int(w*2.0), int(h*2.0)), resample=Image.BICUBIC),
        'gamma08': lambda x: ImageEnhance.Brightness(x).enhance(0.8),
        'gamma12': lambda x: ImageEnhance.Brightness(x).enhance(1.2),
        'q70': lambda x: x,
        'q90': lambda x: x
    }

    img = img.crop(crop_types[crop_type])
    img = alter_types.get(alter_type, lambda x: x)(img)

    new_path = path + '_crop{}_{}.png'.format(crop_type, alter_type)
    if alter_type == 'q70':
        img.save(new_path, quality=70, optimize=True, progressive=True)
    elif alter_type == 'q90':
        img.save(new_path, quality=90, optimize=True, progressive=True)
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