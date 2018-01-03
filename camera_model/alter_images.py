# coding: utf-8

import os
from multiprocessing import Pool
from kaggle.utils import create_logger

import cv2
import scipy
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from sklearn.cluster import KMeans

home_path = '/var/local/pgladkov/camera_model'
data_path = '/var/local/pgladkov/camera_model/data'
train_path = data_path + '/train'
test_path = data_path + '/test'
features_path = home_path + '/features'

logger = create_logger('alter_images')

altered_types = {'JPEG compression with quality factor': [70, 90],
                 'resizing (via bicubic interpolation) by a factor of': [0.5, 0.8, 1.5, 2.0],
                 'gamma correction using gamma': [0.8, 1.2]}


def get_features1(path):
    st = []
    try:
        st += [random.choice([0, 1])]
        img = ''
        if st[0] == 0:
            img = Image.open(path) #.resize((512,512), resample=Image.NEAREST)
            w, h = img.size
            img = img.crop((w // 2 - 256, h // 2 - 256, w // 2 + 256, h // 2 + 256))
            img.save(path.split('/')[-1]+'.png')
        else:
            alter_to = random.choice(altered_types['resizing (via bicubic interpolation) by a factor of'])
            img = Image.open(path)
            w, h = img.size
            img = img.resize((int(w*alter_to), int(h*alter_to)), resample=Image.BILINEAR)
            w, h = img.size
            img = img.crop((w // 2 - 256, h // 2 - 256, w // 2 + 256, h // 2 + 256))
            img.save(path.split('/')[-1]+'.png')
        im_stats_ = ImageStat.Stat(img)
        st += im_stats_.sum
        st += im_stats_.mean
        st += im_stats_.rms
        st += im_stats_.var
        st += im_stats_.stddev
        img = np.array(img)[:, :, :3]
        for i in range(3):
            st += [scipy.stats.kurtosis(img[:, :, i].ravel())]
            st += [scipy.stats.skew(img[:, :, i].ravel())]
            st += [np.min(img[:, :, i].ravel())]
            st += [np.mean(img[:, :, i].ravel())]
            st += [np.median(img[:, :, i].ravel())]
            st += [np.max(img[:, :, i].ravel())]

        # should align with image alter above or will not work with test alters
        img = cv2.imread(path.split('/')[-1]+'.png') #cv2.resize(cv2.imread(path), (512,512))
        cy = img.shape[0] // 2
        cx = img.shape[1] // 2
        img = img[cy - 256:cy + 256, cx - 256:cx + 256]
        bw = cv2.imread(path.split('/')[-1]+'.png', 0) #cv2.resize(cv2.imread(path,0), (512,512))
        bw = bw[cy - 256:cy + 256, cx - 256:cx + 256]
        st += list(cv2.calcHist([bw], [0], None, [256], [0, 256]).flatten()) #bw
        st += list(cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()) #r
        st += list(cv2.calcHist([img], [1], None, [256], [0, 256]).flatten()) #g
        st += list(cv2.calcHist([img], [2], None, [256], [0, 256]).flatten()) #b
        m, s = cv2.meanStdDev(img)
        st += list(m.flatten())
        st += list(s.flatten())
        st += [cv2.Laplacian(bw, cv2.CV_64F).var()]
        st += [cv2.Laplacian(img, cv2.CV_64F).var()]
        st += [cv2.Sobel(bw, cv2.CV_64F, 1, 0, ksize=5).var()]
        st += [cv2.Sobel(bw, cv2.CV_64F, 0, 1, ksize=5).var()]
        st += [cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5).var()]
        st += [cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5).var()]
        st += [(bw < 50).sum()]
        st += [(bw > 205).sum()]
        os.remove(path.split('/')[-1]+'.png')
    except:
        print(path)
    return [path, st]


def get_features2(path):
    st = []
    try:
        img = Image.open(path)
        im_stats_ = ImageStat.Stat(img)
        st += im_stats_.sum
        st += im_stats_.mean
        st += im_stats_.rms
        st += im_stats_.var
        st += im_stats_.stddev
        img = np.array(img)[:, :, :3]
        for i in range(3):
            st += [scipy.stats.kurtosis(img[:, :, i].ravel())]
            st += [scipy.stats.skew(img[:, :, i].ravel())]
            st += [np.min(img[:, :, i].ravel())]
            st += [np.mean(img[:, :, i].ravel())]
            st += [np.median(img[:, :, i].ravel())]
            st += [np.max(img[:, :, i].ravel())]
        img = cv2.imread(path)
        bw = cv2.imread(path, 0)
        st += list(cv2.calcHist([bw], [0], None, [256], [0,256]).flatten()) #bw
        st += list(cv2.calcHist([img], [0], None, [256], [0,256]).flatten()) #r
        st += list(cv2.calcHist([img], [1], None, [256], [0,256]).flatten()) #g
        st += list(cv2.calcHist([img], [2], None, [256], [0,256]).flatten()) #b
        m, s = cv2.meanStdDev(img)
        st += list(m.flatten())
        st += list(s.flatten())
        st += [cv2.Laplacian(bw, cv2.CV_64F).var()]
        st += [cv2.Laplacian(img, cv2.CV_64F).var()]
        st += [cv2.Sobel(bw, cv2.CV_64F, 1, 0, ksize=5).var()]
        st += [cv2.Sobel(bw, cv2.CV_64F, 0, 1, ksize=5).var()]
        st += [cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5).var()]
        st += [cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5).var()]
        st += [(bw < 50).sum()]
        st += [(bw > 205).sum()]
    except:
        print(path)
    return [path, st]


def normalize_img(rows, func):
    imf_d = {}
    p = Pool(10)
    ret = p.map(func, rows)
    for i in range(len(ret)):
        imf_d[ret[i][0]] = ret[i][1]

    fdata = [imf_d[f] for f in rows]
    return fdata


VERSION = 1

if __name__ == '__main__':

    train = pd.read_csv('{}/train_pca_features_v{}.csv'.format(features_path, 1))
    test = pd.read_csv('{}/test_pca_features_v{}.csv'.format(features_path, 1))

    train['path'] = train.apply(lambda x: '{}/{}/{}'.format(train_path, x['camera'], x['fname']), axis=1)
    train['altered'] = 0

    test['path'] = test.apply(lambda x: '{}/{}'.format(test_path, x['fname']), axis=1)
    test['altered'] = test['fname'].map(lambda x: 1 if 'manip' in x else 0)

    logger.info('train.shape {}'.format(train.shape))
    logger.info('test.shape {}'.format(test.shape))

    logger.info('train normalize_img')
    X_train = normalize_img(train['path'].values, get_features1)

    logger.info('test normalize_img')
    X_test = normalize_img(test['path'].values, get_features2)

    X_test = np.append([[x] for x in test['altered'].values], X_test, axis=1)

    for c in ['pca0', 'pca1', 'pca2', 'pca3', 'pca4']:
        X_train = np.append(X_train, [[x] for x in train[c].values], axis=1)
        X_test = np.append(X_test, [[x] for x in test[c].values], axis=1)

    logger.info('X_train.shape {}'.format(X_train.shape))
    logger.info('X_test.shape {}'.format(X_test.shape))

    pd.DataFrame(X_train).to_csv('{}/train_features_v{}.csv'.format(features_path, 3), index=False)
    pd.DataFrame(X_test).to_csv('{}/test_features_v{}.csv'.format(features_path, 3), index=False)
