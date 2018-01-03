# coding: utf-8

import os
import multiprocessing as mp
from kaggle.utils import create_logger

import cv2
import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA

VERSION = 1

data_path = '/var/local/pgladkov/camera_model/data'
train_path = data_path + '/train'
test_path = data_path + '/test'

logger = create_logger('pca_baseline')


def get_center_crop(img, d=250):
    cy = img.shape[0] // 2
    cx = img.shape[1] // 2
    return img[cy - d:cy + d, cx - d:cx + d]


def color_stats(q, iolock):
    while True:
        img_path = q.get()
        if img_path is None:
            break
        
        if type(img_path) is tuple:
            img = cv2.imread(train_path + '/' + img_path[0] + '/' + img_path[1])
            key = img_path[1]
        else:
            img = cv2.imread(test_path + '/' + img_path)
            key = img_path         
        
        # Some images read return info in a 2nd dim. We only want the first dim.
        if img.shape == (2,):
            img = img[0]
        
        # crop to center as in test    
        img = get_center_crop(img)
        pca_feats = get_pca_features(img)
        color_info[key] = (pca_feats[0][0], pca_feats[0][1],
                           pca_feats[0][2], pca_feats[0][3], pca_feats[0][4])


def get_pca_features(img):
    img = np.ravel(img).reshape(1, -1)
    return pf.transform(img)


if __name__ == '__main__':

    cameras = os.listdir(train_path)

    train_images = []
    for camera in cameras:
        for fname in sorted(os.listdir(train_path + '/' + camera)):
            train_images.append((camera, fname))

    train = pd.DataFrame(train_images, columns=['camera', 'fname'])
    logger.info('train.shape {}'.format(train.shape))

    test_images = []
    for fname in sorted(os.listdir(test_path)):
        test_images.append(fname)

    test = pd.DataFrame(test_images, columns=['fname'])
    logger.info('test.shape {}'.format(test.shape))

    n_components = 5
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)

    # Get some training data for PCA
    random_images = train.sample(100)

    img_set_reds = []
    for i, r in random_images.iterrows():
        # If you uncomment last part, you can extract features only over a certain channel
        x = get_center_crop(cv2.imread(train_path + '/' + train['camera'][i] + '/' + train['fname'][i]))
        img_set_reds.append(np.ravel(x))  # PCA takes instances as flatten vectors, not 2-d array

    img_set_reds = np.asarray(img_set_reds)
    logger.info('img_set_reds.shape {}'.format(img_set_reds.shape))

    pf = pca.fit(np.asarray(img_set_reds))

    cols = ['pca0', 'pca1', 'pca2', 'pca3', 'pca4']

    for col in cols:
        train[col] = None
        test[col] = None

    NCORE = 8

    color_info = mp.Manager().dict()

    q = mp.Queue(maxsize=NCORE)
    iolock = mp.Lock()
    pool = mp.Pool(NCORE, initializer=color_stats, initargs=(q, iolock))

    for i in train_images:
        q.put(i)

    for i in test_images:
        q.put(i)

    for _ in range(NCORE):
        q.put(None)
    pool.close()
    pool.join()

    color_info = dict(color_info)

    for n, col in enumerate(cols):
        train[col] = train['fname'].apply(lambda x: color_info[x][n])
        test[col] = test['fname'].apply(lambda x: color_info[x][n])

    train.to_csv('train_pca_features_v{}.csv'.format(VERSION), index=False)
    test.to_csv('test_pca_features_v{}.csv'.format(VERSION), index=False)