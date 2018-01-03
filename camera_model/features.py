# coding: utf-8

import cv2
import scipy
import numpy as np
from PIL import Image, ImageStat


def get_features(img, bw):
    st = []
    st += list(cv2.calcHist([bw], [0], None, [256], [0, 256]).flatten())  # bw
    st += list(cv2.calcHist([img], [0], None, [256], [0, 256]).flatten())  # r
    st += list(cv2.calcHist([img], [1], None, [256], [0, 256]).flatten())  # g
    st += list(cv2.calcHist([img], [2], None, [256], [0, 256]).flatten())  # b
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
    return st


def image_stat(img):
    im_stats_ = ImageStat.Stat(img)
    st = []
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
    return st