import numpy as np
from keras import backend as K
from sklearn import metrics


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def fbeta_score(y_true, y_pred, beta=1):

    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    return (1 + bb) * (p * r) / (bb * p + r + K.epsilon())


def fmeasure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)


def fmeasure_th(y_true, y_pred):
    best_thresh = 0.5
    best_score = 0.0
    for thresh in np.arange(0.1, 0.501, 0.01):
        thresh = np.round(thresh, 2)
        score = metrics.f1_score(y_true, (y_pred > thresh).astype(int))
        if score > best_score:
            best_thresh = thresh
            best_score = score
    return best_score, best_thresh
