import pandas as pd

from kaggle.utils import create_logger

logger = create_logger('data_utils')


ALL_AUGMENTATIONS = ['resize05', 'resize08', 'resize15', 'resize20', 'gamma08', 'gamma12', 'q70', 'q90']
GAMMA = ['gamma08', 'gamma12']
RESIZE = ['resize05', 'resize08', 'resize15', 'resize20']
QUALITY = ['q70', 'q90']
ROTATES = [90, 180, 270, 0]
CROPS = [0, 1, 2, 3, 4]


def label_transform(labels):
    labels = pd.get_dummies(pd.Series(labels))
    index = labels.columns.values
    return labels, index
