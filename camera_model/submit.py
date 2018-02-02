# coding: utf-8

import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean

from collections import Counter

from utils.data_utils import label_transform, PredictFileSequence
from settings import train_path, test_path, submissions_path, weights_path
from utils.file_utils import get_train_df, get_test_df
from models import CNN1, CNNConv4, CNNConv10
from kaggle.utils import create_logger

from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000

logger = create_logger('submit')

if __name__ == '__main__':

    batch_size = 16
    crop_shape = 64

    train_df = get_train_df(train_path, use_crop=False, use_original=True)
    logger.info('train.shape={}'.format(train_df.shape))

    y, label_index = label_transform(train_df['camera'].values)
    y = np.array(y)

    model = CNNConv10.model(len(label_index), batch_size, crop_shape)

    model_version = 'cnn_conv10-20180129-2308.best'

    file_path = "{}/{}.hdf5".format(weights_path, model_version)
    model.load_weights(file_path)

    s_name = '{}/submission-{}-gmean.csv'.format(submissions_path, model_version)

    test_df = get_test_df(test_path)

    predictions = []

    for i in range(20):
        X_test = PredictFileSequence(test_df['path'].values, batch_size, crop_shape)
        predicts = model.predict_generator(X_test, use_multiprocessing=True, workers=10)
        predictions.append(predicts)

    final_predicts = []

    for row in zip(*predictions):
        predicts = gmean(row, axis=0)
        final_predicts.append(predicts)

    final_predicts = np.argmax(final_predicts, axis=1)
    final_predicts = [label_index[p] for p in final_predicts]

    df = pd.DataFrame(columns=['fname', 'camera'])
    df['fname'] = test_df['fname'].values
    df['camera'] = final_predicts

    df.to_csv(s_name, index=False)

    logger.info('save to {}'.format(s_name))