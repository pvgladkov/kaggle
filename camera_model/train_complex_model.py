# coding: utf-8

import time
import numpy as np
import pandas as pd

from scipy.stats.mstats import gmean

from utils.data_utils import label_transform
from settings import train_path, test_path, submissions_path, weights_path, telegram_bot_api_key, chat_it, log_path
from utils.file_utils import get_train_df, get_test_df
from camera_model.models.cnn1.data_provider import PredictFileSequence, PredictValidateFileSequence, train_val_gen
from camera_model.models import CNN1, CNNConv4, CNNConv10
from kaggle.utils import create_logger, time_v
from keras import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier

from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier

from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = 1000000000
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = create_logger('train_model')


def make_x(xtrain, ytrain, feature_model, scale):
    new_x = None
    new_y = None
    ytrain = np.array(ytrain).reshape((len(ytrain), 1))
    for i in range(scale):
        logger.info(i)
        _x = feature_model.predict_generator(xtrain, workers=workers, use_multiprocessing=True)
        if new_x is None:
            new_x = _x
            new_y = ytrain
        else:
            new_x = np.concatenate([new_x, _x], axis=0)
            new_y = np.concatenate([new_y, ytrain], axis=0)

    return np.array(new_x), new_y.reshape((new_y.shape[0],))


if __name__ == '__main__':

    start_time = time.time()

    batch_size = 16
    workers = 10
    version = time_v()
    crop_shape = 64

    logger.info('batch_size={}'.format(batch_size))
    logger.info('workers={}'.format(workers))
    logger.info('version={}'.format(version))

    train_df = get_train_df(train_path, use_crop=False, use_original=True)
    test_df = get_test_df(test_path)
    logger.info('train.shape={}'.format(train_df.shape))

    y, label_index = label_transform(train_df['camera'].values)
    y = np.array(y)
    logger.info('y.shape={}'.format(y.shape))

    train_train, train_validate, y_train, y_validate = train_test_split(train_df['path'].values, y,
                                                                        test_size=0.1, random_state=777)

    X_train = PredictValidateFileSequence(train_train, batch_size, crop_shape)
    X_validate = PredictValidateFileSequence(train_validate, batch_size, crop_shape)
    X_test = PredictFileSequence(test_df['path'].values, batch_size, crop_shape)

    model = CNNConv10.model(len(label_index), batch_size, crop_shape)
    model_name = CNNConv10.name

    model_version = 'cnn_conv10-20180116-2114.latest-20'
    model.load_weights('{}/{}.hdf5'.format(weights_path, model_version))

    logger.info('CNN predict')
    predicts = model.predict_generator(X_validate, use_multiprocessing=True, workers=workers)
    final_predicts = np.argmax(predicts, axis=1)

    score = accuracy_score(np.argmax(y_validate, axis=1), final_predicts)
    logger.info('CNN score {}'.format(score))

    logger.info('train model')

    layer_name = 'global_max_pooling2d_1'
    intermediate_layer_model = models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    logger.info('train features')
    X1_train, y1_train = make_x(X_train, np.argmax(y_train, axis=1), intermediate_layer_model, 10)
    logger.info('shape {} {}'.format(X1_train.shape, y1_train.shape))

    logger.info('train')
    # svm_model = XGBClassifier(max_depth=12, learning_rate=0.01, nthread=20, silent=True)
    # svm_model = SVC(C=1, probability=True)
    svm_model = OneVsRestClassifier(LinearSVC())
    svm_model.fit(X1_train, y1_train)

    logger.info('validate features')

    predictions = []
    y1_val = None
    for i in range(1):
        logger.info('predict {}'.format(i))
        X1_val, y1_val = make_x(X_validate, np.argmax(y_validate, axis=1), intermediate_layer_model, 1)
        svm_predicts = svm_model.predict(X1_val)
        predictions.append(svm_predicts)

    val_predicts = []
    for row in zip(*predictions):
        predicts = gmean(row, axis=0)
        val_predicts.append(predicts)

    val_predicts = np.argmax(val_predicts, axis=1)

    score = accuracy_score(y1_val, val_predicts)
    logger.info('SVM score {}'.format(score))

    logger.info('test')

    test_predictions = []
    for i in range(20):
        logger.info('predict {}'.format(i))
        X1_test, _ = make_x(X_test, [], intermediate_layer_model, 1)
        svm_predicts = svm_model.decision_function(X1_test)
        test_predictions.append(svm_predicts)

    test_predicts = []
    for row in zip(*test_predictions):
        predicts = gmean(row, axis=0)
        test_predicts.append(predicts)

    test_predicts = np.argmax(test_predicts, axis=1)

    final_predicts = [label_index[p] for p in test_predicts]
    assert len(final_predicts) == len(test_df['path'].values)

    df = pd.DataFrame(columns=['fname', 'camera'])
    df['fname'] = test_df['fname'].values
    df['camera'] = final_predicts

    s_name = '{}/submission-{}-svm-1.csv'.format(submissions_path, model_version)
    df.to_csv(s_name, index=False)

    logger.info('save to {}'.format(s_name))