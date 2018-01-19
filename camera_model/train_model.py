# coding: utf-8

import time
import numpy as np

from utils.data_utils import label_transform, train_val_gen
from settings import train_path, test_path, submissions_path, weights_path, telegram_bot_api_key, chat_it, log_path
from utils.file_utils import get_train_df
from utils.data_utils import PredictFileSequence
from models import CNN1, CNNConv4, CNNConv10
from kaggle.utils import create_logger, time_v
from keras import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier

from sklearn.svm import SVC

from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = 1000000000
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = create_logger('train_model')

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
    logger.info('train.shape={}'.format(train_df.shape))

    y, label_index = label_transform(train_df['camera'].values)
    y = np.array(y)
    logger.info('y.shape={}'.format(y.shape))

    train_train, train_validate, y_train, y_validate = train_test_split(train_df['path'].values, y,
                                                                        test_size=0.1, random_state=777)

    X_train = PredictFileSequence(train_train, batch_size, crop_shape)
    X_validate = PredictFileSequence(train_validate, batch_size, crop_shape)

    model = CNNConv10.model(len(label_index), batch_size, crop_shape)
    model_name = CNNConv10.name

    model.load_weights('{}/cnn_conv10-20180116-2114.latest-20.hdf5'.format(weights_path))

    logger.info('CNN predict')
    predicts = model.predict_generator(X_validate, use_multiprocessing=True, workers=workers)
    final_predicts = np.argmax(predicts, axis=1)

    score = accuracy_score(np.argmax(y_validate, axis=1), final_predicts)
    logger.info('CNN score {}'.format(score))

    logger.info('train model')

    layer_name = 'global_max_pooling2d_1'
    intermediate_layer_model = models.Model(inputs=model.input,
                                            outputs=model.get_layer(layer_name).output)

    X = intermediate_layer_model.predict_generator(X_train, workers=workers, use_multiprocessing=True)
    logger.info(X.shape)

    X_val = intermediate_layer_model.predict_generator(X_validate, workers=workers, use_multiprocessing=True)
    logger.info(X_val.shape)

    logger.info('train')
    svm_model = SVC(C=0.1)
    svm_model.fit(X, np.argmax(y_train, axis=1))

    logger.info('predict')
    predicts = svm_model.predict(X_val)

    score = accuracy_score(np.argmax(y_validate, axis=1), predicts)
    logger.info('SVM score {}'.format(score))



