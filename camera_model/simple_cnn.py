# coding: utf-8

import sys
import time
import numpy as np
import pandas as pd


from keras.callbacks import (ModelCheckpoint, EarlyStopping)

from utils.data_utils import PredictFileSequence, TrainFileSequenceOnFly, TrainFileSequence
from utils.callbacks import TelegramMonitor
from settings import train_path, test_path, submissions_path, weights_path, telegram_bot_api_key, chat_it
from utils.file_utils import get_test_df, get_train_df, load_files
from models import CNN1, ResNet50
from sklearn.model_selection import train_test_split
from kaggle.utils import create_logger, time_v

from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000

logger = create_logger('simple_cnn')


def label_transform(labels):
    labels = pd.get_dummies(pd.Series(labels))
    index = labels.columns.values
    return labels, index


def init_callbacks(w_file, m_name, v):
    checkpoint = ModelCheckpoint(w_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # early = EarlyStopping(monitor="val_acc", mode="max", patience=10)
    _name = '{}-{}'.format(m_name, v)
    telegram = TelegramMonitor(api_token=telegram_bot_api_key, chat_id=chat_it, model_name=_name)
    return [checkpoint, telegram]


if __name__ == '__main__':

    start_time = time.time()

    batch_size = 8
    workers = 25
    version = time_v()
    epoch = 200

    logger.info('batch_size={}'.format(batch_size))
    logger.info('workers={}'.format(workers))
    logger.info('version={}'.format(version))
    logger.info('epoch={}'.format(epoch))

    train_df = get_train_df(train_path, use_crop=False, use_original=True)
    logger.info('train.shape={}'.format(train_df.shape))

    test_df = get_test_df(test_path)
    logger.info('test.shape={}'.format(test_df.shape))

    y, label_index = label_transform(train_df['camera'].values)
    y = np.array(y)
    logger.info('y.shape={}'.format(y.shape))

    logger.info('train model')

    model = CNN1.model(len(label_index), batch_size)
    model_name = CNN1.name

    model.load_weights('{}/cnn1-20180108-0822.best.hdf5'.format(weights_path))

    file_path = "{}/{}-{}.best.hdf5".format(weights_path, model_name, version)
    callbacks_list = init_callbacks(file_path, model_name, version)

    train_train, train_validate, y_train, y_validate = train_test_split(train_df['path'].values, y,
                                                                        test_size=0.1, random_state=777)

    X_y_train = TrainFileSequenceOnFly(train_train, y_train, batch_size)
    X_y_validate = TrainFileSequenceOnFly(train_validate, y_validate, batch_size)

    history = model.fit_generator(X_y_train, epochs=epoch, verbose=2, use_multiprocessing=True,
                                  workers=workers, callbacks=callbacks_list, validation_data=X_y_validate)

    model.load_weights(file_path)

    logger.info('make prediction')
    X_test = PredictFileSequence(test_df['path'].values, batch_size)

    predicts = model.predict_generator(X_test, use_multiprocessing=True, workers=workers)

    predicts = np.argmax(predicts, axis=1)
    predicts = [label_index[p] for p in predicts]
    assert test_df.shape[0] == len(predicts), '{} != {}'.format(test_df.shape[0], len(predicts))

    df = pd.DataFrame(columns=['fname', 'camera'])
    df['fname'] = test_df['fname'].values
    df['camera'] = predicts

    s_name = '{}/submission-{}-{}.csv'.format(submissions_path, model_name, version)
    df.to_csv(s_name, index=False)
    logger.info('save to {}'.format(s_name))

    logger.info('{} sec'.format(time.time() - start_time))
