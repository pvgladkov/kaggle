# coding: utf-8

import time
import numpy as np
import pandas as pd

from models.cnn2 import train_val_gen, CNNConv18

from utils.callbacks import init_callbacks
from utils.data_utils import label_transform
from kaggle.utils import create_logger, time_v

from settings import weights_path, telegram_bot_api_key, chat_it, log_path, home_path

from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = 1000000000
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = create_logger('train_cnn')

if __name__ == '__main__':

    start_time = time.time()

    batch_size = 8
    workers = 40
    version = time_v()
    epoch = 20
    crop_shape = 224

    logger.info('batch_size={}'.format(batch_size))
    logger.info('workers={}'.format(workers))
    logger.info('version={}'.format(version))
    logger.info('epoch={}'.format(epoch))

    train_df = pd.read_csv(home_path + '/train_data.csv')
    logger.info('train.shape={}'.format(train_df.shape))

    y, label_index = label_transform(train_df['camera'].values)
    y = np.array(y)
    logger.info('y.shape={}'.format(y.shape))

    logger.info('train model')

    model = CNNConv18.model(len(label_index), batch_size, crop_shape)
    model_name = CNNConv18.name

    weights = '{}/cnn2.cnn_conv18-20180204-1014.best.hdf5'.format(weights_path)
    model.load_weights(weights)
    logger.info('load {}'.format(weights))

    file_path = "{}/{}-{}.best.hdf5".format(weights_path, model_name, version)
    callbacks_list = init_callbacks(file_path, model_name, version, telegram_bot_api_key, chat_it, log_path, 64)

    train_files = train_df['path'].values

    start_epoch = 300

    for i in range(10):

        X_y_train, X_y_validate = train_val_gen(train_files, y, batch_size, crop_shape)

        history = model.fit_generator(X_y_train, epochs=epoch, verbose=2, use_multiprocessing=True,
                                      workers=workers, callbacks=callbacks_list, validation_data=X_y_validate,
                                      validation_steps=1*len(X_y_validate))

        start_epoch += epoch

        latest_file_path = "{}/{}-{}.latest-{}.hdf5".format(weights_path, model_name, version, start_epoch)
        logger.info('save weights {}'.format(latest_file_path))
        model.save(latest_file_path)

    logger.info('{} sec'.format(time.time() - start_time))
