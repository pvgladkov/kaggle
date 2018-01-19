# coding: utf-8

import time
import numpy as np

from utils.data_utils import label_transform, train_val_gen
from settings import train_path, test_path, submissions_path, weights_path, telegram_bot_api_key, chat_it, log_path
from utils.file_utils import get_train_df
from models import CNN1, CNNConv4, CNNConv10
from utils.callbacks import init_callbacks
from kaggle.utils import create_logger, time_v

from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = 1000000000
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = create_logger('train_cnn')

if __name__ == '__main__':

    start_time = time.time()

    batch_size = 16
    workers = 40
    version = time_v()
    epoch = 20
    crop_shape = 64

    logger.info('batch_size={}'.format(batch_size))
    logger.info('workers={}'.format(workers))
    logger.info('version={}'.format(version))
    logger.info('epoch={}'.format(epoch))

    train_df = get_train_df(train_path, use_crop=False, use_original=True)
    logger.info('train.shape={}'.format(train_df.shape))

    y, label_index = label_transform(train_df['camera'].values)
    y = np.array(y)
    logger.info('y.shape={}'.format(y.shape))

    logger.info('train model')

    model = CNNConv10.model(len(label_index), batch_size, crop_shape)
    model_name = CNNConv10.name

    weights = '{}/cnn_conv10-20180118-2357.latest-20.hdf5'.format(weights_path)
    model.load_weights(weights)
    logger.info('load {}'.format(weights))

    file_path = "{}/{}-{}.best.hdf5".format(weights_path, model_name, version)
    callbacks_list = init_callbacks(file_path, model_name, version, telegram_bot_api_key, chat_it, log_path, 64)

    for i in range(10):
        X_y_train, X_y_validate = train_val_gen(train_df['path'].values, y, batch_size, crop_shape)

        history = model.fit_generator(X_y_train, epochs=epoch, verbose=2, use_multiprocessing=True,
                                      workers=workers, callbacks=callbacks_list, validation_data=X_y_validate,
                                      steps_per_epoch=1*len(X_y_train), validation_steps=1*len(X_y_validate))

        latest_file_path = "{}/{}-{}.latest-{}.hdf5".format(weights_path, model_name, version, epoch)
        logger.info('save weights {}'.format(latest_file_path))
        model.save(latest_file_path)

    logger.info('{} sec'.format(time.time() - start_time))
