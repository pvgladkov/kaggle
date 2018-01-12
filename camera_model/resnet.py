# coding: utf-8

import time
import numpy as np

from utils.data_utils import train_val_gen, label_transform
from utils.submissions import make_prediction
from utils.callbacks import init_callbacks
from settings import train_path, test_path, weights_path, telegram_bot_api_key, chat_it, submissions_path
from utils.file_utils import get_test_df, get_train_df
from models import ResNet50XfrLearn, ResNet50
from kaggle.utils import create_logger, time_v

from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000

logger = create_logger('simple_cnn')


if __name__ == '__main__':

    start_time = time.time()

    batch_size = 4
    workers = 25
    version = time_v()
    epoch = 40
    crop_shape = 224

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

    model = ResNet50XfrLearn.model(len(label_index), batch_size)
    model_name = ResNet50XfrLearn.name

    model.load_weights('{}/resnet50xfr-20180112-1227.best.hdf5'.format(weights_path))

    best_file_path = "{}/{}-{}.best.hdf5".format(weights_path, model_name, version)
    callbacks_list = init_callbacks(best_file_path, model_name, version, telegram_bot_api_key, chat_it)

    X_y_train, X_y_validate = train_val_gen(train_df['path'].values, y, batch_size, crop_shape)

    history = model.fit_generator(X_y_train, epochs=epoch, verbose=2, use_multiprocessing=True,
                                  workers=workers, callbacks=callbacks_list, validation_data=X_y_validate)

    latest_file_path = "{}/{}-{}.latest-{}.hdf5".format(weights_path, model_name, version, epoch)
    model.save(latest_file_path)

    s_name = '{}/submission-{}-{}.csv'.format(submissions_path, model_name, version)
    make_prediction(model, label_index, s_name, batch_size, crop_shape, test_path)

    logger.info('save prediction {}'.format(s_name))
    logger.info('{} sec'.format(time.time() - start_time))
