# coding: utf-8

import numpy as np
import pandas as pd

from file_utils import get_test_df
from data_utils import PredictFileSequence


def make_prediction(model, label_index, s_name, batch_size, crop_shape, test_path):
    test_df = get_test_df(test_path)
    X_test = PredictFileSequence(test_df['path'].values, batch_size, crop_shape)

    predicts = model.predict_generator(X_test, use_multiprocessing=True, workers=10)

    predicts = np.argmax(predicts, axis=1)
    predicts = [label_index[p] for p in predicts]
    assert test_df.shape[0] == len(predicts), '{} != {}'.format(test_df.shape[0], len(predicts))

    df = pd.DataFrame(columns=['fname', 'camera'])
    df['fname'] = test_df['fname'].values
    df['camera'] = predicts

    df.to_csv(s_name, index=False)
    return s_name
