import math

import numpy as np
from PIL import Image
from keras.utils import Sequence


class TrainSequence(Sequence):

    def __init__(self, paths, y_train, batch_size):
        self.x, self.y = paths, y_train
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(1.0 * len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([read_and_resize(p) for p in batch_x]), np.array(batch_y)


class PredictSequence(Sequence):
    def __init__(self, paths, batch_size):
        self.x = paths
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(1.0 * len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([read_and_resize(p) for p in batch_x])


def read_and_resize(f_path):
    im_array = np.array(Image.open(f_path), dtype="uint8")
    pil_im = Image.fromarray(im_array)
    new_array = np.array(pil_im.resize((256, 256)))
    return new_array / 255.0
