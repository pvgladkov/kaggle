# coding: utf-8

from keras import Input, activations, models, optimizers, losses
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D, Dropout, GlobalMaxPool2D, Dense


class CNN1(object):

    name = 'cnn1'

    @staticmethod
    def model(n_class, batch_size):
        input_shape = (256, 256, 3)
        batch_shape = (None, 256, 256, 3)
        inp = Input(batch_shape=batch_shape)
        norm_inp = BatchNormalization()(inp)
        img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu, padding="same")(norm_inp)
        img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu, padding="same")(img_1)
        img_1 = MaxPooling2D(pool_size=(3, 3))(img_1)
        img_1 = Dropout(rate=0.2)(img_1)
        img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu, padding="same")(img_1)
        img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu, padding="same")(img_1)
        img_1 = MaxPooling2D(pool_size=(3, 3))(img_1)
        img_1 = Dropout(rate=0.2)(img_1)
        img_1 = Convolution2D(64, kernel_size=2, activation=activations.relu, padding="same")(img_1)
        img_1 = Convolution2D(20, kernel_size=2, activation=activations.relu, padding="same")(img_1)
        img_1 = GlobalMaxPool2D()(img_1)
        img_1 = Dropout(rate=0.2)(img_1)
        dense_1 = Dense(20, activation=activations.relu)(img_1)
        dense_1 = Dense(n_class, activation=activations.softmax)(dense_1)

        model = models.Model(inputs=inp, outputs=dense_1)
        opt = optimizers.Adam()

        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        model.summary()
        return model
