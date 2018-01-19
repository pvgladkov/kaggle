# coding: utf-8

from keras import Input, activations, models, optimizers, losses
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D, Dropout, GlobalMaxPool2D, Dense


class CNN1(object):

    name = 'cnn1'

    @staticmethod
    def model(n_class, batch_size, shape):
        batch_shape = (None, shape, shape, 3)
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
        opt = optimizers.SGD(lr=1e-4, momentum=0.9)

        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        model.summary()
        return model


class CNNConv4(object):

    name = 'cnn_conv4'

    @staticmethod
    def model(n_class, batch_size, shape):
        batch_shape = (None, shape, shape, 3)
        inp = Input(batch_shape=batch_shape)
        norm_inp = BatchNormalization()(inp)

        img = Convolution2D(32, kernel_size=4, activation=activations.linear, padding="same", strides=1)(norm_inp)
        img = MaxPooling2D(pool_size=(2, 2), strides=2)(img)

        img = Convolution2D(48, kernel_size=5, activation=activations.linear, padding="same", strides=1)(img)
        img = MaxPooling2D(pool_size=(2, 2), strides=2)(img)

        img = Convolution2D(64, kernel_size=5, activation=activations.linear, padding="same", strides=1)(img)
        img = MaxPooling2D(pool_size=(2, 2), strides=2)(img)

        img = Convolution2D(128, kernel_size=5, activation=activations.linear, padding="same", strides=1)(img)
        img = GlobalMaxPool2D()(img)

        img = Dropout(rate=0.2)(img)

        img = Dense(128, activation=activations.relu)(img)
        dense = Dense(n_class, activation=activations.softmax)(img)

        model = models.Model(inputs=inp, outputs=dense)
        opt = optimizers.SGD(lr=1e-3, momentum=0.9)

        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        model.summary()
        return model


class CNNConv10(object):
    name = 'cnn_conv10'

    @staticmethod
    def model(n_class, batch_size, shape):
        batch_shape = (None, shape, shape, 3)
        inp = Input(batch_shape=batch_shape)
        norm_inp = BatchNormalization()(inp)

        img = Convolution2D(32, kernel_size=4, activation=activations.relu, padding="same", strides=1)(norm_inp)
        img = Convolution2D(32, kernel_size=4, activation=activations.relu, padding="same", strides=1)(img)
        img = MaxPooling2D(pool_size=(2, 2), strides=2)(img)

        img = Convolution2D(48, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img)
        img = Convolution2D(48, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img)
        img = MaxPooling2D(pool_size=(2, 2), strides=2)(img)

        img = Convolution2D(64, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img)
        img = Convolution2D(64, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img)
        img = Convolution2D(64, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img)
        img = MaxPooling2D(pool_size=(2, 2), strides=2)(img)

        img = Convolution2D(128, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img)
        img = Convolution2D(128, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img)
        img = Convolution2D(128, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img)
        img = GlobalMaxPool2D()(img)

        img = Dropout(rate=0.2)(img)

        img = Dense(128, activation=activations.relu)(img)
        dense = Dense(n_class, activation=activations.softmax)(img)

        model = models.Model(inputs=inp, outputs=dense)
        opt = optimizers.SGD(lr=1e-6, momentum=0.9)

        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        model.summary()
        return model
