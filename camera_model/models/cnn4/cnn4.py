# coding: utf-8

from keras import Input, activations, models, optimizers, losses
from keras.layers import (BatchNormalization, Convolution2D, MaxPooling2D, Dropout,
                          GlobalMaxPool2D, Dense, concatenate, Reshape, GlobalAveragePooling2D)


class CNNConv10(object):
    name = 'cnn4.cnn_conv10'

    @staticmethod
    def model(n_class, batch_size, shape):
        batch_shape = (None, shape, shape, 3)

        inp1 = Input(batch_shape=batch_shape)
        inp2 = Input(batch_shape=batch_shape)
        inp3 = Input(batch_shape=batch_shape)

        manipulated = Input(shape=(1,))

        norm_inp1 = BatchNormalization()(inp1)
        norm_inp2 = BatchNormalization()(inp2)
        norm_inp3 = BatchNormalization()(inp3)

        # 1 level
        img1 = Convolution2D(32, kernel_size=4, activation=activations.relu, padding="same", strides=1)(norm_inp1)
        img1 = Convolution2D(32, kernel_size=4, activation=activations.relu, padding="same", strides=1)(img1)
        img1 = MaxPooling2D(pool_size=(2, 2), strides=2)(img1)

        img2 = Convolution2D(32, kernel_size=4, activation=activations.relu, padding="same", strides=1)(norm_inp2)
        img2 = Convolution2D(32, kernel_size=4, activation=activations.relu, padding="same", strides=1)(img2)
        img2 = MaxPooling2D(pool_size=(2, 2), strides=2)(img2)

        img3 = Convolution2D(32, kernel_size=4, activation=activations.relu, padding="same", strides=1)(norm_inp3)
        img3 = Convolution2D(32, kernel_size=4, activation=activations.relu, padding="same", strides=1)(img3)
        img3 = MaxPooling2D(pool_size=(2, 2), strides=2)(img3)

        # 2 level
        img1 = Convolution2D(48, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img1)
        img1 = Convolution2D(48, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img1)
        img1 = MaxPooling2D(pool_size=(2, 2), strides=2)(img1)

        img2 = Convolution2D(48, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img2)
        img2 = Convolution2D(48, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img2)
        img2 = MaxPooling2D(pool_size=(2, 2), strides=2)(img2)

        img3 = Convolution2D(48, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img3)
        img3 = Convolution2D(48, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img3)
        img3 = MaxPooling2D(pool_size=(2, 2), strides=2)(img3)

        # 3 level
        img1 = Convolution2D(64, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img1)
        img1 = Convolution2D(64, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img1)
        img1 = Convolution2D(64, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img1)
        img1 = MaxPooling2D(pool_size=(2, 2), strides=2)(img1)

        img2 = Convolution2D(64, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img2)
        img2 = Convolution2D(64, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img2)
        img2 = Convolution2D(64, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img2)
        img2 = MaxPooling2D(pool_size=(2, 2), strides=2)(img2)

        img3 = Convolution2D(64, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img3)
        img3 = Convolution2D(64, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img3)
        img3 = Convolution2D(64, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img3)
        img3 = MaxPooling2D(pool_size=(2, 2), strides=2)(img3)

        # 4 level
        img1 = Convolution2D(128, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img1)
        img1 = Convolution2D(128, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img1)
        img1 = Convolution2D(128, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img1)
        img1 = GlobalAveragePooling2D()(img1)

        img2 = Convolution2D(128, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img2)
        img2 = Convolution2D(128, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img2)
        img2 = Convolution2D(128, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img2)
        img2 = GlobalAveragePooling2D()(img2)

        img3 = Convolution2D(128, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img3)
        img3 = Convolution2D(128, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img3)
        img3 = Convolution2D(128, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img3)
        img3 = GlobalAveragePooling2D()(img3)

        # concat
        img = concatenate([img1, img2, img3])
        img = Reshape((-1,))(img)

        img = Dense(256, activation=activations.relu)(img)
        img = Dropout(rate=0.2)(img)

        x = concatenate([img, manipulated])

        x = Dense(128, activation=activations.relu)(x)
        prediction = Dense(n_class, activation=activations.softmax)(x)

        model = models.Model(inputs=[inp1, inp2, inp3, manipulated], outputs=prediction)
        opt = optimizers.Adam(lr=0.5*1e-4)

        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        model.summary()
        return model


