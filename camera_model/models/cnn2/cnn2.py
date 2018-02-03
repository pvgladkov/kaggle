from keras import Input, activations, models, optimizers, losses
from keras.layers import (BatchNormalization, Convolution2D, MaxPooling2D, Dropout,
                          GlobalAveragePooling2D, Dense, Flatten, concatenate, Reshape)


class CNNConv18(object):
    name = 'cnn2.cnn_conv18'

    @staticmethod
    def model(n_class, batch_size, shape):
        batch_shape = (None, shape, shape, 3)

        inp = Input(batch_shape=batch_shape)
        manipulated = Input(shape=(1,))

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
        img = MaxPooling2D(pool_size=(2, 2), strides=2)(img)

        img = Convolution2D(256, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img)
        img = Convolution2D(256, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img)
        img = Convolution2D(256, kernel_size=5, activation=activations.relu, padding="same", strides=1)(img)
        img = GlobalAveragePooling2D()(img)

        img = Reshape((-1,))(img)

        x = concatenate([img, manipulated])

        x = Dense(512, activation=activations.relu)(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(128, activation=activations.relu)(x)
        prediction = Dense(n_class, activation=activations.softmax)(x)

        model = models.Model(inputs=[inp, manipulated], outputs=prediction)
        opt = optimizers.Adam(lr=1e-4)

        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        model.summary()
        return model