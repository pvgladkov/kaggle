# coding: utf-8

from keras.applications import resnet50, VGG16, VGG19

from keras import activations, models, optimizers, losses
from keras.layers import Dropout, Dense, Flatten, GlobalAveragePooling2D, Reshape, Input


class ResNet50(object):

    name = 'resnet50'

    @staticmethod
    def model(n_class, batch_size):
        model = resnet50.ResNet50(input_shape=(256, 256, 3), classes=10, weights=None)
        opt = optimizers.Adam()
        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        model.summary()
        return model


class ResNet50XfrLearn(object):

    name = 'resnet50xfr'

    @staticmethod
    def model(n_class, batch_size, shape):

        model = resnet50.ResNet50(include_top=False, input_shape=(shape, shape, 3), pooling='avg')

        input_image = Input(shape=(shape, shape, 3))
        x = input_image
        x = model(x)
        # x = GlobalAveragePooling2D()(x)
        x = Reshape((-1,))(x)

        x = Dense(512, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(n_class, activation='softmax')(x)

        model_final = models.Model(inputs=input_image, outputs=predictions)

        opt = optimizers.Adam(lr=1e-4)
        model_final.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        model_final.summary()
        return model_final


class VGG19XfrLearn(object):
    name = 'vgg19xfr'

    @staticmethod
    def model(n_class, batch_size, shape):

        model = VGG19(include_top=False, input_shape=(shape, shape, 3), pooling='avg')

        input_image = Input(shape=(shape, shape, 3))
        x = input_image
        x = model(x)
        # x = GlobalAveragePooling2D()(x)
        x = Reshape((-1,))(x)

        x = Dense(512, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(n_class, activation='softmax')(x)

        model_final = models.Model(inputs=input_image, outputs=predictions)

        opt = optimizers.Adam(lr=1e-4)
        model_final.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        model_final.summary()
        return model_final
