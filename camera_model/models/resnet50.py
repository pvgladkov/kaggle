# coding: utf-8

from keras.applications import resnet50

from keras import activations, models, optimizers, losses
from keras.layers import Dropout, Dense, Flatten, GlobalAveragePooling2D


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
    def model(n_class, batch_size):
        model = resnet50.ResNet50(include_top=False)

        for layer in model.layers:
            layer.trainable = True

        x = model.output
        x = GlobalAveragePooling2D()(x)

        x = Dense(1024, activation='relu')(x)
        predictions = Dense(n_class, activation='softmax')(x)

        model_final = models.Model(inputs=model.input, outputs=predictions)

        opt = optimizers.SGD(lr=1e-4, momentum=0.9)
        model_final.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        model_final.summary()
        return model_final
