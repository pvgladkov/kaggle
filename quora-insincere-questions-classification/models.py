from keras.models import Sequential, Model
from keras.layers import Dense, Bidirectional, LSTM, Input, Embedding, Dropout, add, multiply
from layers import Attention


def lstm_model(max_len, embedding_size):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(max_len, embedding_size)))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def lstm_model2(max_len, embedding_size):
    """
    :param max_len: sentence length
    :param embedding_size: embedding vector size
    :return:
    """
    inp = Input(shape=(max_len, embedding_size))
    x = Bidirectional(LSTM(128, return_sequences=True))(inp)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(rate=0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def lstm_model_dme(max_len, embedding_size):
    inp_1 = Input(shape=(max_len, embedding_size))
    inp_2 = Input(shape=(max_len, embedding_size))
    inp_3 = Input(shape=(max_len, embedding_size))

    x_1 = Dense(256)(inp_1)
    x_2 = Dense(256)(inp_2)
    x_3 = Dense(256)(inp_3)

    att_1 = Attention(max_len)(x_1)
    att_2 = Attention(max_len)(x_2)
    att_3 = Attention(max_len)(x_3)

    att_x_1 = multiply([x_1, att_1])
    att_x_2 = multiply([x_2, att_2])
    att_x_3 = multiply([x_3, att_3])

    inp = add([att_x_1, att_x_2, att_x_3])

    x = Bidirectional(LSTM(128, return_sequences=True))(inp)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(rate=0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[inp_1, inp_2, inp_3], outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
