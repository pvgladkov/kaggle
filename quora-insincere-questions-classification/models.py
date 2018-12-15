from keras.models import Sequential, Model
from keras.layers import Dense, Bidirectional, LSTM, Input, Embedding
from metrics import fmeasure


def lstm_model(max_len, embedding_size):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(max_len, embedding_size)))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', fmeasure])
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
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', fmeasure])

    return model
