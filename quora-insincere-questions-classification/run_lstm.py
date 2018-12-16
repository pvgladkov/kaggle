import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from models import lstm_model2, lstm_model
from embeddings import get_embedding_matrix_glove, get_embedding_matrix_para, get_embedding_matrix_fasttext

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import Sequence
from keras.callbacks import Callback
from metrics import fmeasure_th

embed_size = 300
max_len = 30


class TrainSequence(Sequence):
    def __init__(self, x_train, y_train, batch_size, emb_matrix, shuffle=False):
        self.x, self.y = x_train, y_train
        self.batch_size = batch_size
        self.matrix = emb_matrix
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(1.0 * len(self.x) / self.batch_size)

    def seq_to_array(self, text_seq):
        embeds = [self.matrix[x] for x in text_seq]
        return np.array(embeds)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([self.seq_to_array(t) for t in batch_x]), np.array(batch_y)

    def on_epoch_end(self):
        if self.shuffle:
            index = np.arange(len(self.x))
            i = np.random.permutation(index)
            self.x, self.y = self.x[i], self.y[i]


class F1ScoreCallback(Callback):
    def __init__(self, test_data, y):
        super(F1ScoreCallback, self).__init__()
        self.test_data = test_data
        self.y = y

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict_generator(self.test_data, steps=len(self.test_data))
        best_score, best_thresh = fmeasure_th(self.y, y_pred)
        print('\nf1 best_score={}, best_thresh={} \n'.format(best_score, best_thresh))


if __name__ == '__main__':

    data_path = './'

    train_df = pd.read_csv(data_path + "/train.csv")
    train_df, val_df = train_test_split(train_df, test_size=0.08, random_state=100)
    print('train.shape={}, val.shape={}'.format(train_df.shape, val_df.shape))

    train_X = train_df["question_text"].fillna("_##_").values
    val_X = val_df["question_text"].fillna("_##_").values

    print('load embeddings')
    embedding_matrix, word_index = get_embedding_matrix_glove(data_path, embed_size)

    tokenizer = Tokenizer()
    tokenizer.word_index = word_index

    train_X = tokenizer.texts_to_sequences(train_X)
    val_X = tokenizer.texts_to_sequences(val_X)

    train_X = pad_sequences(train_X, maxlen=max_len, value=-1)
    val_X = pad_sequences(val_X, maxlen=max_len, value=-1)

    train_y = np.array(train_df["target"])
    val_y = np.array(val_df["target"])

    train_generator = TrainSequence(train_X, train_y, 256, embedding_matrix, shuffle=True)
    val_generator = TrainSequence(val_X, val_y, 256, embedding_matrix)

    model = lstm_model2(max_len, embedding_matrix[0].shape[0])
    # model = lstm_model(max_len, embed_size)
    model.summary()
    model.fit_generator(train_generator, epochs=20, steps_per_epoch=1000,
                        validation_data=val_generator, validation_steps=len(val_generator), verbose=True,
                        callbacks=[F1ScoreCallback(val_generator, val_y)])

    y_pred = model.predict_generator(val_generator, steps=len(val_generator))
    score, thresh = fmeasure_th(val_y, y_pred)
    print('\nf1 best_score={}, best_thresh={} \n'.format(score, thresh))