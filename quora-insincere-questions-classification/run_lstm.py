import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from models import lstm_model2, lstm_model_dme, lstm_model_dme_4
from embeddings import (get_embedding_matrices, get_embedding_matrices_normalized, clean_text,
                         clean_numbers, replace_typical_misspell, get_embedding_matrices_normalized_4)

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import Sequence
from keras.callbacks import Callback, TensorBoard, ReduceLROnPlateau
from metrics import fmeasure_th

embed_size = 300
max_len = 30

# single, cat, proj
MODE = 'proj'


class TrainSequence(Sequence):
    def __init__(self, x_train, y_train, batch_size, emb_matrices, shuffle=False, mode='cat'):
        self.x, self.y = x_train, y_train
        self.batch_size = batch_size
        self.matrices = emb_matrices
        self.shuffle = shuffle
        self.mode = mode

    def __len__(self):
        return math.ceil(1.0 * len(self.x) / self.batch_size)

    def seq_to_array(self, text_seq):
        if self.mode in {'cat', 'single'}:
            embeds = [np.concatenate([self._matrix_value(matrix, w) for matrix in self.matrices]) for w in text_seq]
            return np.array(embeds)
        if self.mode == 'proj':
            embs = {}
            for i, matrix in enumerate(self.matrices):
                embs[i] = [self._matrix_value(matrix, w) for w in text_seq]
            return list(embs.values())

    @staticmethod
    def _matrix_value(matrix, wi):
        if wi == -1:
            return np.zeros(matrix.shape[1], dtype='float32')
        return matrix[wi]

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.mode in {'cat', 'single'}:
            return np.array([self.seq_to_array(s) for s in batch_x]), np.array(batch_y)
        if self.mode == 'proj':
            embs = {}
            for i, matrix in enumerate(self.matrices):
                embs[i] = []
            for s in batch_x:
                seq_t = self.seq_to_array(s)
                for i, seq in enumerate(seq_t):
                    embs[i].append(seq)
            return list(np.array(v) for v in embs.values()), np.array(batch_y)

    def on_epoch_end(self):
        if self.shuffle:
            index = np.arange(len(self.x))
            i = np.random.permutation(index)
            self.x, self.y = self.x[i], self.y[i]


class PredictSequence(TrainSequence):
    def __init__(self, x_test, batch_size, emb_matrices, mode='cat'):
        fake_y = np.zeros((len(x_test),))
        super(PredictSequence, self).__init__(x_test, fake_y, batch_size, emb_matrices, False, mode)

    def __getitem__(self, idx):
        x, _ = super(PredictSequence, self).__getitem__(idx)
        return x


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
    test_df = pd.read_csv(data_path + '/test.csv')

    train_df["question_text"] = train_df["question_text"].str.lower()
    test_df["question_text"] = test_df["question_text"].str.lower()

    train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x))
    test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))

    train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_numbers(x))
    test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_numbers(x))

    train_df["question_text"] = train_df["question_text"].apply(lambda x: replace_typical_misspell(x))
    test_df["question_text"] = test_df["question_text"].apply(lambda x: replace_typical_misspell(x))

    train_df, val_df = train_test_split(train_df, test_size=0.08, random_state=100)
    print('train.shape={}, val.shape={}'.format(train_df.shape, val_df.shape))

    train_X = train_df["question_text"].fillna("_##_").values
    val_X = val_df["question_text"].fillna("_##_").values
    test_X = test_df["question_text"].fillna("_##_").values

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(list(train_X) + list(val_X) + list(test_X))

    print('load embeddings')
    embedding_matrix_1, embedding_matrix_2, embedding_matrix_3, embedding_matrix_4, _ = get_embedding_matrices_normalized_4(data_path, embed_size, tokenizer.word_index)

    train_X = tokenizer.texts_to_sequences(train_X)
    val_X = tokenizer.texts_to_sequences(val_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    train_X = pad_sequences(train_X, maxlen=max_len, value=-1)
    val_X = pad_sequences(val_X, maxlen=max_len, value=-1)
    test_X = pad_sequences(test_X, maxlen=max_len, value=-1)

    train_y = np.array(train_df["target"])
    val_y = np.array(val_df["target"])

    num_embeds = 1
    if MODE in {'cat', 'proj'}:
        embeds = [embedding_matrix_1, embedding_matrix_2, embedding_matrix_3, embedding_matrix_4]
        num_embeds = len(embeds)
    elif MODE == 'single':
        embeds = [embedding_matrix_1]
    else:
        embeds = None

    train_generator = TrainSequence(train_X, train_y, 256, embeds, shuffle=True, mode=MODE)
    val_generator = TrainSequence(val_X, val_y, 256, embeds, mode=MODE)

    if MODE == 'single':
        model = lstm_model2(max_len, embed_size)
    elif MODE == 'cat':
        model = lstm_model2(max_len, num_embeds*embed_size)
    else:
        model = lstm_model_dme_4(max_len, embed_size)


    def init_callbacks():
        reduce_lr = ReduceLROnPlateau()
        tb = TensorBoard(log_dir='./logs', write_grads=True)
        return [F1ScoreCallback(val_generator, val_y), tb, reduce_lr]

    model.summary()
    model.fit_generator(train_generator, epochs=2, steps_per_epoch=len(train_generator),
                        validation_data=val_generator, validation_steps=len(val_generator), verbose=True,
                        callbacks=init_callbacks())

    y_pred = model.predict_generator(val_generator, steps=len(val_generator))
    score, thresh = fmeasure_th(val_y, y_pred)
    print('\nf1 best_score={}, best_thresh={} \n'.format(score, thresh))

    test_generator = PredictSequence(test_X, 256, embeds, mode=MODE)
    y_test = model.predict_generator(test_generator, steps=len(test_generator))

    pred_test_y = (y_test > thresh).astype(int)
    test_df = pd.read_csv(data_path + "/test.csv", usecols=["qid"])
    out_df = pd.DataFrame({"qid": test_df["qid"].values})
    out_df['prediction'] = pred_test_y
    out_df.to_csv(data_path + "/submission.csv", index=False)