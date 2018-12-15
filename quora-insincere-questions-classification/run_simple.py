import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from models import lstm_model
from embeddings import glove_embedding


def text_to_array(text):
    empty_emb = np.zeros(300)
    text = text[:-1].split()[:30]
    embeds = [embeddings_index.get(x, empty_emb) for x in text]
    embeds += [empty_emb] * (30 - len(embeds))
    return np.array(embeds)


def batch_gen_train(df):
    batch_size = 128
    n_batches = math.ceil(len(df) / batch_size)
    while True:
        df = df.sample(frac=1.)
        for i in range(n_batches):
            texts = df.iloc[i * batch_size:(i + 1) * batch_size, 1]
            text_arr = np.array([text_to_array(text) for text in texts])
            yield text_arr, np.array(df["target"][i * batch_size:(i + 1) * batch_size])


def batch_gen_test(df):
    batch_size = 256
    n_batches = math.ceil(len(df) / batch_size)
    for i in range(n_batches):
        texts = df.iloc[i * batch_size:(i + 1) * batch_size, 1]
        text_arr = np.array([text_to_array(text) for text in texts])
        yield text_arr


if __name__ == '__main__':

    data_path = './'

    train_df = pd.read_csv(data_path + "/train.csv")
    train_df, val_df = train_test_split(train_df, test_size=0.1)

    print('train.shape={}, val.shape={}'.format(train_df.shape, val_df.shape))

    embeddings_index = glove_embedding(data_path + '/glove.840B.300d/glove.840B.300d.txt')
    print('Found %s word vectors.' % len(embeddings_index))

    val_vects = np.array([text_to_array(X_text) for X_text in val_df["question_text"]])
    val_y = np.array(val_df["target"])

    model = lstm_model()
    mg = batch_gen_train(train_df)
    model.fit_generator(mg, epochs=20, steps_per_epoch=1000,
                        validation_data=(val_vects, val_y), verbose=True)

    test_df = pd.read_csv(data_path + "/test.csv")

    all_preds = []
    for x in batch_gen_test(test_df):
        all_preds.extend(model.predict(x).flatten())

    y_te = (np.array(all_preds) > 0.5).astype(np.int)

    submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})
    submit_df.to_csv("submission.csv", index=False)
