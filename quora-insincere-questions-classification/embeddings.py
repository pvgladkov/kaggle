import numpy as np
import io
from collections import defaultdict


def _embedding(data_file, size=300):
    index = {}
    f = io.open(data_file, encoding="utf8", errors='ignore')
    for line in f:
        values = line.split(" ")
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        if coefs.shape != (size, ):
            continue
        index[word] = coefs
    f.close()
    return index


def _transform(embeddings_index, embed_size):
    matrix = defaultdict(lambda: np.zeros(embed_size, dtype='float32'))
    matrix.update({i: embeddings_index[w] for i, w in enumerate(embeddings_index.keys())})
    index = {w: i for i, w in enumerate(embeddings_index.keys())}
    return matrix, index


def get_embedding_matrix_glove(data_path, embed_size):
    embeddings_index = _embedding(data_path + '/glove.840B.300d/glove.840B.300d.txt')
    return _transform(embeddings_index, embed_size)


def get_embedding_matrix_fasttext(data_path, embed_size):
    embeddings_index = _embedding(data_path + '/wiki-news-300d-1M/wiki-news-300d-1M.vec')
    return _transform(embeddings_index, embed_size)


def get_embedding_matrix_para(data_path, embed_size):
    embeddings_index = _embedding(data_path + '/paragram_300_sl999/paragram_300_sl999.txt')
    return _transform(embeddings_index, embed_size)


def get_embedding_matrices(data_path, embed_size):

    embeddings_index_1 = _embedding(data_path + '/glove.840B.300d/glove.840B.300d.txt')
    embeddings_index_2 = _embedding(data_path + '/wiki-news-300d-1M/wiki-news-300d-1M.vec')
    embeddings_index_3 = _embedding(data_path + '/paragram_300_sl999/paragram_300_sl999.txt')

    vocab = set(embeddings_index_1.keys()) | set(embeddings_index_2.keys()) | set(embeddings_index_3.keys())
    index = {w: i for i, w in enumerate(vocab)}

    matrix_1 = defaultdict(lambda: np.zeros(embed_size, dtype='float32'))
    matrix_1.update({i: embeddings_index_1.get(w, np.zeros(embed_size, dtype='float32')) for w, i in index.items()})

    matrix_2 = defaultdict(lambda: np.zeros(embed_size, dtype='float32'))
    matrix_2.update({i: embeddings_index_2.get(w, np.zeros(embed_size, dtype='float32')) for w, i in index.items()})

    matrix_3 = defaultdict(lambda: np.zeros(embed_size, dtype='float32'))
    matrix_3.update({i: embeddings_index_3.get(w, np.zeros(embed_size, dtype='float32')) for w, i in index.items()})

    return matrix_1, matrix_2, matrix_3, index


def get_embedding_matrices_normalized(data_path, embed_size):

    embeddings_index_1 = _embedding(data_path + '/glove.840B.300d/glove.840B.300d.txt')
    embeddings_index_2 = _embedding(data_path + '/wiki-news-300d-1M/wiki-news-300d-1M.vec')
    embeddings_index_3 = _embedding(data_path + '/paragram_300_sl999/paragram_300_sl999.txt')

    vocab = set(embeddings_index_1.keys()) | set(embeddings_index_2.keys()) | set(embeddings_index_3.keys())
    index = {w: i for i, w in enumerate(vocab)}

    matrix_1 = t(embeddings_index_1, index, embed_size)

    matrix_2 = t(embeddings_index_2, index, embed_size)

    matrix_3 = t(embeddings_index_3, index, embed_size)

    return matrix_1, matrix_2, matrix_3, index


def t(e_matrix, vocab2idx, embed_size):
    m = {}
    indices_to_normalize = []
    indices_to_zero = []
    for word, i in vocab2idx.items():
        v = e_matrix.get(word)
        if v is not None:
            m[i] = v
            indices_to_normalize.append(i)
        else:
            m[i] = np.zeros(embed_size, dtype='float32')
            indices_to_zero.append(i)

    matrix = np.stack(m.values())
    return normalize_embeddings(matrix, indices_to_normalize, indices_to_zero)


def normalize_embeddings(embeddings, indices_to_normalize, indices_to_zero):
    if len(indices_to_normalize) > 0:
        embeddings = embeddings - embeddings[indices_to_normalize, :].mean(0)
    if len(indices_to_zero) > 0:
        embeddings[indices_to_zero, :] = 0
    return embeddings
