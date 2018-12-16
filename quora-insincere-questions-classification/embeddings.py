import numpy as np
import io
from collections import defaultdict


def _embedding(data_file):
    index = {}
    f = io.open(data_file, encoding="utf8", errors='ignore')
    for line in f:
        values = line.split(" ")
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
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
