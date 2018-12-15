import numpy as np


def glove_embedding(data_file):
    index = {}
    f = open(data_file)
    for line in f:
        values = line.split(" ")
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        index[word] = coefs
    f.close()
    return index
