import numpy as np
import io
import re
from collections import defaultdict

from gensim.models import KeyedVectors


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


def get_embedding_matrices(data_path, embed_size, index=None):

    embeddings_index_1 = _embedding(data_path + '/glove.840B.300d/glove.840B.300d.txt')
    embeddings_index_2 = _embedding(data_path + '/wiki-news-300d-1M/wiki-news-300d-1M.vec')
    embeddings_index_3 = _embedding(data_path + '/paragram_300_sl999/paragram_300_sl999.txt')

    if index is None:
        vocab = set(embeddings_index_1.keys()) | set(embeddings_index_2.keys()) | set(embeddings_index_3.keys())
        index = {w: i for i, w in enumerate(vocab)}

    matrix_1 = defaultdict(lambda: np.zeros(embed_size, dtype='float32'))
    matrix_1.update({i: embeddings_index_1.get(w, np.zeros(embed_size, dtype='float32')) for w, i in index.items()})

    matrix_2 = defaultdict(lambda: np.zeros(embed_size, dtype='float32'))
    matrix_2.update({i: embeddings_index_2.get(w, np.zeros(embed_size, dtype='float32')) for w, i in index.items()})

    matrix_3 = defaultdict(lambda: np.zeros(embed_size, dtype='float32'))
    matrix_3.update({i: embeddings_index_3.get(w, np.zeros(embed_size, dtype='float32')) for w, i in index.items()})

    return matrix_1, matrix_2, matrix_3, index


def get_embedding_matrices_normalized(data_path, embed_size, index=None):

    embeddings_index_1 = _embedding(data_path + '/glove.840B.300d/glove.840B.300d.txt')
    embeddings_index_2 = _embedding(data_path + '/wiki-news-300d-1M/wiki-news-300d-1M.vec')
    embeddings_index_3 = _embedding(data_path + '/paragram_300_sl999/paragram_300_sl999.txt')

    if index is None:
        vocab = set(embeddings_index_1.keys()) | set(embeddings_index_2.keys()) | set(embeddings_index_3.keys())
        index = {w: i for i, w in enumerate(vocab)}

    matrix_1 = t(embeddings_index_1, index, embed_size)
    matrix_2 = t(embeddings_index_2, index, embed_size)
    matrix_3 = t(embeddings_index_3, index, embed_size)

    return matrix_1, matrix_2, matrix_3, index


def get_embedding_matrices_normalized_4(data_path, embed_size, index):

    embeddings_index_1 = _embedding(data_path + '/glove.840B.300d/glove.840B.300d.txt')
    embeddings_index_2 = _embedding(data_path + '/wiki-news-300d-1M/wiki-news-300d-1M.vec')
    embeddings_index_3 = _embedding(data_path + '/paragram_300_sl999/paragram_300_sl999.txt')

    _path = data_path + '/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
    embeddings_index_4_w2v = KeyedVectors.load_word2vec_format(_path, binary=True)
    embeddings_index_4 = {}
    for word in embeddings_index_4_w2v.index2word:
        embeddings_index_4[word] = embeddings_index_4_w2v[word]

    matrix_1 = t(embeddings_index_1, index, embed_size)
    matrix_2 = t(embeddings_index_2, index, embed_size)
    matrix_3 = t(embeddings_index_3, index, embed_size)
    matrix_4 = t(embeddings_index_4, index, embed_size)

    return matrix_1, matrix_2, matrix_3, matrix_4, index


def t(e_matrix, vocab2idx, embed_size):
    all_e = np.stack(e_matrix.values())
    emb_mean, emb_std = all_e.mean(), all_e.std()

    m = np.random.normal(emb_mean, emb_std, (len(vocab2idx) + 1, embed_size))
    indices_to_normalize = []
    indices_to_zero = []
    for word, i in vocab2idx.items():
        v = e_matrix.get(word)
        if v is not None:
            m[i] = v
        indices_to_normalize.append(i)

    return normalize_embeddings(m, indices_to_normalize, indices_to_zero)


def normalize_embeddings(embeddings, indices_to_normalize, indices_to_zero):
    if len(indices_to_normalize) > 0:
        embeddings = embeddings - embeddings[indices_to_normalize, :].mean(0)
    if len(indices_to_zero) > 0:
        embeddings[indices_to_zero, :] = 0
    return embeddings


def build_vocab(sentences):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


def clean_text(x):
    _p = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#',
               '*', '+', '\\', '•', '~', '@', '£',
               '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â',
               '█', '½', 'à', '…',
               '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―',
               '¥', '▓', '—', '‹', '─',
               '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸',
               '¾', 'Ã', '⋅', '‘', '∞',
               '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',
               '¹', '≤', '‡', '√', '“', '”']

    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in _p:
        x = x.replace(punct, f' {punct} ')
    # for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
    #     x = x.replace(punct, '')
    return x


def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x


def _get_misspell():
    misspell_dict = {
        'colour': 'color',
        'centre': 'center',
        'didnt': 'did not',
        'doesnt': 'does not',
        'isnt': 'is not',
        'shouldnt': 'should not',
        'favourite': 'favorite',
        'travelling': 'traveling',
        'counselling': 'counseling',
        'theatre': 'theater',
        'cancelled': 'canceled',
        'labour': 'labor',
        'organisation': 'organization',
        'wwii': 'world war 2',
        'citicise': 'criticize',
        'instagram': 'social medium',
        'whatsapp': 'social medium',
        'snapchat': 'social medium'}
    misspell_re = re.compile('(%s)' % '|'.join(misspell_dict.keys()))
    return misspell_dict, misspell_re


def replace_typical_misspell(text):
    misspellings, misspellings_re = _get_misspell()

    def replace(match):
        return misspellings[match.group(0)]

    return misspellings_re.sub(replace, text)

