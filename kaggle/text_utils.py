# coding: utf-8

import re
from nltk import SnowballStemmer


stemmer = SnowballStemmer('english')


def str_preprocess(text_string):
    """
    Нормализация текста
    :param text_string:
    :return:
    """
    # удаляем специальные символы
    text_string = unicode(text_string).lower().replace('&quot;', '"') \
        .replace('&amp;', '&').replace('&gt;', '>') \
        .replace('&lt;', '<').replace('&apos;', '\'')

    if text_string in ['nan', 'nat', 'none', '']:
        return ''
    return ' '.join([stemmer.stem(word)[0]
                     for word in re.findall(u'[a-zA-ZЁ-ё]+', text_string, flags=re.UNICODE)])