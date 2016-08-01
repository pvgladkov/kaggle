import pandas as pd
import numpy as np

import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
import os


def parse_app_events(df):

    def get_col_names(df, col):
        most_freq = df.groupby([col]).count().iloc[:,0:1]
        most_freq.columns = ['count']
        most_freq = most_freq.sort_values(['count'], ascending=False).head(20)
        most_freq_list = most_freq.index.tolist()
#         most_freq.index = range(len(most_freq.index))
#         most_freq[col] = pd.Series(most_freq_list, index=most_freq.index)

        cols = []
        for col in most_freq_list:
            cur_col = 'app_id_' + str(col) + '_is_installed'
            cols.append(cur_col)
        return cols
    columns = get_col_names(df, 'app_id')
    print len(columns)
    print columns

#     result = []
#     columns = get_col_names(df)
#     for i in range(10):#range(len(app_events.index))):
#         item = []
    return

def find_cols_freq(path):
    file_read = open(path, 'r')
    i = 0
    dict = {}
    for line in file_read:
        i += 1
        items = line.split(',')
        if items[1] in dict.keys():
            dict[items[1]] += 1
        else:
            dict[items[1]] = 1
    return dict

def tuple_to_file(m_list, path):
    file_write = open(path, 'w')
    for item in m_list:
        file_write.write(str(item[0]) + ',' + str(item[1])+'\n')
    file_write.flush()
    file_write.close()
    return

def read_to_dict(path):
    print 'start reading dict'
    file_read = open(path, 'r')
    #result =[]
    result = {}
    for line in file_read:
        items = line.split(',')
        result[items[0]] = int(items[1])+1
        #result.append((items[0], int(items[1])+1))
    return result


if __name__ == '__main__':

    # dict = find_cols_freq('app_events.csv')

    import operator
    m_list = read_to_dict('sorted_dict.csv')
    sorted_x = sorted(m_list.items(), key=operator.itemgetter(1), reverse=True)
    tuple_to_file(sorted_x, 'sorted_dict2.csv')
    print sorted_x

