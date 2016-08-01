import pandas as pd
import numpy as np

import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
import os

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
    result = {}
    for line in file_read:
        items = line.split(',')
        result[items[0]] = int(items[1])
    return result

def read_app_ids(path):
    print 'start reading dict'
    file_read = open(path, 'r')
    result = {}
    for line in file_read:
        items = line.split(',')
        result[items[0]] = 0
    return result

def read_event_id(path):
    print 'start reading list'
    file_read = open(path, 'r')
    result = []
    for line in file_read:
        result.append(line)
    return result

def init_struct(app_ids, event_ids):
    dict = {}
    for event_id in event_ids:
        dict[event_id] = app_ids.copy()
    return dict

def file_parse(dict, path):
    file_read = open(path, 'r')
    i = 0
    for line in file_read:
        i += 1
        items = line.split(',')
        dict[items[0]][items[1]] = 1
    return dict

def print_header(write_file, app_ids):
    write_file.write('event_id')
    for app_id in app_ids:
        write_file.write(',app_id_'+ str(app_id)+'_is_installed')
    write_file.write('\n')
    return

def struct_to_file(dict, app_ids):
    write_file = open('app_events_features.csv', 'w')
    print_header(write_file, app_ids)
    for key, value in dict.iteritems():
        write_file.write(str(key))
        for app_id in app_ids:
            write_file.write(','+str(value[app_id]))
        write_file.write('\n')
    write_file.flush()
    write_file.close()
    return

if __name__ == '__main__':

    # dict = find_cols_freq('app_events.csv')

    # import operator
    # m_list = read_to_dict('sorted_dict.csv')
    # sorted_x = sorted(m_list.items(), key=operator.itemgetter(1), reverse=True)
    # tuple_to_file(sorted_x, 'sorted_dict2.csv')
    # print sorted_x

    app_ids = read_app_ids('sorted_dict.csv')
    event_ids = read_event_id('event_id_list.csv')
    print event_ids


