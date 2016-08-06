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

def init_app_id_dict(path):
    print 'start reading dict...'
    file_read = open(path, 'r')
    result = {}
    for i in range(1001):
    #for line in file_read:
        line = file_read.readline()
        items = line.split(',')
        result[items[0]] = 0
    return result

def read_app_ids(path):
    print 'start reading app_ids...'
    file_read = open(path, 'r')
    result = []
    for i in range(1001):
    #for line in file_read:
        line = file_read.readline()
        items = line.split(',')
        result.append(items[0])
    return result

def read_event_id(path):
    print 'start reading list'
    file_read = open(path, 'r')
    result = []
    for line in file_read:
        result.append(line)
    return result

def file_parse(dict, app_ids, path):
    print 'Started parsing big file....'
    cur_dict = dict.copy()
    file_read = open(path, 'r')
    file_read.readline()
    write_file = open('app_events_features.csv', 'w')
    print_header(write_file, app_ids)
    line = file_read.readline()
    item = line.split(',')
    cur_event_id = item[0]
    app_id = item[1]
    cur_dict[app_id] = 1

    i = 0
    while line <> '':
        i += 1
        line = file_read.readline()
        if line <> '':
            item = line.split(',')
            event_id = item[0]
            app_id = item[1]
            if event_id <> cur_event_id:
                struct_to_file(write_file, cur_dict, cur_event_id, app_ids)
                cur_dict = dict.copy()
                cur_event_id = event_id
            else:
                cur_dict[app_id] = 1

    struct_to_file(write_file, cur_dict, cur_event_id, app_ids)
    write_file.flush()
    write_file.close()
    return

def print_header(write_file, app_ids):
    write_file.write('event_id')
    for app_id in app_ids:
        write_file.write(',app_id_'+ str(app_id)+'_is_installed')
    write_file.write('\n')
    return

def struct_to_file(write_file, dict, event_id, app_ids):
    write_file.write(str(event_id))
    for app_id in app_ids:
        write_file.write(','+str(dict[app_id]))
    write_file.write('\n')
    return

def check_event_id_repeat(path):
    repeat_fl = False
    event_id_list = []
    read_file = open(path, 'r')
    read_file.readline()
    line = read_file.readline()
    item = line.split(',')
    cur_event_id = item[0]
    event_id_list.append(cur_event_id)
    i = 0
    while line <> '':
        i += 1
        line = read_file.readline()
        if line <> '':
            item = line.split(',')
            event_id = item[0]
            if event_id <> cur_event_id:
                if event_id not in event_id_list:
                    cur_event_id = event_id
                    event_id_list.append(cur_event_id)
                else:
                    repeat_fl = True
                    break
    return repeat_fl

if __name__ == '__main__':

    # dict = find_cols_freq('app_events.csv')

    # import operator
    # m_list = read_to_dict('sorted_dict.csv')
    # sorted_x = sorted(m_list.items(), key=operator.itemgetter(1), reverse=True)
    # tuple_to_file(sorted_x, 'sorted_dict2.csv')
    # print sorted_x

    dict = init_app_id_dict('sorted_dict.csv')
    app_ids = read_app_ids('sorted_dict.csv')

    dict = file_parse(dict, app_ids,  'app_events.csv')
    print 'Thats all falks...'
    # print event_ids

    # flag = check_event_id_repeat('app_events.csv')
    # print 'Repeats: ', flag




