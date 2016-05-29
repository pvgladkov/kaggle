# coding: utf-8

import numpy as np 
import pandas as pd

import os
data_dir = '/home/pavel/P/kaggle_data/expedia'

train = pd.read_csv(os.path.join(data_dir, 'train.csv'),
                    dtype={'is_booking':bool,'srch_destination_id':np.int32, 'hotel_cluster':np.int32},
                    usecols=['srch_destination_id','is_booking','hotel_cluster'],
                    chunksize=1000000)
aggs = []

for chunk in train:
    agg = chunk.groupby(['srch_destination_id',
                         'hotel_cluster'])['is_booking'].agg(['sum','count'])
    agg.reset_index(inplace=True)
    aggs.append(agg)

aggs = pd.concat(aggs, axis=0)

CLICK_WEIGHT = 0.05
agg = aggs.groupby(['srch_destination_id','hotel_cluster']).sum().reset_index()
agg['count'] -= agg['sum']
agg = agg.rename(columns={'sum':'bookings','count':'clicks'})
agg['relevance'] = agg['bookings'] + CLICK_WEIGHT * agg['clicks']


def most_popular(group, n_max=5):
    relevance = group['relevance'].values
    hotel_cluster = group['hotel_cluster'].values
    most_popular = hotel_cluster[np.argsort(relevance)[::-1]][:n_max]
    return np.array_str(most_popular)[1:-1]


most_pop = agg.groupby(['srch_destination_id']).apply(most_popular)
most_pop = pd.DataFrame(most_pop).rename(columns={0:'hotel_cluster'})


test = pd.read_csv(os.path.join(data_dir, 'test.csv'),
                    dtype={'srch_destination_id':np.int32},
                    usecols=['srch_destination_id'],)


test = test.merge(most_pop, how='left',left_on='srch_destination_id',right_index=True)

most_pop_all = agg.groupby('hotel_cluster')['relevance'].sum().nlargest(5).index
most_pop_all = np.array_str(most_pop_all)[1:-1]


test.hotel_cluster.fillna(most_pop_all, inplace=True)

pop_set = set(most_pop_all)


def add(x):
    cl = x['hotel_cluster']
    if isinstance(cl, int):
        ids_list = [str(cl)]
    else:
        try:
            ids_list = [str(i) for i in x['hotel_cluster'].split(' ')]
        except ValueError:
            ids_list = list(pop_set)
    
    if len(ids_list) < 5:
        free = pop_set - set(ids_list)
        for i in free:
            ids_list.append(i)
            if len(ids_list) >= 5:
                break
        x['hotel_cluster'] = ' '.join(ids_list)
    return x

new_test = test.apply(add, axis=1)

new_test.hotel_cluster.to_csv('simple_stat_2.csv',header=True, index_label='id')

