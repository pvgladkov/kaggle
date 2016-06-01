
import os

import pandas as pd
import numpy as np


data_dir = '/home/pavel/P/kaggle_data/expedia'

train_dtypes = {'site_name': np.int8, 'posa_continent': np.int8, 
                'user_location_country': np.int16, 'user_location_region': np.int16, 
                'user_location_city': np.int32, 'orig_destination_distance': np.float64, 
                'user_id': np.int32, 'is_mobile': np.bool, 'is_package': np.bool,
                'channel': np.int8, 'srch_adults_cnt': np.int8, 'srch_children_cnt': np.int8,
                'srch_rm_cnt': np.int8, 'srch_destination_id': np.int32, 
                'srch_destination_type_id': np.int8, 'is_booking': np.bool, 
                'cnt': np.int16, 'hotel_continent': np.int8, 'hotel_country': np.int16, 
                'hotel_market': np.int16, 'hotel_cluster': np.int8}

train = pd.read_csv(os.path.join(data_dir, 'train.csv'),
                    dtype=train_dtypes,
                    usecols=['date_time', 'hotel_cluster', 'is_booking', 'srch_ci', 'srch_co'])


rng = pd.date_range(start='1/1/2013', end='1/1/2016', freq='d')
ts = pd.Series(0, index=rng)
cico_df = pd.DataFrame(ts)
cico_df.columns = ['events']

booking_train = train[train['is_booking']==1]
booking_train.shape

booking_train['date'] = pd.to_datetime(booking_train.date_time, errors='raise')
booking_train = booking_train.drop(['date_time'], axis=1)

booking_train['datetime_ci'] = pd.to_datetime(booking_train.srch_ci, errors='raise')
booking_train['datetime_co'] = pd.to_datetime(booking_train.srch_co, errors='raise')

for hc in booking_train.hotel_cluster.unique():
    cico_df[hc] = 0
    cico_df[hc] = cico_df[hc].astype(np.int16)
cico_df = cico_df.drop(['events'], axis=1)

def fill_hotels(x):
    global cico_df
    cico_df.loc[(cico_df.index <= x['datetime_co']) & (cico_df.index >= x['datetime_ci']), x['hotel_cluster']] += 1

t = booking_train.apply(fill_hotels, axis=1)

cico_df.to_csv(os.path.join(data_dir, 'cluster_load.csv'))

