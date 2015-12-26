
submit = False

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
import xgboost
from sklearn import cross_validation
from sklearn.neural_network import MLPClassifier

import time
import datetime

%matplotlib inline
import matplotlib.pyplot as plt

np.random.seed(0)

#Loading data
df_train = pd.read_csv('train_users_2.csv')
df_test = pd.read_csv('test_users.csv')
sessions = pd.read_csv('sessions.csv')
age_gender_bkts = pd.read_csv('age_gender_bkts.csv')

labels = df_train['country_destination'].values
df_train = df_train.drop(['country_destination'], axis=1)
id_test = df_test['id']
piv_train = df_train.shape[0]


#Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
#Removing id and date_first_booking
df_all = df_all.drop(['date_first_booking'], axis=1)
#Filling nan
df_all = df_all.fillna(-1)

#date_account_created
dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
df_all['dac_year'] = dac[:,0]
df_all['dac_month'] = dac[:,1]
df_all['dac_day'] = dac[:,2]
df_all = df_all.drop(['date_account_created'], axis=1)

#timestamp_first_active
tfa = np.vstack(df_all.timestamp_first_active.astype(str)\
                .apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_day'] = tfa[:,2]
df_all = df_all.drop(['timestamp_first_active'], axis=1)

#Age
# av = df_all.age.values
# df_all['age'] = np.where(np.logical_or(av<14, av>100), -1, av)



def add_stat(x):
    if x['age'] == -1:
        return x
    if x['gender'] == 'FEMALE':
        g_value = 'female'
    else:
        g_value = 'male'
    
    i = 0
    a_value = '100+'
    while i <= 100:
        if x['age'] >= i and x['age'] <= i+4:
            a_value = "%d-%d" % (i, i+4)
            break
        i += 5
        
    for c_value in age_gender_bkts.country_destination.unique():
        val = age_gender_bkts.population_in_thousands[(age_gender_bkts['age_bucket'] == a_value) & 
                              (age_gender_bkts['country_destination'] == c_value) &
                             (age_gender_bkts['gender'] == g_value)]
        if val.count() == 1: 
            t = np.sum(age_gender_bkts.population_in_thousands[(age_gender_bkts['age_bucket'] == a_value) 
                                                               & (age_gender_bkts['gender'] == g_value)])
            x[c_value + '_stat'] = round(int(val) / t, 2)

    return x

# по каждому юзеру заполним статистикой по полу и возрасту для всех стран
import os
if os.path.exists('all_with_country_stat.csv'):
    df_all = pd.read_csv('all_with_country_stat.csv')
else:
    for c in age_gender_bkts.country_destination.unique():
        df_all[c + '_stat'] = -1
        df_all = df_all.apply(add_stat, axis=1)
    df_all.to_csv('all_with_country_stat.csv', index=False)



df_all.insert(1, 'freak', df_all.apply(lambda x: int(x['gender']=='OTHER'), axis=1))

# diff dac tfa
def func(x):
    dac = time.mktime(datetime.datetime.strptime("%d/%d/%d" % (x['dac_day'], x['dac_month'], x['dac_year']),
                                                 "%d/%m/%Y").timetuple())
    tfa = time.mktime(datetime.datetime.strptime("%d/%d/%d" % (x['tfa_day'], x['tfa_month'], x['tfa_year']),
                                                 "%d/%m/%Y").timetuple())
    return round((dac-tfa) / 86400)

df_all.insert(1, 'diff_dac_tfa', df_all.apply(func , axis=1)) 

def add_session_event(df, field, value):
    df['user_id'] = df.id
    df = pd.merge(df, sessions[sessions[field] == value], how='left', on=('user_id'))
    df.insert(1, value, df.apply(lambda x: int(x[field]==value), axis=1))

    _f = ['action', 'action_type', 'action_detail', 'device_type', 'secs_elapsed', 'user_id']
    df = df.drop(_f, axis=1)
    df = df.drop_duplicates(subset='id')
    return df

def add_session_time(df, field, value):
    df['user_id'] = df.id
    group_sess = sessions[sessions[field] == value].groupby('user_id', as_index=False)
    times = group_sess.agg({'secs_elapsed': 'sum'})
    df_with_times = pd.merge(df, times, how='left', on=('user_id'))
    df_with_times.secs_elapsed[df_with_times.secs_elapsed.isnull()] = 0
    df_with_times[field + '_' + value + '_secs_elapsed'] = df_with_times.secs_elapsed
    _f = ['secs_elapsed', 'user_id']
    df = df_with_times.drop(_f, axis=1)
    df = df.drop_duplicates(subset='id')
    return df

def add_rel_session_time(df, field, value):
    import math
    df = add_session_time(df, field, value)
    _f = field + '_' + value + '_secs_elapsed'

    def _func(x):
        if x[_f] > 10:
            x[_f] = round(np.log(x[_f]), 2)
        else:
            x[_f] = 0
        return x

    df = df.apply(_func, axis=1)
    return df

df_all = add_session_event(df_all, 'action_type', 'booking_request')
df_all = add_session_event(df_all, 'action_type', 'message_post')
df_all = add_session_event(df_all, 'action', 'phone_verification_success')
df_all = add_session_event(df_all, 'action', 'languages_multiselect')
df_all = add_session_event(df_all, 'action', 'verify')
df_all = add_session_event(df_all, 'action', 'coupon_field_focus')
df_all = add_session_event(df_all, 'action', 'jumio_redirect')
df_all = add_session_event(df_all, 'action', 'ajax_google_translate_reviews')

df_all = add_session_event(df_all, 'action_detail', 'post_checkout_action')
df_all = add_session_event(df_all, 'action_detail', 'guest_receipt')
df_all = add_session_event(df_all, 'action_detail', 'translate_listing_reviews')
df_all = add_session_event(df_all, 'action_detail', 'apply_coupon_click_success')

df_all = add_session_event(df_all, 'action', 'apply_coupon_click')
df_all = add_session_event(df_all, 'action_type', 'partner_callback')

df_all = add_session_event(df_all, 'action_detail', 'p4')
df_all = add_session_event(df_all, 'action_detail', 'p5')
df_all = add_session_event(df_all, 'action_detail', 'your_trips')

df_all = add_rel_session_time(df_all, 'action_detail', 'view_search_results')

df_all = df_all.drop(['id'], axis=1)
df_all.shape

df_all = df_all.drop(['first_affiliate_tracked', 'first_browser', 'affiliate_provider'], axis=1)

#One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel',
             'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    if f in df_all.columns.values:
        df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
        df_all = df_all.drop([f], axis=1)
        df_all = pd.concat((df_all, df_all_dummy), axis=1)

#Splitting train and test
vals = df_all.values
X = vals[:piv_train]
le = LabelEncoder()
y = le.fit_transform(labels)   
X_test = vals[piv_train:]

# max_depth=6, learning_rate=0.3, n_estimators=25,
#                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0

# missing=-1,

xgb = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25, nthread=-1,
                    objective='multi:softprob', subsample=1, colsample_bytree=0.5, seed=0)                  

# algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=0
clf = MLPClassifier(random_state=0, max_iter=1000)

def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k=5, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def score_predictions(preds, truth, n_modes=5):
    """
    preds: pd.DataFrame
      one row for each observation, one column for each prediction.
      Columns are sorted from left to right descending in order of likelihood.
    truth: pd.Series
      one row for each obeservation.
    """
    assert(len(preds)==len(truth))
    r = pd.DataFrame(0, index=preds.index, columns=preds.columns, dtype=np.float64)
    for col in preds.columns:
        r[col] = (preds[col] == truth) * 1.0

    score = pd.Series(r.apply(ndcg_at_k, axis=1, reduce=True), name='score')
    return score

def model_score(model, train, target, metric=True):
    ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = cross_validation.train_test_split(train, target, test_size=0.2)
    
    if metric:
        probas = model.fit(ROCtrainTRN, ROCtrainTRG, eval_metric='ndcg@5').predict_proba(ROCtestTRN)
    else:
        probas = model.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
    
    cts = []
    for i in range(len(ROCtestTRN)):
        cts += [np.argsort(probas[i])[::-1][:5].tolist()]
    
    preds = pd.DataFrame(cts)
    truth = pd.Series(ROCtestTRG)

    s = score_predictions(preds, truth)
    return np.sum(s) / len(s)

# XGBClassifier: 0.832527603591 0.87871 стата по странам, все события
# 0.832149507502 лучший без ограничений на возраст subsample=1 0.87807 ss1a
# 0.831651021345 лучший 87749

# 0.831574540708 без пола
# 0.831494270041 с missing=-1
# 0.831636997968 max_depth=8, learning_rate=0.25
# 0.831888486718 max_depth=6, learning_rate=0.25 s25 0.87720
# 0.831052267667 max_depth=6, learning_rate=0.1
# 0.83134376092 max_depth=8, learning_rate=0.1
# 0.83325149416700384 max_depth=5, learning_rate=0.3 s3r5 0.87639

# 0.831497805586 без first_affiliate_tracked
# 0.831912216819 без signup_app 0.87637
# 0.831557790207 без affiliate_channel
# 0.831843414646 без affiliate_provider
# 0.831323859656 без signup_flow

# 0.83166115849 без dac_day
# 0.831902929539 без dac_day и tfa_day 0.87593
# 0.829047678728 без dac_day и tfa_day n_estimators=250
# 0.830603493262 без dac_day и tfa_day n_estimators=100
# 0.831863527023 без dac_day и tfa_day n_estimators=50
# 0.831918444295 без dac_day и tfa_day n_estimators=20
# 0.831811446356 без dac_day и tfa_day n_estimators=15

# 0.832264028647 без ограничений на возраст
# 0.83193723328 без ограничений на возраст max_depth=6, learning_rate=0.35, n_estimators=25, subsample=1

# 0.832120048709 без ограничений на возраст subsample=1 learning_rate=0.39
# 0.831768999872 subsample=1

# XGBClassifier: 0.831730234428 MLPClassifier: 0.806820675063 без ограничений на возраст + стата по странам
# MLPClassifier: 0.800235856099 0.86345 с дефолтными настройками, стата по странам max_iter=1000
# XGBClassifier: 0.832640145373 удалил 'first_affiliate_tracked', 'first_browser'
# XGBClassifier: 0.832587501259 удалил 'first_affiliate_tracked', 'first_browser', 'affiliate_provider'

print 'XGBClassifier:', model_score(xgb, X, y)
# print 'MLPClassifier:', model_score(clf, X, y, False)

# scores = []
# for depth in range(4, 10, 1):
#     for rate in range(5, 50, 5):
#         l_rate = round(float(rate)/ 100, 2)
#         model = XGBClassifier(max_depth=depth, learning_rate=l_rate, n_estimators=25, 
#                               objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)
#         score = model_score(model, X, y)
#         item = {'max_depth': depth, 'learning_rate': l_rate, 'score': score}
#         scores.append(item)
#         print depth, l_rate

# scores.sort(key=lambda x: x['score'])
# for it in scores:
#     print it

# scores_df = pd.DataFrame(scores)
# scores_df.to_csv('scores.csv')

if submit:

    xgb.fit(X, y, eval_metric='ndcg@5')

    y_pred = xgb.predict_proba(X_test)

    ids = []  #list of ids
    cts = []  #list of countries
    for i in range(len(id_test)):
        idx = id_test[i]
        ids += [idx] * 5
        cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

    sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
    sub.to_csv('sub_alls1.csv',index=False)


