
submit = False

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost
from sklearn import cross_validation
import time
import datetime

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost.sklearn import XGBClassifier

np.random.seed(0)

#Loading data
df_train = pd.read_csv('train_users_2.csv')
df_test = pd.read_csv('test_users.csv')
sessions = pd.read_csv('sessions.csv')
age_gender_bkts = pd.read_csv('age_gender_bkts.csv')
countries = pd.read_csv('countries.csv')

len_df_test = df_test.shape[0]

df_test['country_destination'] = '--'

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
# df_all.loc[df_all.age < 15, 'age'] = 15
# df_all.loc[df_all.age > 100, 'age'] = 100



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

filename = 'all_with_country_stat_new_2.csv'
if os.path.exists(filename):
    df_all = pd.read_csv(filename)
else:
    for c in age_gender_bkts.country_destination.unique():
        df_all[c + '_stat'] = -1
    df_all = df_all.apply(add_stat, axis=1)
    df_all.to_csv(filename, index=False)

# for c_value in age_gender_bkts.country_destination.unique():
#     val = age_gender_bkts.population_in_thousands[(age_gender_bkts['age_bucket'] == '25-29') & 
#                                                   (age_gender_bkts['country_destination'] == c_value) & 
#                                                   (age_gender_bkts['gender'] == 'male')]
#     if val.count() == 1: 
#         t = np.sum(age_gender_bkts.population_in_thousands[(age_gender_bkts['age_bucket'] == '25-29') & 
#                                                            (age_gender_bkts['gender'] == 'male')])
#         df_all[c_value + '_stat'][df_all['age'] == -1] = round(int(val) / t, 2)

# df_train = df_train[(df_train['country_destination'] != 'NDF') | (df_train.index % 6 > 0)]
# df_train = df_train[((df_train['country_destination']=='--')) | 
#                     ((df_train['country_destination']!='NDF')&(df_train['age']>0)&(df_train['gender']!='-unknown-')) | 
#                     ((df_train['country_destination'] == 'NDF'))]

# df_all = df_all[((df_all['country_destination']=='--')) | 
#                 ((df_all['country_destination']!='NDF')&(df_all['age']>0)&(df_all['gender']!='-unknown-')) | 
#                 ((df_all['country_destination'] == 'NDF'))]

# возраст заменим средним
# df_all.age[df_all['age'] > 150] = 26

# добавим еще языковой признак для стран
df_all['lang_dist'] = 0
def add_language_d(x):
    if x['language'] == 'en':
        x['lang_dist'] = 1
    if x['language'] == 'de':
        x['lang_dist'] = 0.27
    if x['language'] == 'es':
        x['lang_dist'] = 0.08
    if x['language'] == 'fr':
        x['lang_dist'] = 0.08
    if x['language'] == 'it':
        x['lang_dist'] = 0.11
    if x['language'] == 'nl':
        x['lang_dist'] = 0.37
    if x['language'] == 'pt':
        x['lang_dist'] = 0.05
    return x

df_all = df_all.apply(add_language_d, axis=1)

# признак что пользуется apple
df_all['mac_user'] = 0
def add_mac_user(x):
    if x['first_device_type'] in ['Mac Desktop', 'iPhone', 'iPad']:
        x['mac_user'] = 1
    return x

# df_all = df_all.apply(add_mac_user, axis=1)

# признак что указал пол "другой"
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

def get_ag_session_data():
    grpby = sessions.groupby(['user_id'])['secs_elapsed'].sum().reset_index()
    grpby.columns = ['user_id','secs_elapsed']
    action_type = pd.pivot_table(sessions, index = ['user_id'],columns = ['action_type'],
                                 values = 'action',aggfunc=len,fill_value=0).reset_index()
    action_type = action_type.drop(['booking_response'],axis=1)
    device_type = pd.pivot_table(sessions, index = ['user_id'],columns = ['device_type'],
                                 values = 'action',aggfunc=len,fill_value=0).reset_index()
    
    sessions_data = pd.merge(action_type,device_type,on='user_id',how='inner')
    sessions_data = pd.merge(sessions_data,grpby,on='user_id',how='inner')
    
    sessions_data = sessions_data.drop(['secs_elapsed', 'booking_request', 'click', 'data', 'message_post',
                                        'modify', 'partner_callback', 'submit', 'view'], axis=1)
    
    return sessions_data

def add_ag_session_data(df):
    df['user_id'] = df.id
    ag_session_data = get_ag_session_data()
    df = pd.merge(df, ag_session_data, how='left', on=('user_id'))
    df = df.drop(['user_id'], axis=1)
    df = df.fillna(-1)
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

# df_all = add_ag_session_data(df_all)



df_all = df_all.drop(['id'], axis=1)
df_all.shape

df_all[df_all['country_destination'] == 'IT'].to_csv('IT.csv')

#One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel',
             'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    if f in df_all.columns.values:
        df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
        df_all = df_all.drop([f], axis=1)
        df_all = pd.concat((df_all, df_all_dummy), axis=1)

df_all = df_all[df_all['country_destination'] != 'AU']
df_train = df_train[df_train['country_destination'] != 'AU']

df_all = df_all[df_all['country_destination'] != 'PT']
df_train = df_train[df_train['country_destination'] != 'PT']

df_all = df_all[df_all['country_destination'] != 'NL']
df_train = df_train[df_train['country_destination'] != 'NL']

df_all = df_all[df_all['country_destination'] != 'DE']
df_train = df_train[df_train['country_destination'] != 'DE']

df_all = df_all[df_all['country_destination'] != 'CA']
df_train = df_train[df_train['country_destination'] != 'CA']

df_all = df_all[df_all['country_destination'] != 'ES']
df_train = df_train[df_train['country_destination'] != 'ES']

df_all = df_all[df_all['country_destination'] != 'GB']
df_train = df_train[df_train['country_destination'] != 'GB']



labels = df_train['country_destination'].values
id_test = df_test['id']
piv_train = df_train.shape[0]

all_AU = df_all[df_all['country_destination'] == 'AU'].drop(['country_destination'], axis=1).values
all_CA = df_all[df_all['country_destination'] == 'CA'].drop(['country_destination'], axis=1).values
all_DE = df_all[df_all['country_destination'] == 'DE'].drop(['country_destination'], axis=1).values
all_ES = df_all[df_all['country_destination'] == 'ES'].drop(['country_destination'], axis=1).values
all_FR = df_all[df_all['country_destination'] == 'FR'].drop(['country_destination'], axis=1).values
all_GB = df_all[df_all['country_destination'] == 'GB'].drop(['country_destination'], axis=1).values
all_IT = df_all[df_all['country_destination'] == 'IT'].drop(['country_destination'], axis=1).values
all_NL = df_all[df_all['country_destination'] == 'NL'].drop(['country_destination'], axis=1).values
all_PT = df_all[df_all['country_destination'] == 'PT'].drop(['country_destination'], axis=1).values
all_US = df_all[df_all['country_destination'] == 'US'].drop(['country_destination'], axis=1).values
all_other = df_all[df_all['country_destination'] == 'other'].drop(['country_destination'], axis=1).values

df_all = df_all.drop(['country_destination'], axis=1)

#Splitting train and test
vals = df_all.values
X = vals[:piv_train]

#########
# X  = np.append(X, all_AU.repeat(1, axis=0), axis=0)
# labels = np.append(labels, 1*len(all_AU)*['AU'])

# X  = np.append(X, all_CA.repeat(5, axis=0), axis=0)
# labels = np.append(labels, 5*len(all_CA)*['CA'])

# X  = np.append(X, all_DE.repeat(5, axis=0), axis=0)
# labels = np.append(labels, 5*len(all_DE)*['DE'])

# X  = np.append(X, all_ES.repeat(5, axis=0), axis=0)
# labels = np.append(labels, 5*len(all_ES)*['ES'])

# X  = np.append(X, all_FR.repeat(2, axis=0), axis=0)
# labels = np.append(labels, 2*len(all_FR)*['FR'])

# X  = np.append(X, all_GB.repeat(5, axis=0), axis=0)
# labels = np.append(labels, 5*len(all_GB)*['GB'])

# X  = np.append(X, all_IT.repeat(5, axis=0), axis=0)
# labels = np.append(labels, 5*len(all_IT)*['IT'])

# X  = np.append(X, all_NL.repeat(5, axis=0), axis=0)
# labels = np.append(labels, 5*len(all_NL)*['NL'])

# X  = np.append(X, all_PT.repeat(5, axis=0), axis=0)
# labels = np.append(labels, 5*len(all_PT)*['PT'])

# X  = np.append(X, all_US.repeat(2, axis=0), axis=0)
# labels = np.append(labels, 2*len(all_US)*['US'])

# X  = np.append(X, all_other.repeat(2, axis=0), axis=0)
# labels = np.append(labels, 2*len(all_other)*['other'])

le = LabelEncoder()
y = le.fit_transform(labels)   
X_test = vals[piv_train:]

# max_depth=6, learning_rate=0.3, n_estimators=25,
#                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0

# missing=-1,

fxgb = lambda : XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25, nthread=-1,
                              objective='multi:softprob', subsample=1, colsample_bytree=0.5, seed=0)                  

# algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=0
nnc = MLPClassifier(random_state=0, max_iter=1000)

gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

bgc = BaggingClassifier(base_estimator=fxgb())

FCLSF = lambda: [RandomForestClassifier(n_estimators=25, max_depth=6, n_jobs=-1, criterion='gini'),
                 RandomForestClassifier(n_estimators=25, max_depth=6, n_jobs=-1, criterion='entropy'),
                 ExtraTreesClassifier(n_estimators=25, max_depth=6, n_jobs=-1, criterion='gini'),
                 ExtraTreesClassifier(n_estimators=25, max_depth=6, n_jobs=-1, criterion='entropy')
        #KNeighborsClassifier(n_jobs=-1),
#         XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25, nthread=-1,
#                       objective='multi:softprob', subsample=1, colsample_bytree=0.5, seed=0),
        #MLPClassifier(random_state=0, max_iter=1000)
        ]

from sklearn.cross_validation import StratifiedKFold, ShuffleSplit

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
    found = False
    for col in preds.columns:
        r[col] = (preds[col] == truth) * 1.0
        
    score = pd.Series(r.apply(ndcg_at_k, axis=1, reduce=True), name='score')
    return score

def _score(ROCtestTRN, ROCtestTRG, probas):
    cts = []
    for i in range(len(ROCtestTRN)):
        cts += [np.argsort(probas[i])[::-1][:5].tolist()]
    
    preds = pd.DataFrame(cts)
    truth = pd.Series(ROCtestTRG)
    
    preds.to_csv('preds_AU.csv', header=False)
    truth.to_csv('truth_AU.csv', header=False)
    
    s = score_predictions(preds, truth)
    return np.sum(s) / len(s)
    

def model_score(model_name, train, target, metric=True):
    
    if model_name == 'xgb':
        model = fxgb()
    else:
        raise Exception('invalid model')
    
    skf = ShuffleSplit(target.shape[0], n_iter=3, test_size=0.2, random_state=0)
    scores = []
    
    for train_index, test_index in skf:
        ROCtrainTRN, ROCtestTRN = train[train_index], train[test_index]
        ROCtrainTRG, ROCtestTRG = target[train_index], target[test_index]
        if metric:
            model.fit(ROCtrainTRN, ROCtrainTRG, eval_metric='ndcg@5')
        else:
            model.fit(ROCtrainTRN, ROCtrainTRG)

        probas = model.predict_proba(ROCtestTRN)
        s = _score(ROCtestTRN, ROCtestTRG, probas)
        print s
        scores.append(s)
        
    return np.array(scores).mean()

N_FOLDS = 2

def dataset_blend(X, y, X_submission, n_folds):
    """ 
    получить датасет
    """
    skf = list(StratifiedKFold(y, n_folds))
    
    clsf = FCLSF()
    
    dataset_blend_train = np.zeros((X.shape[0], len(clsf)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clsf)))

    for j, clf in enumerate(clsf):
        print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:,1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

    dataset_blend_train = np.append(X, dataset_blend_train, axis=1)
    dataset_blend_test = np.append(X_submission, dataset_blend_test, axis=1)
    
    return dataset_blend_train, dataset_blend_test

def blending(train, test, target):
    clf = fxgb()
    clf.fit(train, target)
    probas = clf.predict_proba(test)
    return probas

def score_stacking(train, target):
    """ 
    результат модели
    """
    n_folds = N_FOLDS
    
    # кросс-валидация на нескольких сэмплах
    skf = ShuffleSplit(target.shape[0], n_iter=3, test_size=0.2, random_state=0)
    scores = []
    
    for train_index, test_index in skf:
        ROCtrainTRN, ROCtestTRN = train[train_index], train[test_index]
        ROCtrainTRG, ROCtestTRG = target[train_index], target[test_index]
        X = ROCtrainTRN
        y = ROCtrainTRG
        X_submission = ROCtestTRN

        dataset_blend_train, dataset_blend_test = dataset_blend(X, y, X_submission, n_folds)
        probas = blending(dataset_blend_train, dataset_blend_test, y)
    
        s = _score(ROCtestTRN, ROCtestTRG, probas)
        print s
        scores.append(s)
        
    return np.array(scores).mean()

def predict_stacking(train, target, X_submission):
    """
    обучить на полном датасете и предсказать
    """
    n_folds = N_FOLDS
    dataset_blend_train, dataset_blend_test = dataset_blend(train, target, X_submission, n_folds)

    probas = blending(dataset_blend_train, dataset_blend_test, target)
    return probas

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
# XGBClassifier: 0.832832770289 удалил 'first_affiliate_tracked', 'first_browser' + mac_user
# XGBClassifier: 0.832828335352 0.87808 без статы по странам, удалил 'first_affiliate_tracked', 'first_browser'
# XGBClassifier: 0.832593609128 добавил расстояние языка
# XGBClassifier: 0.832739285371 добавил расстояние языка, убрал diff
# XGBClassifier: 0.83265603981 0.87869
# XGBClassifier: 0.833045826683 0.87800 c новыми данными по сессиям

# XGBClassifier: 0.86813210501 0.87892 без сложных стран
# XGBClassifier: 0.900100721605 0.86890 c доп фильтрацией stacking 0.86481

print 'XGBClassifier:', model_score('xgb', X, y, False)
# print 'MLPClassifier:', model_score(clf, X, y, False)
# print 'Stacking:', score_stacking(X, y)

print 'Stacking:', score_stacking(X, y)

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

def save(y_pred, name):
    ids = []  #list of ids
    cts = []  #list of countries
    for i in range(len(id_test)):
        idx = id_test[i]
        ids += [idx] * 5
        cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

    sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
    sub.to_csv(name,index=False)  

assert(len(X_test) == len_df_test)

if submit:
#     xgb = fxgb()
    
#     bgc = BaggingClassifier(base_estimator=xgb)
#     xgb.fit(X, y)
#     y_pred = xgb.predict_proba(X_test)
    y_pred = predict_stacking(X, y, X_test)
    save(y_pred, 'wo_c_filter_data_stacking.csv')

le.classes_


