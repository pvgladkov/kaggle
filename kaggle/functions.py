import pandas as pd


def ohe_df(df_all, ohe_feats):
    """
    One-hot-encoding features
    :param df_all: pd.DataFrame
    :param ohe_feats:
    :param ex:
    :return:
    """
    for f in ohe_feats:
        if f in df_all.columns.values:
            df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
            df_all = df_all.drop([f], axis=1)
            df_all = pd.concat((df_all, df_all_dummy), axis=1)
    return df_all
