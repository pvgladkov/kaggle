# coding: utf-8

from utils.file_utils import get_test_df, get_train_df
from settings import train_path, test_path
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    test = get_test_df(test_path)
    train = get_train_df(train_path, use_crop=False, use_original=True)

    # test.to_csv('test.csv', index=False)
    # train.to_csv('train_all.csv', index=False)

    train_train, train_validate, _, _ = train_test_split(train['path'].values, train['camera'].values,
                                                         test_size=0.1, random_state=777)

    print train_train[:10]

    # train_train.to_csv('train.csv', index=False)
    # train_validate.to_csv('validate.csv', index=False)