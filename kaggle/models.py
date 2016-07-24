from sklearn.cross_validation import StratifiedKFold
import numpy as np
from copy import deepcopy


def cross_validation_score(model_obj, train, target, func):
    """
    :param model_obj:
    :param train: np.Array
    :param target: np.Array
    :param func:
    :return:
    """
    skf = StratifiedKFold(target, n_folds=4, random_state=1234)
    scores = []

    model = deepcopy(model_obj)

    for train_index, test_index in skf:
        X_train, X_test = train[train_index], train[test_index]
        y_train, y_test = target[train_index], target[test_index]

        model.fit(X_train, y_train)
        probas = model.predict_proba(X_test)
        score_k = func(y_test, probas)
        scores.append(score_k)

    return np.array(scores).mean()
