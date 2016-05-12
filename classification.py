#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scikit-learnの機械学習でよく使われるものを関数化
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
import seaborn as sns


# from malss import MALSS


def smart(lists: list):
    """A1 A10 A2みたいなものをスマートに並び替える"""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    lists.sort(key=alphanum_key)
    return lists


def sort_smart(lists: list):
    try:
        return smart(lists)
    except:
        return sorted(lists)


def plot_confusion_matrix(cm: np.ndarray, genre_list: list):
    plt.clf()
    plt.matshow(cm, fignum=False, cmap="Blues", vmin=0, vmax=1.0)
    plt.xticks(list(range(len(genre_list))), genre_list, rotation=90, verticalalignment='bottom')
    plt.yticks(list(range(len(genre_list))), genre_list)
    plt.title("confusion_matrix")
    plt.colorbar()
    plt.grid(False)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.grid(False)
    plt.show()


class Classification():
    """使用関数のクラス化"""

    def __init__(self, train: np.ndarray, train_label: np.ndarray):
        """
        :param train: 学習データ
        :param train_label: 正解データ
        """
        self.train = train
        self.train_label = train_label

    def set_train(self, train: np.ndarray, train_label: np.ndarray):
        """
        :param train: 学習データ
        :param train_label: 正解データ
        """
        self.train = train
        self.train_label = train_label

    def set_test(self, test: np.ndarray, test_label: np.ndarray):
        """
        テストデータを設定
        :param test: テストデータ
        :param test_label: 正解ラベル
        """
        self.test = test
        self.test_label = test_label

    def set_split(self, train: np.ndarray, train_label: np.ndarray, test_size: float = 0.25):
        """
        データを学習用とテスト用に分割
        :param train: 学習データ
        :param train_label: 正解ラベル
        :param test_size: テストデータの割合
        """
        self.train, self.test, self.train_label, self.test_label = \
            train_test_split(train, train_label, test_size=test_size, random_state=0)

    def set_classifier(self, clf):
        self.clf = clf

    def svm_gridsearch(self, n: int = 5):
        """
        :param n: 交差検定の交差数
        """
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': np.logspace(-4, -2, 10),
                             'C': np.r_[np.logspace(0, 2, 10), np.logspace(2, 3, 10)]},
                            {'kernel': ['linear'], 'gamma': np.logspace(-4, -2, 10),
                             'C': np.r_[np.logspace(0, 2, 10), np.logspace(2, 3, 10)]}]

        cv = StratifiedKFold(self.train_label, n_folds=n, shuffle=True)
        clf = GridSearchCV(SVC(probability=True, class_weight='balanced', decision_function_shape='ovr'),
                           tuned_parameters, cv=cv, n_jobs=-1)

        print("grid search...")
        clf.fit(self.train, self.train_label)
        print(clf.best_estimator_)
        self.clf = clf.best_estimator_
        self.bestclf = clf.best_estimator_
        print("set classifier")

    def lr_gridsearch(self, n: int = 5):
        """
        :param n: 交差検定の交差数
        """
        parameters = {'penalty': ["l1", "l2"],
                      'C': np.r_[np.logspace(0, 2, 50), np.logspace(2, 3, 50)],
                      'class_weight': [None, "auto"]}

        cv = StratifiedKFold(self.train_label, n_folds=n, shuffle=True)
        clf = GridSearchCV(LogisticRegression(multi_class='multinomial', solver='lbfgs'), parameters, cv=cv, n_jobs=-1)

        print("grid search...")
        clf.fit(self.train, self.train_label)
        print(clf.best_estimator_)
        self.clf = clf.best_estimator_
        self.bestclf = clf.best_estimator_
        print("set classifier")

    def cv(self, k: int = 5):
        """
        交差検定を行う
        :param k: 交差数
        """
        self.key = sort_smart(sorted(list(set(self.train_label))))
        self.conf_mat = np.zeros((len(self.key), len(self.key)))
        self.miss = []

        self.merge_true = np.array([])
        self.merge_pred = np.array([])

        cv = StratifiedKFold(self.train_label, n_folds=k, shuffle=True)

        for train_index, test_index in cv:
            cv_train = self.train[train_index]
            cv_trainlabel = self.train_label[train_index]
            cv_test = self.train[test_index]
            cv_testlabel = self.train_label[test_index]

            self.clf.fit(cv_train, cv_trainlabel)
            cv_pred = self.clf.predict(cv_test)

            for i in range(0, len(cv_testlabel)):
                if cv_testlabel[i] != cv_pred[i]:
                    self.miss.append([test_index[i], cv_testlabel[i], cv_pred[i]])

            self.merge_true = np.hstack([self.merge_true, cv_testlabel])
            self.merge_pred = np.hstack([self.merge_pred, cv_pred])
            # print classification_report(cv_testlabel,cv_pred)
            cm = confusion_matrix(cv_testlabel, cv_pred, self.key)
            self.conf_mat = self.conf_mat + cm
        # scores = cross_validation.cross_val_score(self.clf,self.train,self.train_label,cv=cv)
        # print "\nAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std() * 2)
        print('\nfinal classification report\n')
        print('accuracy score:', accuracy_score(self.merge_true, self.merge_pred), '\n')
        print(classification_report(self.merge_true, self.merge_pred, labels=self.key))
        self.conf_mat = np.array([list(c / float(sum(c))) for c in self.conf_mat])
        plot_confusion_matrix(self.conf_mat, self.key)
        self.miss.sort(key=lambda x: x[0])

    def prediction(self):
        self.clf.fit(self.train, self.train_label)
        pred = self.clf.predict(test)
        print(classification_report(test_label, pred))
        plot_confusion_matrix(confusion_matrix(test_label, pred), key)


def report_classification(train: np.ndarray, train_label: np.ndarray, name: str = 'result_classification'):
    """
    MALSSというツール
    http://qiita.com/canard0328/items/5da95ff4f2e1611f87e1
    :param name: ファイル名
    :param train_label: 正解ラベル
    :param train: 学習データ
    """
    cls = MALSS('classification', standardize=False, n_jobs=-1, random_state=0, lang='jp')
    cls.fit(train, train_label, name)
