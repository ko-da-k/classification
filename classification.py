#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scikit-learnの機械学習でよく使われるものを関数化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.learning_curve import learning_curve
import seaborn as sns
from tools import sort_smart


# from malss import MALSS



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

    def svm_gridsearch(self, n: int = 5,
                       C_range: np.ndarray = np.r_[np.logspace(0, 2, 10), np.logspace(2, 3, 10)],
                       gamma_range: np.ndarray = np.logspace(-4, -2, 10),
                       plot: bool = False):
        """
        gridsearchを行う関数
        :param n: 交差数
        :param C_range: Cパラメータの範囲
        :param gamma_range: gammaパラメータの範囲
        :param plot: 可視化するかしないか
        :return:
        """
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': gamma_range,
                             'C': C_range},
                            {'kernel': ['linear'], 'C': C_range}]

        cv = StratifiedKFold(self.train_label, n_folds=n, shuffle=True)
        clf = GridSearchCV(SVC(probability=True, class_weight='balanced', decision_function_shape='ovr'),
                           tuned_parameters, cv=cv, n_jobs=-1)

        print("grid search...")
        clf.fit(self.train, self.train_label)

        self.clf = clf.best_estimator_
        self.bestclf = clf.best_estimator_

        scores = [x[1] for x in clf.grid_scores_[:len(C_range) * len(gamma_range)]]
        scores = np.array(scores).reshape(len(C_range), len(gamma_range))
        self.rbf_scores = pd.DataFrame(scores, index=C_range, columns=gamma_range)
        self.rbf_scores.index.name = "C"
        self.rbf_scores.columns.name = "gamma"
        self.linear_scores = [x[1] for x in clf.grid_scores_[len(C_range) * len(gamma_range):]]

        if plot:
            sns.heatmap(self.rbf_scores, annot=True, cmap="Blues")
            sns.plt.show()
            plt.plot(C_range, self.linear_scores)
            plt.xlabel("C")
            plt.ylabel("accuracy")
            plt.show()

        print(clf.best_estimator_)
        print("set classifier")

    def lr_gridsearch(self, n: int = 5, C_range: np.ndarray = np.r_[np.logspace(0, 2, 10), np.logspace(2, 3, 10)]):
        """
        ロジスティック回帰分類のgridsearch
        パラメータは、lpfgs法の場合(多クラス分類に使用)はl2ノルムしか取れない
        :param n: 交差数
        :param C_range: Cパラメータの範囲
        :return:
        """
        parameters = {'penalty': ["l2"],
                      'C': C_range,
                      'class_weight': [None, "balanced"]}
        """
        parameters = {'penalty': ["l1", "l2"],
                      'C': C_range,
                      'class_weight': [None, "balanced"]}
        """
        cv = StratifiedKFold(self.train_label, n_folds=n, shuffle=True)
        clf = GridSearchCV(LogisticRegression(multi_class='multinomial', solver='lbfgs'), parameters, cv=cv, n_jobs=-1)

        print("grid search...")
        clf.fit(self.train, self.train_label)
        self.clf = clf.best_estimator_
        self.bestclf = clf.best_estimator_
        scores = [x[1] for x in clf.grid_scores_]
        scores = np.array(scores).reshape(len(C_range), 2)
        self.scores = pd.DataFrame(scores, index=C_range, columns=parameters["class_weight"])
        self.scores.index.name = "C"
        self.scores.columns.name = "class_weight"
        sns.heatmap(self.scores, annot=True, cmap="Blues")
        sns.plt.show()
        print(clf.best_estimator_)
        print("set classifier")

    def cv(self, k: int = 5, plot: bool = False):
        """
        交差検定を行う
        :param k: 交差数
        :param plot: 可視化するかしないか
        """
        self.key = sort_smart(list(set(self.train_label)))
        self.conf_mat = np.zeros((len(self.key), len(self.key)))
        self.miss = []

        merge_true = np.array([])
        merge_pred = np.array([])

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

            merge_true = np.hstack([merge_true, cv_testlabel])
            merge_pred = np.hstack([merge_pred, cv_pred])
            # print classification_report(cv_testlabel,cv_pred)
            cm = confusion_matrix(cv_testlabel, cv_pred, self.key)
            self.conf_mat = self.conf_mat + cm

        print('\nfinal classification report\n')
        print('accuracy score:', accuracy_score(merge_true, merge_pred), '\n')
        print(classification_report(merge_true, merge_pred, labels=self.key))
        if plot:
            recall_mat = np.array([list(c / float(sum(c))) for c in self.conf_mat])
            recall_mat = pd.DataFrame(recall_mat, index=self.key, columns=self.key)
            recall_mat.index.name = "True class"
            recall_mat.columns.name = "Predict class"
            sns.heatmap(recall_mat, annot=True, cmap="Blues", vmax=1.0, vmin=0.0)
            sns.plt.show()
        self.miss.sort(key=lambda x: x[0])
        self.miss = pd.DataFrame(self.miss, columns=["index", "True_label", "Pred_label"])
        print(self.conf_mat)

    def prediction(self):
        clf = self.clf.fit(self.train, self.train_label)
        pred = clf.predict(self.test)
        print(classification_report(self.test_label, pred))

    def draw_learning_curve(self):
        """http://aidiary.hatenablog.com/entry/20150826/1440596779"""
        train_sizes, train_scores, test_scores = learning_curve(self.clf, self.train, self.train_label, cv=10,
                                                                scoring="mean_squared_error",
                                                                train_sizes=np.linspace(0.5, 1.0, 10))
        plt.plot(train_sizes, train_scores.mean(axis=1), label="training scores")
        plt.plot(train_sizes, test_scores.mean(axis=1), label="test scores")
        plt.legend(loc="best")
        plt.show()

    def draw_roc_curve(self):

        train, test, train_label, test_label = train_test_split(self.train, self.train_label)
        clf = self.bestclf
        probas_ = clf.fit(train, train_label).predict_proba(test)

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(test_label, probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        print("Area under the ROC curve :", roc_auc)

        # Plot ROC curve
        plt.clf()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
