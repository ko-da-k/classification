#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import time
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from classification import Classification


if __name__ == "__main__":

    print('iris data')
    iris = datasets.load_iris()

    # 学習データとテストデータを3：1に分割
    train, test, train_label, test_label = train_test_split(iris.data, iris.target, test_size=0.25, random_state=0)
    clf = Classification(train, train_label)  # 学習データをセット
    clf.set_test(test, test_label)  # テストデータをセット
    clf.svm_gridsearch(5)  # 5交差でグリッドサーチ
    clf.cv(5)  # 5交差検定の結果を表示
    print("test Result")
    clf.prediction()  # テストデータの分類結果を表示
