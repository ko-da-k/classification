scikit-learnでよく使うものを自分が使いやすいようにクラス化

環境 Python 3.5.1 :: Anaconda 2.5.0 (x86_64)
 
使い方
------
sample.pyがテストファイル
 
### 分類の使い方 ###
    clf = classification.Classification(train, train_label)  # 学習データをセット
    clf.set_test(test, test_label)  # テストデータをセット
    clf.svm_gridsearch(5)  # 5交差でグリッドサーチ
    print(clf.bestclf)  # グリッドサーチの結果を表示
    clf.cv(5) # 5交差検定の結果を表示
    print("test Result")
    clf.prediction()  # テストデータの分類結果を表示 
    
グリッドサーチが終わった時点で、インスタンス変数にグリッドサーチで得られたパラメータの分類器ができている。``self.clf``
グリッドサーチのパラメータは中で設定してあるので調整必須
