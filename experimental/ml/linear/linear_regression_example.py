from pyspark.mllib.regression import LinearRegressionWithSGD, LabeledPoint, Vectors

# 文章の文字数から年齢を予測するexample
def predict_from_len():

    # データのローディング，特徴設計
    data = [LabeledPoint(1.0, Vectors(1.0, 2.0))]

    # 線型回帰モデルの学習
    lrm = LinearRegressionWithSGD.train(data)

    # テスト
    test = [LabeledPoint(1.0, Vectors(1.0, 2.0))] 
    lrm.predict(test)