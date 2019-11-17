import os, sys
from pyspark.mllib.regression import LinearRegressionWithSGD, LabeledPoint, Vectors

sys.path.append('../../../')
from my_parser import featurizer, csv_parser

# 文章の文字数から年齢を予測するexample
def lr_example():

    pwd = os.path.dirname(os.path.abspath(__file__))
    path = pwd + '../../../example_data/20190528sentences_data_integrated.csv'
    print(pa)
    df = csv_parser.load_as_df(path)
    df.show(3)

    converted = featurizer.convert_df_to_feature(df)
    sample = converted.take(3)
    for e in sample:
        print(e)

    # データのローディング，特徴設計
    data = [LabeledPoint(1.0, Vectors(1.0, 2.0))]

    # 線型回帰モデルの学習
    lrm = LinearRegressionWithSGD.train(data)

    # テスト
    test = [LabeledPoint(1.0, Vectors(1.0, 2.0))] 
    lrm.predict(test)

if __name__ == "__main__":
    lr_example()