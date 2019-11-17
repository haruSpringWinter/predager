import os, sys
import numpy as np
from pyspark.mllib.regression import LinearRegressionWithSGD, LabeledPoint, Vectors
from pyspark.mllib.linalg import Vectors
from my_parser import featurizer, csv_parser

# 文章の文字数から年齢を予測するexample
def lr_example():

    pwd = os.path.dirname(os.path.abspath(__file__))
    path = pwd + '/example_data/20190528sentences_data_integrated.csv'
    print(path)
    df = csv_parser.load_as_df(path)
    df.show(3)

    converted = featurizer.convert_df_to_feature(df)
    converted = converted.map(
        lambda x: LabeledPoint(x[0], concat_vectors(x[2]))
    )
    converted = converted.zipWithIndex()
    train_rdd = converted.filter(
        lambda x: x[1] % 2 == 0
    ).map(
        lambda x: x[0]
    )

    test_rdd = converted.filter(
        lambda x: x[1] % 2 == 1
    ).map(
        lambda x: x[0]
    )

    sample = train_rdd.take(3)
    for e in sample:
        print(e)

    # 線型回帰モデルの学習
    lrm = LinearRegressionWithSGD.train(train_rdd)

    # テスト
    lrm.predict(test_rdd)


def concat_vectors(vecs: list):
    vec = None
    for idx, vec in enumerate(vecs):
        if idx == 0:
            vec = np.array(vec)
        else:
            vec += np.array(vec)
    return Vectors.dense(*vec)


if __name__ == "__main__":
    lr_example()