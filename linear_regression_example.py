import os
import numpy as np
from pyspark.mllib.regression import LinearRegressionWithSGD, LabeledPoint
from pyspark.mllib.linalg import Vectors
from parser import csv_parser
from preprocess import featurizer
from schema.twitter_schema import twitter_schema


# 文章の文字数から年齢を予測するexample
def lr_example():
    min_freq = 1
    n_common = 10

    pwd = os.path.dirname(os.path.abspath(__file__))
    path = pwd + '/example_data/twitter_2020-03-10_slim.csv'
    print(path)
    df = csv_parser.load_as_df(path, twitter_schema)
    df.show(3)

    converted = featurizer.convert_df_to_feature(df, n_common, min_freq).filter(
        lambda row: row['age'] is not None and row['feature'] is not None
    )
    converted = converted.map(
        # (age, sex, feature)
        lambda row: LabeledPoint(row['age'], concat_vectors(row['feature']))
    )
    converted = converted.zipWithIndex()

    sample = converted.take(3)

    train_rdd = converted.filter(
        lambda x: x[1] % 2 == 0
    ).map(
        lambda x: x[0]
    )

    feature_dim = len(train_rdd.first().features)

    test_rdd = converted.filter(
        lambda x: x[1] % 2 == 1
    ).map(
        lambda x: x[0]
    ).filter(
        lambda x: len(x.features) == feature_dim
    ).collect()

    print("confirming dim of train rdd")
    sample = train_rdd.take(3)
    for e in sample:
        print(e.features)
        print(len(e.features))

    lrm = LinearRegressionWithSGD.train(train_rdd)
    n = len(test_rdd)

    mse = 0
    # テスト
    for lp in test_rdd:
        gt = lp.label
        feat = lp.features
        pred = lrm.predict(feat)
        print(gt, pred)
        mse += (pred - gt) * (pred - gt)

    import math
    rmse = math.sqrt(mse / n)

    print('Root mean square error: ' + str(rmse))


def concat_vectors(vecs: list):
    concat_vec = np.array([])
    for idx, vec in enumerate(vecs):
        if idx == 0:
            concat_vec = np.array(vec)
        else:
            concat_vec = concat_vec + np.array(vec)

    return Vectors.dense(*concat_vec)


if __name__ == "__main__":
    lr_example()
