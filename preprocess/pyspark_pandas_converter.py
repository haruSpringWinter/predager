from pyspark.sql import DataFrame, SparkSession
import pandas as pd
from preprocess.preprocessor import shape_df
from featurizer.word2vec_featurizer import Word2VecFeaturizer


def convert_spark_to_pd(df: DataFrame):
    headers = ['labels']
    converted = []
    data_array = df.rdd.collect()
    cnt = 1
    for label, vec in data_array:
        if cnt == 1:
            dim = len(vec)
            for i in range(1, dim+1):
                headers.append("feat"+str(i))
        vec_list = vec.tolist()
        row_list = [label] + vec_list
        cnt += 1
        converted.append(row_list)

    pandas_df = pd.DataFrame(
        converted,
        columns=headers
    )

    return pandas_df


if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName('Spark SQL and DataFrame') \
        .getOrCreate()
    df = spark.createDataFrame(
        [(21, "male", "友達が作ってくれたビネの白ドレス可愛すぎてたまらん😍"),
         (30, "female", "できればダブりたくないが初期の方のLRは避けたい"),
         (40, "male", "だから一生孤独でも構わんよ親にも作れと言われているけど"),
         ],
        ("age", "sex", "sentence")
    )
    converted_df = shape_df(spark, df).drop("age")
    converted_df.show(3)
    # converted_df = df
    # datapath = "../param/word2vec/entity_vector/entity_vector.model.bin"
    datapath = "../param/word2vec/twitter_model/w2v_gensim/word2vec_tweet.model"
    wv = Word2VecFeaturizer(spark, datapath, False)
    new_df = wv.featurize(converted_df)
    new_df.show(3)
    pdf = convert_spark_to_pd(new_df)
    print(pdf.head(3))