from pyspark.sql import DataFrame, SparkSession
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
from  pyspark.ml.linalg import Vectors
from preprocess.preprocessor import shape_df


class TfidfFeaturizer:
    spark = None

    def __init__(self, sc):
        self.spark = sc

    def featurize(self, df:DataFrame) -> DataFrame:
        data_list = df.rdd.collect()
        label_list = []
        mod_list = []
        for data in data_list:
            label_list.append(data[0])
            tmp_list = []
            node_list = data[1]
            for word in node_list:
                tmp_list.append(word)
            modified_sentence = ' '.join(tmp_list)
            mod_list.append(modified_sentence)
        vectorizer = TfidfVectorizer()
        tfidf_list = vectorizer.fit_transform(mod_list).toarray().tolist()
        vec_list = []
        for tfidf in tfidf_list:
            vec_list.append(Vectors.dense(tfidf))
        zip_list = zip(label_list, vec_list)
        new_df = self.spark.createDataFrame(
            zip_list,
            ("label", "features")
        )
        return new_df

if __name__ == '__main__':
    spark = SparkSession.builder\
        .appName('Spark SQL and DataFrame')\
        .getOrCreate()
    df = spark.createDataFrame(
        [(21, "male", "友達が作ってくれたビネの白ドレス可愛すぎてたまらん😍"),
         (30, "female", "できればダブりたくないが初期の方のLRは避けたい"),
         (40, "male", "だから一生孤独でも構わんよ親にも作れと言われているけど"),
         ],
        ("age", "sex", "sentence")
    )
    converted_df = shape_df(spark, df).drop('age')
    featurizer = TfidfFeaturizer(spark)
    result_df = featurizer.featurize(converted_df)
    result_df.show(3)