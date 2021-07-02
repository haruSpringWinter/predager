from pyspark.sql import DataFrame, SparkSession
import MeCab
from gensim.corpora import Dictionary
from pyspark.ml.linalg import Vectors
from preprocess.preprocessor import shape_df


class EmojiKaomojiCountFeaturizer():

    global_dict = Dictionary()
    spark = None

    def __init__(self, sc):
        self.spark = sc

    def featurize(self, df:DataFrame) -> DataFrame:
        data_list = df.rdd.collect()
        mecab = MeCab.Tagger('-Ochasen')
        label_list = []
        wakati_list = []
        for data in data_list:
            tmp_list = []
            node_list = data[1]
            for node in node_list:
                word = node[0]
                tmp_list.append(word)
            if len(tmp_list) != 0:
                label_list.append(data[0])
                wakati_list.append(tmp_list)

        self.global_dict.add_documents(wakati_list)
        dim = len(self.global_dict)
        vec_list = []
        for wakati in wakati_list:
            vec = [0 for _ in range(dim)]
            for word in wakati:
                vec[self.global_dict.token2id[word]] = 1
            vec_list.append(Vectors.dense(vec))
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
    oneHot = OneHotFeaturizer(spark)
    result_df = oneHot.featurize(converted_df)
    result_df.show(3)