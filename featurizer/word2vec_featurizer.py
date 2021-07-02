import numpy as np
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models import word2vec
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.linalg import Vectors
from preprocess.preprocessor import shape_df


class Word2VecFeaturizer:
    data_path = ''
    spark = None
    useKeyedVectors = True

    def __init__(self, sc, path, useKeyedVectors=True):
        self.spark = sc
        self.data_path = path
        self.useKeyedVectors = useKeyedVectors

    def featurize(self, df):
        data_list = df.rdd.collect()
        label_list = []
        vec_list = []
        sum_vec = np.array([])
        if self.useKeyedVectors:
            model = KeyedVectors.load_word2vec_format(self.data_path, binary=True)
        else:
            model = Word2Vec.load(self.data_path).wv
        vc_list = model.index_to_key
        for data in data_list:
            node_list = data[1]
            cnt = 0
            for word in node_list:
                if word in vc_list:
                    cnt += 1
                    if cnt == 1:
                        sum_vec = model.get_vector(word).copy()
                    else:
                        sum_vec += model.get_vector(word)
            label_list.append(float(data[0]))
            if cnt != 0:
                result_vec = sum_vec / cnt
                vec_list.append(Vectors.dense(result_vec.tolist()))
            else:
                vec_list.append(Vectors.dense([0]*model.vector_size))

        zip_list = zip(label_list, vec_list)
        new_df = self.spark.createDataFrame(
            zip_list,
            ("label", "features")
        )
        # new_df.select(*new_df.columns).printSchema()
        return new_df



if __name__ == '__main__':
    # import findspark
    # findspark.init("/usr/local/Cellar/apache-spark/3.1.2/libexec")
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
    converted_df = shape_df(spark, df).drop("age")
    converted_df.show(3)
    # converted_df = df
    # datapath = "../param/word2vec/entity_vector/entity_vector.model.bin"
    datapath = "../param/word2vec/twitter_model/w2v_gensim/word2vec_tweet.model"
    wv = Word2VecFeaturizer(spark, datapath, False)
    new_df = wv.featurize(converted_df)
    new_df.show(3)