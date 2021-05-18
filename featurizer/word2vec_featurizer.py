import numpy as np
from gensim.models import KeyedVectors
from pyspark.sql import DataFrame, SparkSession
import MeCab


class Word2VecFeaturizer:
    data_path = ''
    wc_list = []
    spark = None

    def __init__(self, sc, path, wc=None):
        self.spark = sc
        self.data_path = path
        self.wc_list = wc

    def featurize(self, df):
        model = KeyedVectors.load_word2vec_format(self.data_path, binary=True)
        mecab = MeCab.Tagger('-Ochasen')
        data_list = df.rdd.collect()
        sum_vec = np.array([])
        label_list = []
        vec_list = []
        for data in data_list:
            node = mecab.parseToNode(data[1])
            vc_list = model.index_to_key
            cnt = 0
            while node:
                word = node.surface
                word_class = node.feature.split(",")[0]
                if word in vc_list and (self.wc_list is None or word_class in self.wc_list):
                    cnt += 1
                    if cnt == 1:
                        sum_vec = model.get_vector(word).copy()
                    else:
                        sum_vec += model.get_vector(word)
                node = node.next
            if cnt != 0:
                label_list.append(data[0])
                result_vec = sum_vec / cnt
                vec_list.append(result_vec.tolist())

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
        [(1, "友達が作ってくれたビネの白ドレス可愛すぎてたまらん😍"),
         (0, "できればダブりたくないが初期の方のLRは避けたい"),
         (0, "だから一生孤独でも構わんよ親にも作れと言われているけど"),
         ],
        ("label", "sentence")
    )
    datapath = "../param/word2vec/entity_vector/entity_vector.model.bin"
    wv = Word2VecFeaturizer(spark, datapath)
    df = wv.featurize(df)
    df.show(3)