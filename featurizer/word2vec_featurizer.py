import numpy as np
from gensim.models import KeyedVectors
from pyspark.sql import DataFrame, SparkSession
import MeCab


class Word2VecFeaturizer:
    data_path = ''
    wc_list = []

    def __init__(self, path, wc=None):
        self.data_path = path
        self.wc_list = wc

    def gen_vec(self, row):
        model = KeyedVectors.load_word2vec_format(self.data_path, binary=True)
        mecab = MeCab.Tagger('-Ochasen')
        cnt = 0
        sum_vec = np.array([])
        result_vec = sum_vec
        node = mecab.parseToNode(row[1])
        vc_list = model.index_to_key
        while node:
            word = node.surface
            word_class = node.feature.split(",")[0]
            if word in vc_list and word_class in self.wc_list:
                cnt += 1
                if cnt == 1:
                    sum_vec = model.get_vector(word).copy()
                else:
                    sum_vec += model.get_vector(word)
            node = node.next
        if cnt != 0:
            result_vec = sum_vec / cnt
        return row[0], result_vec.tolist()

    def featurize(self, df:DataFrame) -> DataFrame:
        if self.wc_list is None:
            self.wc_list = ["名詞", "形容詞", "動詞"]
        return df.rdd.map(lambda x: self.gen_vec(x)).toDF(schema=["label", "features"])

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
    wv = Word2VecFeaturizer(datapath)
    df = wv.featurize(df)
    df.show(3)