import numpy as np
from gensim.models import KeyedVectors
from pyspark.sql import DataFrame, SparkSession
import MeCab

def genVec(data_path, row, wc_list):
    model = KeyedVectors.load_word2vec_format(data_path, binary=True)
    mecab = MeCab.Tagger('-Ochasen')
    cnt = 0
    sum_vec = np.array([])
    result_vec = sum_vec
    node = mecab.parseToNode(row[1])
    vc_list = model.index_to_key
    while node:
        word = node.surface
        word_class = node.feature.split(",")[0]
        if word in vc_list and word_class in wc_list:
            cnt += 1
            if cnt == 1:
                sum_vec = model.get_vector(word).copy()
            else:
                sum_vec += model.get_vector(word)
        node = node.next
    if cnt != 0:
        result_vec = sum_vec / cnt
    return row[0], result_vec.tolist()

def featurize(data_path:str, df:DataFrame, wc_list=None):
    if wc_list is None:
        wc_list = ["名詞", "形容詞", "動詞"]
    return df.rdd.map(lambda x: genVec(data_path, x, wc_list)).toDF(schema=["label", "features"])

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
    df = featurize(datapath, df)
    df.show(3)