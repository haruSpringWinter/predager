from pyspark.sql import DataFrame, SparkSession
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora import Dictionary


class OneHotFeaturizer():

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
            node = mecab.parseToNode(data[1])
            while node:
                word = node.surface
                tmp_list.append(word)
                node = node.next
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
            vec_list.append(vec)
        zip_list = zip(label_list, vec_list)
        new_df = spark.createDataFrame(
            zip_list,
            ("label", "features")
        )
        for v in vec_list:
            print(v)
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
    oneHot = OneHotFeaturizer(spark)
    result_df = oneHot.featurize(df)
    result_df.show(3)