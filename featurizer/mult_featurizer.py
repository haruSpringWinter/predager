from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors


class MultiFeaturizer:
    spark = None
    feat_list = []
    df_list = []

    def __init__(self, sc, f_list, df_list=None):
        self.spark = sc
        self.feat_list = f_list
        self.df_list = df_list

    def featurize(self, df=None):
        label_list = []
        vec_list = []
        cnt = 0
        for featurizer in self.feat_list:
            if df is not None:
                new_df = featurizer.featurize(df)
            elif self.df_list is not None:
                new_df = featurizer.featurize(self.df_list[cnt])
            data_list = new_df.rdd.collect()
            cnt += 1
            for i in range(len(data_list)):
                data = data_list[i]
                if cnt == 1:
                    label_list.append(data[0])
                    vec_list.append(data[1].tolist())
                else:
                    vec_list[i].extend(data[1].tolist())

        dense_vec_list = []
        for vec in vec_list:
            dense_vec_list.append(Vectors.dense(vec))
        zip_list = zip(label_list, dense_vec_list)
        new_df = self.spark.createDataFrame(
            zip_list,
            ["label", "features"]
        )
        return new_df


if __name__ == '__main__':
    from bert_featurizer import BertFeaturizer
    from word2vec_featurizer import Word2VecFeaturizer

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
    data_path = "../param/bert/Japanese_L-24_H-1024_A-16_E-30_BPE_WWM_transformers"
    bert = BertFeaturizer(spark, data_path)
    datapath = "../param/word2vec/entity_vector/entity_vector.model.bin"
    wv = Word2VecFeaturizer(spark, datapath)
    featurizer = MultiFeaturizer(spark, [bert, wv])
    result_df = featurizer.featurize(df)
    result_df.show(3)