from pyspark.sql import DataFrame, SparkSession
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfFeaturizer:
    def featurize(df:DataFrame):
        label_list = df.rdd.map(lambda r: r[0]).collect()
        sentence_list = df.rdd.map(lambda r: r[1]).collect()
        mecab = MeCab.Tagger('-Ochasen')
        mod_list = []
        for sentence in sentence_list:
            tmp_list = []
            node = mecab.parseToNode(sentence)
            while node:
                word = node.surface
                tmp_list.append(word)
                node = node.next
            modified_sentence = ' '.join(tmp_list)
            mod_list.append(modified_sentence)
        vectorizer = TfidfVectorizer()
        tfidf_mat = vectorizer.fit_transform(mod_list)
        zip_list = zip(label_list, tfidf_mat.toarray().tolist())
        new_df = spark.createDataFrame(
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
    result_df = TfidfFeaturizer.featurize(df)
    result_df.show(3)