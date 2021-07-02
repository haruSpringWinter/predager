from pyspark.sql import SparkSession
import MeCab
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from  pyspark.ml.linalg import Vectors
from preprocess.preprocessor import shape_df


class BertFeaturizer:
    data_path = ''
    wc_list = []
    spark = None

    def __init__(self, sc, path, wc=None):
        self.spark = sc
        self.data_path = path
        self.wc_list = wc

    def featurize(self, df):
        bert_model = BertModel.from_pretrained(self.data_path)
        bert_tokenizer = BertTokenizer(self.data_path+"/vocab.txt", do_lower_case=False, do_basic_tokenize=False)
        mecab = MeCab.Tagger('-Ochasen')
        data_list = df.rdd.collect()
        label_list = []
        vec_list = []
        for data in data_list:
            tmp_list = []
            node_list = data[1]
            for word in node_list:
                tmp_list.append(word)
            if len(tmp_list) != 0:
                label_list.append(float(data[0]))
                bert_tokens = bert_tokenizer.tokenize(" ".join(["[CLS]"] + tmp_list + ["[SEP]"]))
                token_ids = bert_tokenizer.convert_tokens_to_ids(bert_tokens)
                tokens_tensor = torch.tensor(token_ids).unsqueeze(0)
                all_outputs = bert_model(tokens_tensor)
                embedding = all_outputs[-2].detach().numpy()[0]
                vec = np.mean(embedding, axis=0).tolist()
                vec_list.append(Vectors.dense(vec))
        zip_list = zip(label_list, vec_list)
        new_df = self.spark.createDataFrame(
            zip_list,
            ("label", "features")
        )
        return new_df


if __name__ == '__main__':
    sc = SparkSession.builder\
        .appName('Spark SQL and DataFrame')\
        .getOrCreate()
    df = sc.createDataFrame(
        [(21, "male", "友達が作ってくれたビネの白ドレス可愛すぎてたまらん😍"),
         (30, "female", "できればダブりたくないが初期の方のLRは避けたい"),
         (40, "male", "だから一生孤独でも構わんよ親にも作れと言われているけど"),
         ],
        ("age", "sex", "sentence")
    )
    converted_df = shape_df(sc, df).drop('age')
    data_path = "../param/bert/Japanese_L-24_H-1024_A-16_E-30_BPE_WWM_transformers"
    bert = BertFeaturizer(sc, data_path)
    result_df = bert.featurize(converted_df)
    result_df.show(3)