import MeCab
import numpy as np
import normalizer, stopwords_handler
from pyspark import RDD
from pyspark.sql import DataFrame, Row


def to_morph(sentence: str) -> list:
    m = MeCab.Tagger("-Ochasen")
    node = m.parseToNode(sentence)
    node = node.next # skip BOF
    node_list = []
    while node:
        # word = node.surface
        node = node.next
        if node != None:
            feat = node.feature
            surface = node.surface
            node_list.append((surface, feat))
            
    return node_list


def row_to_feature(row: Row) -> tuple:
    sentence = row['sentence']
    age = row['age']
    sex = row['sex']
    nodes = to_morph(sentence)

    # feature_row = Row(('age', age), ('sex', sex), ('feat', morph))
    feature_row = (age, sex, nodes)

    return feature_row


def convert_df_to_feature(df: DataFrame) -> RDD:
    feature_rdd = df.rdd.filter(
        lambda row: row['sentence'] != None
    ).map(
        lambda row: row_to_feature(row)
    )

    return feature_rdd


'''
識別モデル構築の流れ (参考：https://qiita.com/MahoTakara/items/b3d719ed1a3665730826, https://qiita.com/Hironsan/items/2466fe0f344115aff177)

1.  DONE: 単語分割
2.  DONE: 形態素解析
3.  DONE: クリーニング
4.  DONE: 正規化 (ストップワードの除去含む)
5.  TODO: 辞書作成 (単語とIDの対応づけ)
6.  TODO: ベクトル化 (埋め込み or IDからone-hot)
7.  TODO: 文章特徴抽出 (文章の長さなど) 
8.  TODO: 提案特徴抽出 (フォロー/フォロワーの特徴)
9.  TODO: 識別モデルの実装
10. TODO: 評価メトリクスの実装
11. TODO: 実験実行
'''


if __name__ == "__main__":
    import os
    pwd = os.path.dirname(os.path.abspath(__file__))
    from csv_parser import load_as_df

    path = pwd + '/../example_data/20190528sentences_data_integrated.csv'
    df = load_as_df(path)
    df.show(3)

    converted = featurizer.convert_df_to_feature(df)
    sample = converted.take(3)
    for e in sample:
        print(e)