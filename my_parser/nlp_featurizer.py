import MeCab
import numpy as np
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
    morph = to_morph(sentence)
    # feature_row = Row(('age', age), ('sex', sex), ('feat', morph))
    feature_row = (age, sex, morph)

    return feature_row


def convert_df_to_feature(df: DataFrame) -> RDD:
    feature_rdd = df.rdd.filter(
        lambda row: row['sentence'] != None
    ).map(
        lambda row: row_to_feature(row)
    )

    return feature_rdd

'''
識別モデル構築の流れ
1. DONE: 単語分割
2. DONE: 形態素解析
3. TODO: 辞書作成
4. TODO: 文章特徴抽出 (文章の長さなど) 
5. TODO: 提案特徴抽出 (フォロー/フォロワーの特徴)
6. TODO: 識別モデルの実装
7. TODO: 評価メトリクスの実装
8. TODO: 実験RUN
'''