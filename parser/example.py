import csv_parser
import preprocessor
import MeCab
import numpy as np
from pyspark.sql import Row, DataFrame
from pyspark import RDD

def convert_df_to_feature(df: DataFrame) -> RDD:
    feature_df = df.rdd.filter(
        lambda row: row['sentence'] != None
    )

    rdd = feature_df.map(
        lambda row: row_to_feature(row)
    )

    return rdd 


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


path = 'example_data/20190528sentences_data_integrated.csv'
df = csv_parser.load_as_df(path)
df.show(3)


converted = convert_df_to_feature(df)
sample = converted.take(3)
for e in sample:
    print(e)