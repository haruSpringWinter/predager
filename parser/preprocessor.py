import MeCab
import numpy as np
from pyspark import RDD
from pyspark.sql import DataFrame, Row

def to_morph(sentence: str) -> np.array:
    m = MeCab.Tagger("-Ochasen")
    node = m.parseToNode(sentence)
    list = []
    while node:
        word = node.surface
        node = node.next
        list.append(word)
    return np.array(list)

def to_some_feature(sentence: str):
    # TODO: determine featurization process and do it
    pass

def convert_df_to_feature(df: DataFrame) -> DataFrame:
    feature_df = df.rdd.map(
        lambda row: row_to_feature(row)
    ).toDF()

    return feature_df

# Row(age, sex, sentence, url)
def row_to_feature(row: Row) -> Row:
    sentence = row['sencence']
    age = row['age']
    sex = row['sex']
    morph = to_morph(sentence)

    Row(('age', age), ('sex', sex), ('feat', morph))