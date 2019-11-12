import csv_parser
import preprocessor
import MeCab
import numpy as np
from pyspark.sql import Row, DataFrame

def convert_df_to_feature(df: DataFrame) -> DataFrame:
    feature_df = df.rdd.map(
        lambda row: row_to_feature(row)
    )

    return feature_df

def to_morph(sentence: str) -> np.array:
    m = MeCab.Tagger("-Ochasen")
    node = m.parseToNode(sentence)
    list = []
    while node:
        word = node.surface
        node = node.next
        list.append(word)
    return str(list)

# Row(age, sex, sentence, url)
def row_to_feature(row: Row) -> Row:
    sentence = row['sentence']
    age = row['age']
    sex = row['sex']
    morph = to_morph(sentence)

    feature_row = Row(('age', age), ('sex', sex), ('feat', morph))
    return feature_row

path = 'example_data/20190528sentences_data_integrated.csv'
df = csv_parser.load_as_df(path)
df.show(3)

converted = convert_df_to_feature(df)
print(converted.count())
sample = converted.take(3)
for e in sample:
    print(e)