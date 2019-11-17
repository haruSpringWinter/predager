import MeCab
import numpy as np
from pyspark import RDD
from pyspark.sql import DataFrame, Row
from gensim.corpora import Dictionary

from . import normalizer, stopwords_handler, clearner

dct = Dictionary()


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


def update_dictionary(nodes: list) -> None:
    global dct

    surfaces = []
    for node in nodes:
        surfaces.append(node[0])

    dct.add_documents([surfaces])    


def preprocess(row: Row, n: int = 100, min_freq: int = 1, for_test = False) -> tuple:
    global dct
    sentence = row['sentence']
    clean_text = clearner.clean_text(sentence)
    normalized_text = normalizer.normalize(clean_text)
    nodes = to_morph(normalized_text)

    if for_test:
        nodes = [node for node in nodes if node[0] in dct]

    cleaned_nodes = stopwords_handler.remove_stopnodes(nodes)
    update_dictionary(cleaned_nodes)

    age = row['age']
    sex = row['sex']

    # feature_row = Row(('age', age), ('sex', sex), ('feat', morph))
    preprocessed = (age, sex, cleaned_nodes)

    return preprocessed


def convert_row_to_feature_vec(row: tuple) -> tuple:
    global dct

    dim = len(dct)
    nodes = row[2]
    terms = [node[0] for node in nodes]
    vecs = []
    for term in terms:
        vec = [0 for i in range(dim)]
        vec[dct.token2id[term]] += 1
        vecs.append(vec)
    # TODO: Add some features
    
    return (row[0], row[1], vecs)


def convert_df_to_feature(df: DataFrame, for_test = False) -> RDD:
    preproc_rdd = df.rdd.filter(
        lambda row: row['sentence'] != None
    ).map(
        lambda row: preprocess(row)
    )

    feature_rdd = preproc_rdd.map(
        lambda row: convert_row_to_feature_vec(row)
    )

    return feature_rdd


'''
識別モデル構築の流れ (参考：https://qiita.com/MahoTakara/items/b3d719ed1a3665730826, https://qiita.com/Hironsan/items/2466fe0f344115aff177)

1.  DONE: 単語分割
2.  DONE: 形態素解析
3.  DONE: クリーニング
4.  DONE: 正規化 (ストップワードの除去含む)
5.  DONE: 辞書作成 (単語とIDの対応づけ) https://qiita.com/tatsuya-miyamoto/items/f505dfa8d5307f8c6e98　簡単にできそう
6.  DONE: ベクトル化 (埋め込み or IDからone-hot)
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

    converted = convert_df_to_feature(df)
    sample = converted.take(3)
    for e in sample:
        print(e)