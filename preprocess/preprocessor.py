import MeCab
import nagisa
from pyspark import RDD
from pyspark.sql import DataFrame, Row
from gensim.corpora import Dictionary
from schema.twitter_schema import twitter_schema

from preprocess import clearner, normalizer, stopwords_handler

global_dict = Dictionary()

def to_morph(sentence: str, method: str, wc_list=None) -> list:
    node_list = []
    if method == 'mecab':
        m = MeCab.Tagger("-Ochasen")
        node = m.parseToNode(sentence)
        node = node.next # skip BOF
        while node:
            # word = node.surface
            node = node.next
            if node is not None and (wc_list is None or node.feature.split(",")[0] in wc_list):
                word = node.surface
                node_list.append(word)
    elif method == 'nagisa':
        if wc_list is None:
            words = nagisa.tagging(sentence).words
        else:
            words = nagisa.extract(sentence, extract_postags=wc_list).words
        for word in words:
            node_list.append(word)

    return node_list



def convert_str_to_int(sex: str) -> int:
    if sex == 'male':
        return 0
    elif sex == 'female':
        return 1
    else:
        return 2

def preprocess(row: Row, method: str, wc_list=None, for_test: bool = False, dic: dict = None):
    '''
    文章の不要語の除去，正規化，形態素解析を行い，結果としてラベル(年齢/性別)とデータのRowを返す．
    '''
    sentence = row['sentence']
    clean_text = clearner.clean_text(sentence)
    normalized_text = normalizer.normalize(clean_text)
    nodes = to_morph(normalized_text, method, wc_list)

    if for_test:
        nodes = [node for node in nodes if node[0] in dic]

    cleaned_nodes = stopwords_handler.remove_stopnodes(nodes)
    # cleaned_nodes = nodes

    age = row['age']
    sex = row['sex']
    int_sex = convert_str_to_int(sex)

    # feature_row = Row(('age', age), ('sex', sex), ('feat', morph))
    # preprocessed = Row(age = age, sex = int_sex, nodes = cleaned_nodes)
    return age, int_sex, cleaned_nodes

    # return preprocessed

def convert_row_to_feature_vec(row: tuple, dic: dict) -> Row:
    '''
    一行のデータを特徴量化するコード．
    '''
    dim = len(dic)
    nodes = row['nodes']
    terms = [node[0] for node in nodes]
    vecs = []
    for term in terms:
        # いくつかの単語が登録されずにエラーを起こすので try-catchでやり過ごす (なぜかunicode文字列として認識されているっぽい)
        # one-hotエンコーディング
        try:
            vec = [0 for i in range(dim)]
            vec[dic.token2id[term]] += 1
            vecs.append(vec)
        except Exception as e:
            pass
    # TODO: Add some features
    
    return Row(age = row['age'], sex = row['sex'], feature = vecs)

# Unused
# def update_dictionary(nodes: list) -> None:
#     global global_dict

#     surfaces = []
#     for node in nodes:
#         surfaces.append(node[0])
#     global_dict.add_documents([surfaces])    


def convert_df_to_feature(df: DataFrame, n: int = 10, min_freq: int = 1, wc_list=None, for_test=False, dic: dict=None) -> RDD:
    global global_dict
    preproc_rdd = df.rdd.filter(
        lambda row: row['sentence'] != None and row['sentence'].split(' ')[0] != 'RT' and row['sentence'].split(' ')[0] != 'rt'
    ).map(
        # TODO: preprocessなどのコードはhandlersとして登録して自由に処理を切り替えられるようにする (pyTorchのtransforms的な方法)
        lambda row: preprocess(row, wc_list, for_test, dic)
    )

    # 特徴量化時にRDDなめるのと同時にDictを更新しようとするとうまくいかないので個別に更新.
    # 並列処理の時にはglobal変数の更新はうまくいかないっぽい．
    # https://stackoverflow.com/questions/44921837/how-to-update-a-global-variable-inside-rdd-map-operation
    if not for_test:
        def get_vocab(row: Row) -> list:
            # (age, sex, clean_nodes)
            nodes = row['nodes']
            return [node[0] for node in nodes]
        vocab = preproc_rdd.map(
            lambda x:  get_vocab(x)
        ).collect()
        global_dict.add_documents(vocab)

    feature_rdd = preproc_rdd.map(
        lambda row: convert_row_to_feature_vec(row, global_dict)
    )

    return feature_rdd


def shape_df(spark, df: DataFrame, method='mecab', wc_list=None)-> DataFrame:
    preproc_rdd = df.rdd.filter(
        lambda row: row['age'] is not None and row['sex'] is not None and row['sentence'] is not None and row['sentence'].split(' ')[0] != 'RT' and row['sentence'].split(' ')[0] != 'rt'
    ).map(
        lambda row: preprocess(row, method, wc_list)
    )


    preproc_list = preproc_rdd.collect()
    preproc_df = \
        spark.createDataFrame(
            preproc_list,
            ('age', 'sex', 'sentence')
        )
    return preproc_df

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
    from parser.csv_parser import load_as_df

    path = pwd + '/../example_data/twitter_2020-03-10_slim.csv'
    df = load_as_df(path, twitter_schema)
    df.show(3)

    from pyspark.sql import DataFrame, SparkSession
    spark = SparkSession.builder \
        .appName('Spark SQL and DataFrame') \
        .getOrCreate()
    converted_df = shape_df(spark, df, 'nagisa')
    converted_df.show(3)