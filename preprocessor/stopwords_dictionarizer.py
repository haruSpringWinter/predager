"""
形態素解析のみを行ったfeatureとカテゴリのペアを持つような
RDDを元にストップワードの辞書を作成．

inputとなるRDDは形態素解析のみを行ったfeatureと
カテゴリをペアとして持つようなRDD(featurizer.preprocessの出力)を仮定．
"""

import os
import urllib.request
import itertools
import time, datetime
from pyspark import RDD
from collections import Counter

from gensim import corpora

pwd = os.path.dirname(os.path.abspath(__file__))
stopwords = []

dt = datetime.datetime.now()
time = dt_now.strftime('%Y%m%d')

file_base = str(pwd) + "/" + time 

def _extract_words_from_nodes(nodes: list) -> list:
    words = [node[0] for node in nodes]


def create_by_information_gain(raw_rdd: RDD, threshold: float) -> None:
    """
    information gain(IG)を元にして不要語を抽出する．
    各カテゴリckに対して単語tj毎にInformation Gainを以下のように定義

    IG(tj, ck) = p(tj, ck)(log(p(tj, ck)) - log(p(tj)) - log(p(ck)))
                    + p(!tj, ck)(log(p(!tj, ck)) - log(p(!tj)) - log(p(ck)))
    """

    # rdd(key, rows)
    category_group = raw_rdd.groupBy(
        # row(category, words)
        lambda row: row['category']
    )
    category_hist = category_group.map(
        lambda key_and_rows: len(key_and_ros[1]) # length of rows
    )


if __name__ == "__main__":
