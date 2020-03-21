import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import MeCab


def featurize(data_path:str, save:bool = False, saved_path:str = ""):
    model = KeyedVectors.load_word2vec_format("../models/embed/entity_vector.model.bin", binary=True)
    word2vec = model.wv
    mecab = MeCab.Tagger('-Ochasen')
    mecab.parse('')
    df = pd.read_csv(data_path)
    age_df = df['age']
    sex_df = df['sex']
    sentence_df = df['sentence']
    vec_list = []
    for sentence in sentence_df:
        cnt = 0
        sum_vec = np.array([0.0] * 200)
        result_vec = sum_vec
        node = mecab.parseToNode(sentence)
        while node:
            word = node.surface
            if word in word2vec.vocab:
                cnt += 1
                sum_vec += word2vec.get_vector(word)
            node = node.next
        if cnt != 0:
            result_vec = sum_vec / cnt
        vec_list.append(result_vec)
    result_df = pd.DataFrame({
        'age': age_df,
        'sex': sex_df,
        'feature': vec_list
    })
    print(result_df)
