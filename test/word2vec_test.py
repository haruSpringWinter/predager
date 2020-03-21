from gensim.models import KeyedVectors

print("モデルを読み込んでいます...")
model = KeyedVectors.load_word2vec_format("../models/embed/entity_vector.model.bin", binary=True)
print("読み込みが完了しました．")

word2vec = model.wv
if __name__ == '__main__':
    print("類似度を確かめたい単語を入力してください: ", end="")
    str = input()
    if str in word2vec.vocab:
        results = model.wv.most_similar(positive=[str])
        for result in results:
            print(result)
    else:
        print("入力された単語が辞書に存在しませんでした．")

