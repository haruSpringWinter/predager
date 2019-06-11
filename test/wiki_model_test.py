from gensim.models import word2vec

model = word2vec.Word2Vec.load("../models/wiki.model")

print("類似度を確かめたい単語を入力してください: ", end="")
str = input()
results = model.wv.most_similar(positive=[str])
for result in results:
    print(result)
