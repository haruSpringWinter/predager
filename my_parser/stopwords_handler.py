import os
import urllib.request
import itertools
from collections import Counter

from gensim import corpora

pwd = os.path.dirname(os.path.abspath(__file__))
stopwords = []

with open(pwd + '/stopwords.txt', 'r') as f:
    for line in f:
        if line != '':
            stopwords.append(str(line.strip()))


def maybe_download(path: str) -> None:
    url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    if os.path.exists(path):
        print('File already exists.')
    else:
        print('Downloading...')
        # Download the file from `url` and save it locally under `file_name`:
        urllib.request.urlretrieve(url, path)


def create_dictionary(texts: list) -> dict:
    dictionary = corpora.Dictionary(texts)
    return dictionary


def remove_stopwords(words: list, stopwords: list) -> list:
    words = [word for word in words if word not in stopwords]
    return words


def most_common(docs: list, n: int = 100) -> list:
    fdist = Counter()
    for doc in docs:
        for word in doc:
            print(word)
            fdist[word] += 1
    common_words = {word for word, freq in fdist.most_common(n)}
    print('{}/{}'.format(n, len(fdist)))
    return common_words


def get_stop_words(docs: list, n: int = 100, min_freq: int = 1):
    fdist = Counter()
    for doc in docs:
        for word in doc:
            fdist[word] += 1
    common_words = {word for word, freq in fdist.most_common(n)}
    rare_words = {word for word, freq in fdist.items() if freq <= min_freq}
    stopwords = common_words.union(rare_words)
    print('{}/{}'.format(len(stopwords), len(fdist)))
    return stopwords

if __name__ == "__main__":
    from featurizer import to_morph

    sentences = [
        "出現頻度による方式では、テキスト内の単語頻度をカウントし、高頻度(時には低頻度)の単語をテキストから除去します。高頻度の単語を除去するのは、それらの単語がテキスト中で占める割合が高い一方、役に立たないからです。以下の図はある英語の本の最も頻出する50単語の累計頻度をプロットしたものです:",
        "50単語を見てみると、the や of、カンマのような文書分類等に役に立たなさそうな単語がテキストの50%近くを占めていることがわかります。出現頻度による方式ではこれら高頻度語をストップワードとしてテキストから取り除きます。",
        "単語のベクトル表現では、文字列である単語をベクトルに変換する処理を行います。なぜ文字列からベクトルに変換するのかというと、文字列は可変長で扱いにくい、類似度の計算がしにくい等の理由が挙げられます。ベクトル表現するのにも様々な手法が存在するのですが、以下の2つについて紹介します。",
        "one-hot表現はシンプルですが、ベクトル間の演算で何も意味のある結果を得られないという弱点があります。たとえば、単語間で類似度を計算するために内積を取るとしましょう。one-hot表現では異なる単語は別の箇所に1が立っていてその他の要素は0なので、異なる単語間の内積を取った結果は0になってしまいます。これは望ましい結果とは言えません。また、1単語に1次元を割り当てるので、ボキャブラリ数が増えると非常に高次元になってしまいます。",
        "分散表現を使うことでone-hot表現が抱えていた問題を解決できます。たとえば、ベクトル間の演算で単語間の類似度を計算することができるようになります。上のベクトルを見ると、python と ruby の類似度は python と word の類似度よりも高くなりそうです。また、ボキャブラリ数が増えても各単語の次元数を増やさずに済みます。",
        "この説では前処理にどれほどの効果があるのか検証します。具体的には文書分類タスクに前処理を適用した場合としていない場合で分類性能と実行時間を比較しました。結果として、前処理をすることで分類性能が向上し、実行時間は半分ほどになりました。"
    ]

    morphs = [to_morph(sentence) for sentence in sentences]
    docs_with_surfaces = []

    for nodes in morphs:
        surfaces = []
        for node in nodes:
            surfaces.append(node[0])
        docs_with_surfaces.append(surfaces)

    most_commons_10 = most_common(docs_with_surfaces, 100)
    print("most common terms:")
    for elem in most_commons_10:
        print(elem)