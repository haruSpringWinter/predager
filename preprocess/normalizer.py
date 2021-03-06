import re
import unicodedata


import nltk
from nltk.corpus import wordnet


def normalize(text: str) -> str:
    normalized_text = normalize_unicode(text)
    normalized_text = normalize_number(normalized_text)
    normalized_text = lower_text(normalized_text)
    return normalized_text


def lower_text(text: str) -> str:
    return text.lower()


def normalize_unicode(text: str, form: str = 'NFKC') -> str:
    normalized_text = unicodedata.normalize(form, text)
    return normalized_text

'''
Wordnetから語形変化を伴わない標準的な表現を得る
'''
def lemmatize_term(term: str, pos=None) -> str:
    if pos is None:
        synsets = wordnet.synsets(term)
        if not synsets:
            return term
        pos = synsets[0].pos()
        if pos == wordnet.ADJ_SAT:
            pos = wordnet.ADJ
    return nltk.WordNetLemmatizer().lemmatize(term, pos=pos)


def normalize_number(text):
    """
    pattern = r'\d+'
    replacer = re.compile(pattern)
    result = replacer.sub('0', text)
    """
    # 連続した数字を0で置換
    replaced_text = re.sub(r'\d+', '0', text)
    return replaced_text

if __name__ == "__main__":
    example_txt = "Hello!! My name is 細馬. I WANNA BE THE COOL GUY 000000."
    normalized = normalize(example_txt)
    print("example:    " + example_txt)
    print("normalized: " + normalized)