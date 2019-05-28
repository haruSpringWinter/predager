import MeCab
import numpy as np

def to_morph(sentence):
    m = MeCab.Tagger("-Ochasen")
    node = m.parseToNode(sentence)
    list = []
    while node:
        word = node.surface
        node = node.next
        list.append(word)
    return np.array(list)
