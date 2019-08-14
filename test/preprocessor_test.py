import numpy as np
import sys
func_path = '../parser/'
sys.path.append(func_path)
from preprocessor import to_morph

sentence = "ちょっと一人にしておいてくれないか。君と話したい気分じゃないんだ。"
nda = to_morph(sentence)
shape =  nda.shape
for i in range(0, len(nda)):
    print(nda[i])
print("")
