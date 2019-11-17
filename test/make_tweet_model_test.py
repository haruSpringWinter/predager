import sys
func_path = '../models/embed/'
sys.path.append(func_path)
from extract_word2vec import extract_from_txt

path = '../example_data/hanage_wakati.txt'
extract_from_txt(path)
