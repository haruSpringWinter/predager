import numpy as np
import sys
func_path = '../parser/'
sys.path.append(func_path)
from csv_parser import parse

path = '../example_data/20190528sentences_data_integrated.csv'
nda = parse(path)
shape = nda.shape
for i in range(0, shape[0]):
    for j in range(0, shape[1]):
        print(nda[i][j])
    print("")
