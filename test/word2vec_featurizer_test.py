from featurizer.word2vec_featurizer import featurize
if __name__ == '__main__':
    path = '../example_data/twitter/20190528sentences_data_integrated.csv'
    featurize(path)
