import csv_parser, nlp_featurizer
import os

pwd = os.path.dirname(os.path.abspath(__file__))
path = pwd + '/../example_data/20190528sentences_data_integrated.csv'
df = csv_parser.load_as_df(path)
df.show(3)

converted = nlp_featurizer.convert_df_to_feature(df)
sample = converted.take(3)
for e in sample:
    print(e)