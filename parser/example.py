import csv_parser
import preprocessor

path = 'example_data/20190528sentences_data_integrated.csv'
df = csv_parser.load_as_df(path)
df.show(3)

converted = preprocessor.convert_df_to_feature(df)
sample = converted.show(3)