from pyspark.sql import SparkSession
import lightgbm as lgb
from parser.csv_parser import load_as_df
from schema.twitter_schema import twitter_schema
from preprocess.preprocessor import shape_df
from preprocess import pyspark_pandas_converter
from featurizer.word2vec_featurizer import Word2VecFeaturizer
from featurizer.mult_featurizer import MultiFeaturizer
from featurizer.bert_featurizer import BertFeaturizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from featurizer.tfidf_featurizer import TfidfFeaturizer
from sklearn.metrics import accuracy_score


if __name__ == '__main__':

    spark = SparkSession.builder.appName("MyApp") \
        .config("spark.jars.packages", "com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc3") \
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
        .getOrCreate()

    # Load training data
    # dataPath = '../../example_data/twitter/20190528sentences_data_integrated.csv'
    dataPath = '../../example_data/twitter_2020-03-10.csv'
    df = load_as_df(dataPath, twitter_schema)
    converted_df = shape_df(spark, df, 'nagisa').drop("age")
    # converted_df = shape_df(spark, df).drop("age")
    converted_df.show(3)
    # model_path = "../../param/word2vec/entity_vector/entity_vector.model.bin"
    # wv = Word2VecFeaturizer(spark, model_path)
    # feat_df = wv.featurize(converted_df)
    model_path = "../../param/word2vec/twitter_model/w2v_gensim/word2vec_tweet.model"
    # model_path = "../../param/word2vec/niconico_model/nico_vec.bin"
    wv_tweet = Word2VecFeaturizer(spark, model_path, True)
    feat_df = wv_tweet.featurize(converted_df)
    # model_path = "../../param/bert/Japanese_L-24_H-1024_A-16_E-30_BPE_WWM_transformers"
    # bert = BertFeaturizer(spark, model_path)
    # feat_df = bert.featurize(converted_df)
    # Split the data into training and test sets (30% held out for testing)
    # multi_feat = MultiFeaturizer(spark, [wv, wv_tweet])
    # feat_df = multi_feat.featurize(converted_df)
    # converted_df2 = shape_df(spark, df, 'nagisa', ['補助記号']).drop("age")
    # tfidf = TfidfFeaturizer(spark)
    # feat_df = tfidf.featurize(converted_df2)
    # onehot = OneHotFeaturizer(spark)
    # feat_df = onehot.featurize(converted_df)
    # multi_feat = MultiFeaturizer(spark, [wv_tweet, tfidf], [converted_df, converted_df2])
    # feat_df = multi_feat.featurize()
    feat_df.show(3)
    (trainingData, testData) = feat_df.randomSplit([0.8, 0.2], seed=1)
    pd_train = pyspark_pandas_converter.convert_spark_to_pd(trainingData)
    pd_test = pyspark_pandas_converter.convert_spark_to_pd(testData)
    trainX = pd_train.drop("labels", axis=1)
    trainY = pd_train['labels']
    testX = pd_test.drop("labels", axis=1)
    testY = pd_test['labels']
    # 3. call `fit`. (fit のときにはたんに事前に作った data-frame を入れる)
    model = lgb.LGBMClassifier(
        objective='binary',
        num_leaves=1024,
        reg_lambda=0.1,
        reg_alpha=0.004,
        min_child_samples=20,
        max_depth=10,
        num_iterations=1000
    )
    clf = model.fit(trainX, trainY)

    predict_train = model.predict(trainX)
    predict_test = model.predict(testX)

    print('train accuracy = \t {}'.format(accuracy_score(trainY, predict_train)))
    print('test accuracy = \t {}'.format(accuracy_score(testY, predict_test)))
