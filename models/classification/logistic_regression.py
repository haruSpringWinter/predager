from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from parser.csv_parser import load_as_df
from schema.twitter_schema import twitter_schema
from preprocess.preprocessor import shape_df
from featurizer.word2vec_featurizer import Word2VecFeaturizer
from featurizer.mult_featurizer import MultiFeaturizer
from featurizer.bert_featurizer import BertFeaturizer
from featurizer.tfidf_featurizer import TfidfFeaturizer
from featurizer.onehot_featurizer import OneHotFeaturizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if __name__ == '__main__':

    spark = SparkSession.builder \
        .appName('Spark SQL and DataFrame') \
        .getOrCreate()
    # Load training data
    # dataPath = '../../example_data/twitter/20190528sentences_data_integrated.csv'
    dataPath = '../../example_data/twitter_2020-03-10.csv'
    df = load_as_df(dataPath, twitter_schema)
    converted_df = shape_df(spark, df).drop("age")
    converted_df.show(3)
    # model_path = "../../param/word2vec/entity_vector/entity_vector.model.bin"
    # wv = Word2VecFeaturizer(spark, model_path)
    # feat_df = wv.featurize(converted_df)
    model_path = "../../param/word2vec/twitter_model/w2v_gensim/word2vec_tweet.model"
    wv_tweet = Word2VecFeaturizer(spark, model_path, False)
    # feat_df = wv_tweet.featurize(converted_df)
    # model_path = "../../param/word2vec/niconico_model/nico_vec.bin"
    # wv_nico = Word2VecFeaturizer(spark, model_path, False)
    # feat_df = wv_nico.featurize(converted_df)
    # model_path = "../../param/bert/Japanese_L-24_H-1024_A-16_E-30_BPE_WWM_transformers"
    # bert = BertFeaturizer(spark, model_path)
    # feat_df = bert.featurize(converted_df)
    converted_df2 = shape_df(spark, df, 'nagisa', ['補助記号']).drop("age")
    tfidf = TfidfFeaturizer(spark)
    # feat_df = tfidf.featurize(converted_df2)
    # onehot = OneHotFeaturizer(spark)
    # feat_df = onehot.featurize(converted_df)
    multi_feat = MultiFeaturizer(spark, [wv_tweet, tfidf], [converted_df, converted_df2])
    feat_df = multi_feat.featurize()

    (training, test) = feat_df.randomSplit([0.8, 0.2], seed=3)
    # lr = LogisticRegression(maxIter=100, regParam=0.06, elasticNetParam=0.2)
    lr = LogisticRegression(maxIter=30, regParam=0.001, elasticNetParam=0.2)

    # Fit the model
    lrModel = lr.fit(training)

    summary = lrModel.summary
    print("training accuracy: "+str(summary.accuracy))

    # Make predictions.
    predictions = lrModel.transform(test)

    # Select example rows to display.
    predictions.select("prediction", "label", "features").show()

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("test accuracy: " + str(accuracy))