from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from parser.csv_parser import load_as_df
from schema.twitter_schema import twitter_schema
from preprocess.preprocessor import shape_df
from featurizer.word2vec_featurizer import Word2VecFeaturizer
from featurizer.bert_featurizer import BertFeaturizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import GBTClassifier


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
    feat_df = wv_tweet.featurize(converted_df)
    # model_path = "../../param/bert/Japanese_L-24_H-1024_A-16_E-30_BPE_WWM_transformers"
    # bert = BertFeaturizer(spark, model_path)
    # feat_df = bert.featurize(converted_df)
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = feat_df.randomSplit([0.8, 0.2], seed=2)

    gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=15)

    # Train model.  This also runs the indexers.
    model = gbt.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("test accuracy: " + str(accuracy))