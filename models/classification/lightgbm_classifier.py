from pyspark.sql import SparkSession
from parser.csv_parser import load_as_df
from schema.twitter_schema import twitter_schema
from preprocess.preprocessor import shape_df
from featurizer.word2vec_featurizer import Word2VecFeaturizer
from featurizer.mult_featurizer import MultiFeaturizer
from featurizer.bert_featurizer import BertFeaturizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from featurizer.tfidf_featurizer import TfidfFeaturizer


if __name__ == '__main__':

    print('chinge')
    spark = SparkSession.builder.appName("MyApp") \
        .config("spark.jars.packages", "com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc3") \
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
        .getOrCreate()
    print('hanage')
    from mmlspark.lightgbm import LightGBMClassifier
    print('munage')
    model = LightGBMClassifier(baggingSeed=1024,
                               learningRate=0.01,
                               numIterations=1000,
                               maxBin=8,
                               numLeaves=8,
                               metric='auc')
    print('mimige')

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
    (trainingData, testData) = feat_df.randomSplit([0.8, 0.2], seed=3)


    # 3. call `fit`. (fit のときにはたんに事前に作った data-frame を入れる)
    clf = model.fit(trainingData)

    predict_train = model.transform(trainingData)
    predict_test = model.transform(testData)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predict_train)
    print("train accuracy: " + str(accuracy))



    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predict_test)
    print("test accuracy: " + str(accuracy))