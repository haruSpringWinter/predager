from pyspark.sql.types import *
from pyspark.sql import SparkSession, DataFrame
from pyspark import SparkContext, SparkConf
from schema import twitter_schema
from utils.predager_config import predager_config

conf = SparkConf().setAll(
    # TODO: define values by environment variable rather than hard-coding
    [
        ('spark.executor.memory', predager_config.spark.executor.get('memory')),
        ('spark.driver.memory', predager_config.spark.driver.get('memory')),
        ('spark.cores.max', predager_config.spark.cores.get('max'))
    ]
)
sc = SparkContext(conf=conf)
ss = SparkSession(sc).builder.master('local[*]').getOrCreate()


def load_as_df(path: str, schema: StructType) -> DataFrame:
    df = ss.read.csv(path, header=True, schema=schema)
    return df


if __name__ == "__main__":
    import os

    pwd = os.path.dirname(os.path.abspath(__file__))
    path = pwd + '/../example_data/twitter/20190528sentences_data_integrated.csv'
    df = load_as_df(path, twitter_schema)
    df.show(3)
