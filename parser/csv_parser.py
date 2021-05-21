import numpy as np
from typing import *
from pyspark.sql.types import *
from pyspark.sql import SparkSession, DataFrame
from pyspark import SparkContext, SparkConf
from schema.twitter_schema import twitter_schema

conf = SparkConf().setAll(
    # TODO: define values by environment variable rather than hard-coding
    [
        ('spark.executor.memory', '4g'),
        ('spark.driver.memory', '4g'),
        ('spark.cores.max', '4')
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
