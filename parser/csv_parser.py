import numpy as np
from typing import *
from pyspark.sql.types import *
from pyspark.sql import SparkSession, DataFrame
from pyspark import SparkContext, SparkConf

conf = SparkConf().setAll(
        # TODO: define values by environment variable rather than hard-coding
        [
            ('spark.executor.memory', '4g'),
            ('spark.driver.memory', '4g'),
            ('spark.cores.max', '4')
        ]
    )
sc = SparkContext(conf = conf)
ss = SparkSession(sc)

# age,sex,sentence,ur
twitter_schema = StructType([
    StructField('age', IntegerType(), False),
    StructField('sex', StringType(), False),
    StructField('sentence', StringType(), False),
    StructField('url', StringType(), False)
])

def load_as_df(path: str) -> DataFrame:
    df = ss.read.csv(path, header = True, schema = twitter_schema)
    return df
