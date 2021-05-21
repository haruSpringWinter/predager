from pyspark.sql.types import *

twitter_schema = StructType([
    StructField('age', IntegerType(), False),
    StructField('sex', StringType(), False),
    StructField('sentence', StringType(), False)
])
