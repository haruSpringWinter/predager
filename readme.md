# How to setup environment

Clone this repository and then `pip install -r pyp_list.txt` under the project.

## Requirements

Basically refer `pyp_list.txt`.
Sometimes, you need initial settings to install module such as matplotlib and pyspark.

We wanna use docker to make installation easier.


## Setting matplotlib
You need to add `backend : Qt4Agg` to `matplotlibrc`.
You can find `matplotlibrc` by 

```
$ python
> import matplotlib
> matplotlib.matplotlib_fname()
```

## Setting pyspark
```
brew update
brew install apache-spark
```

and then

```
export JAVA_HOME=`/usr/libexec/java_home -v ${your_java_version}`
export SPARK_HOME="/usr/local/Cellar/apache-spark/${your_spark_version}/libexec/"
export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/build:$PYTHONPATH
export PYTHONPATH=$SPARK_HOME/python/lib/py4j-${your_py4j_version}-src.zip:$PYTHONPATH
```
