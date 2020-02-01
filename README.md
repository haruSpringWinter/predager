# predager
Predict age from sentences

### Verified environments
* Docker 2.2.0.0
* Docker-compose v1.25.2

# How to setup environment
```
docker-compose up -d    // build containers and run the command
docker-compose logs -f  // check the logs
docker-compose down     // stop the containers
```

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

# For Developers

## Using pyspark for local Mac

It might have an error about forking the process. 

This issue has a workaround by setting environmental variable such as:
'OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES'.

See also:
https://blog.phusion.nl/2017/10/13/why-ruby-app-servers-break-on-macos-high-sierra-and-what-can-be-done-about-it/
Predict age from sentences