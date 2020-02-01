FROM ubuntu:18.04

ENV LANG ja_JP.UTF-8

RUN apt-get update && apt-get install -y\
  mecab \
  mecab-ipadic-utf8 \
  libmecab-dev \
  language-pack-ja \
  python3-pip \
  python3-dev \
  python3-setuptools \
  build-essential \
  curl \
  wget \
  git \
  xz-utils \
  sudo \
  maven \
  openjdk-8-jdk \
  && apt-get clean

# Setting for PySpark
RUN wget http://ftp.riken.jp/net/apache/spark/spark-2.4.4/spark-2.4.4-bin-hadoop2.6.tgz
RUN tar zxvf spark-2.4.4-bin-hadoop2.6.tgz
RUN sudo mv spark-2.4.4-bin-hadoop2.6 /usr/local/
RUN sudo ln -s /usr/local/spark-2.4.4-bin-hadoop2.6 /usr/local/spark
RUN export SPARK_HOME=/usr/local/spark
RUN export PATH=$PATH:$SPARK_HOME/bin

RUN pip3 install mecab==0.996 requests==2.22.0 gensim==3.8.1 gensim==3.8.1 matplotlib==3.1.1 nltk==3.4.5  numpy==1.17.4 pandas==0.25.3 pyspark==2.4.4 scipy==1.3.2 torch==1.3.1 torchvision==0.4.2

COPY ./ /app

WORKDIR /app