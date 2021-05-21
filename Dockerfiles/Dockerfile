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

# Setting up spark for PySpark
RUN wget http://ftp.riken.jp/net/apache/spark/spark-2.4.7/spark-2.4.7-bin-hadoop2.7.tgz
RUN tar zxvf spark-2.4.7-bin-hadoop2.7.tgz
RUN sudo mv spark-2.4.7-bin-hadoop2.7 /usr/local/
RUN sudo ln -s /usr/local/spark-2.4.7-bin-hadoop2.7 /usr/local/spark
RUN export SPARK_HOME=/usr/local/spark
RUN export PATH=$PATH:$SPARK_HOME/bin
RUN export PYSPARK_PYTHON=/usr/bin/python3
RUN sudo update-java-alternatives --set java-1.8.0-openjdk-amd64

RUN pip3 install mecab==0.996 requests==2.22.0 gensim==3.8.1 gensim==3.8.1 matplotlib==3.1.1 nltk==3.4.5  numpy==1.17.4 pandas==0.25.3 pyspark==2.4.7 scipy==1.3.2 torch==1.3.1 torchvision==0.4.2 bs4==0.0.1

COPY ./ /app

WORKDIR /app
