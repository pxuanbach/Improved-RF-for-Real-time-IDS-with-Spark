#!/bin/bash

sudo yum install java-1.8.0-openjdk-devel -y

sudo yum install python-pip -y

pip install numpy==1.26.4 --no-cache-dir
pip install pandas xgboost pyarrow scikit-learn --no-cache-dir

wget https://dlcdn.apache.org/spark/spark-3.5.5/spark-3.5.5-bin-hadoop3.tgz

tar -xvf spark-3.5.5-bin-hadoop3.tgz

rm -f /root/spark-3.5.5-bin-hadoop3/jars/hadoop-*.jar && \
    rm -f /root/spark-3.5.5-bin-hadoop3/jars/aws-java-sdk-*.jar

wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.6/hadoop-aws-3.3.6.jar -O /root/spark-3.5.5-bin-hadoop3/jars/hadoop-aws-3.3.6.jar && \
wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.261/aws-java-sdk-bundle-1.12.261.jar -O /root/spark-3.5.5-bin-hadoop3/jars/aws-java-sdk-bundle-1.12.261.jar && \
wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-common/3.3.6/hadoop-common-3.3.6.jar -O /root/spark-3.5.5-bin-hadoop3/jars/hadoop-common-3.3.6.jar && \
wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-client/3.3.6/hadoop-client-3.3.6.jar -O /root/spark-3.5.5-bin-hadoop3/jars/hadoop-client-3.3.6.jar && \
wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-client-api/3.3.6/hadoop-client-api-3.3.6.jar -O /root/spark-3.5.5-bin-hadoop3/jars/hadoop-client-api-3.3.6.jar && \
wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-client-runtime/3.3.6/hadoop-client-runtime-3.3.6.jar -O /root/spark-3.5.5-bin-hadoop3/jars/hadoop-client-runtime-3.3.6.jar && \
wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-yarn-server-web-proxy/3.3.6/hadoop-yarn-server-web-proxy-3.3.6.jar -O /root/spark-3.5.5-bin-hadoop3/jars/hadoop-yarn-server-web-proxy-3.3.6.jar

wget https://repo1.maven.org/maven2/ml/dmlc/xgboost4j-spark_2.12/2.0.3/xgboost4j-spark_2.12-2.0.3.jar -O /root/spark-3.5.5-bin-hadoop3/jars/xgboost4j-spark_2.12-2.0.3.jar && \
    wget https://repo1.maven.org/maven2/ml/dmlc/xgboost4j_2.12/2.0.3/xgboost4j_2.12-2.0.3.jar -O /root/spark-3.5.5-bin-hadoop3/jars/xgboost4j_2.12-2.0.3.jar

rm -f /root/spark-3.5.5-bin-hadoop3/jars/guava-*.jar && \
    wget  https://repo1.maven.org/maven2/com/google/guava/guava/30.1-jre/guava-30.1-jre.jar -O /root/spark-3.5.5-bin-hadoop3/jars/guava-30.1-jre.jar

wget  https://repo1.maven.org/maven2/ml/dmlc/xgboost4j-spark_2.12/2.0.3/xgboost4j-spark_2.12-2.0.3.jar -O /root/spark-3.5.5-bin-hadoop3/jars/xgboost4j-spark_2.12-2.0.3.jar && \
    wget  https://repo1.maven.org/maven2/ml/dmlc/xgboost4j_2.12/2.0.3/xgboost4j_2.12-2.0.3.jar -O /root/spark-3.5.5-bin-hadoop3/jars/xgboost4j_2.12-2.0.3.jar

export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk && \
export SPARK_HOME=/root/spark-3.5.5-bin-hadoop3 && \
export PATH=$PATH:$SPARK_HOME/bin

spark-3.5.5-bin-hadoop3/sbin/start-worker.sh spark://103.153.74.216:7077
