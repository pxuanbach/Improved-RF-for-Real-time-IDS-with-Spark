FROM bitnami/spark:3.5.1

USER root

RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/* \
    && pip install \
        numpy==1.26.4 \
        pandas==2.2.1 \
        scikit-learn \
        pyarrow==15.0.0 \ 
        imbalanced-learn==0.13.0 \
        xgboost==3.0.0

# Xóa các JAR Hadoop cũ trong image
RUN rm -f /opt/bitnami/spark/jars/hadoop-*.jar && \
    rm -f /opt/bitnami/spark/jars/aws-java-sdk-*.jar

# Tải các JAR Hadoop và AWS SDK phiên bản mới
RUN wget -q https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.6/hadoop-aws-3.3.6.jar -O /opt/bitnami/spark/jars/hadoop-aws-3.3.6.jar && \
    wget -q https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.261/aws-java-sdk-bundle-1.12.261.jar -O /opt/bitnami/spark/jars/aws-java-sdk-bundle-1.12.261.jar && \
    wget -q https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-common/3.3.6/hadoop-common-3.3.6.jar -O /opt/bitnami/spark/jars/hadoop-common-3.3.6.jar && \
    wget -q https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-client/3.3.6/hadoop-client-3.3.6.jar -O /opt/bitnami/spark/jars/hadoop-client-3.3.6.jar && \
    wget -q https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-client-api/3.3.6/hadoop-client-api-3.3.6.jar -O /opt/bitnami/spark/jars/hadoop-client-api-3.3.6.jar && \
    wget -q https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-client-runtime/3.3.6/hadoop-client-runtime-3.3.6.jar -O /opt/bitnami/spark/jars/hadoop-client-runtime-3.3.6.jar && \
    wget -q https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-yarn-server-web-proxy/3.3.6/hadoop-yarn-server-web-proxy-3.3.6.jar -O /opt/bitnami/spark/jars/hadoop-yarn-server-web-proxy-3.3.6.jar

# Tải JAR cho XGBoost4J-Spark
RUN wget -q https://repo1.maven.org/maven2/ml/dmlc/xgboost4j-spark_2.12/2.0.3/xgboost4j-spark_2.12-2.0.3.jar -O /opt/bitnami/spark/jars/xgboost4j-spark_2.12-2.0.3.jar && \
    wget -q https://repo1.maven.org/maven2/ml/dmlc/xgboost4j_2.12/2.0.3/xgboost4j_2.12-2.0.3.jar -O /opt/bitnami/spark/jars/xgboost4j_2.12-2.0.3.jar

# Xóa các phiên bản guava cũ và tải phiên bản mới
RUN rm -f /opt/bitnami/spark/jars/guava-*.jar && \
    wget -q https://repo1.maven.org/maven2/com/google/guava/guava/30.1-jre/guava-30.1-jre.jar -O /opt/bitnami/spark/jars/guava-30.1-jre.jar

# Configure External Shuffle Service
RUN echo "SPARK_SHUFFLE_SERVICE_ENABLED=true" >> /opt/bitnami/spark/conf/spark-env.sh && \
    echo "SPARK_SHUFFLE_SERVICE_PORT=7337" >> /opt/bitnami/spark/conf/spark-env.sh && \
    chmod +x /opt/bitnami/spark/conf/spark-env.sh

RUN echo "spark.shuffle.service.enabled true" >> /opt/bitnami/spark/conf/spark-defaults.conf && \
    echo "spark.shuffle.service.port 7337" >> /opt/bitnami/spark/conf/spark-defaults.conf && \
    echo "spark.dynamicAllocation.enabled true" >> /opt/bitnami/spark/conf/spark-defaults.conf && \
    echo "spark.dynamicAllocation.shuffleTracking.enabled true" >> /opt/bitnami/spark/conf/spark-defaults.conf && \
    echo "spark.dynamicAllocation.minExecutors 1" >> /opt/bitnami/spark/conf/spark-defaults.conf && \
    echo "spark.dynamicAllocation.maxExecutors 3" >> /opt/bitnami/spark/conf/spark-defaults.conf

COPY minio_is_ready.sh /opt/bitnami/spark/minio_is_ready.sh
RUN chmod +x /opt/bitnami/spark/minio_is_ready.sh

USER 1001