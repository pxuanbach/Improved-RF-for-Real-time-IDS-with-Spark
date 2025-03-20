FROM bitnami/spark:3.5.1

USER root

RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/* \
    && pip install \
        numpy==1.26.4 \
        pandas==2.2.1 \
        scikit-learn \
        pyarrow==15.0.0

# download hadoop-aws-3.3.6.jar and aws-java-sdk-bundle-1.11.1026.jar
	#hadoop-common, hadoop-client
RUN wget -q https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.6/hadoop-aws-3.3.6.jar -O /opt/bitnami/spark/jars/hadoop-aws-3.3.6.jar && \
    wget -q https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.11.1026/aws-java-sdk-bundle-1.11.1026.jar -O /opt/bitnami/spark/jars/aws-java-sdk-bundle-1.11.1026.jar && \
	wget -q https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-common/3.3.6/hadoop-common-3.3.6.jar -O /opt/bitnami/spark/jars/hadoop-common-3.3.6.jar && \
    wget -q https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-client/3.3.6/hadoop-client-3.3.6.jar -O /opt/bitnami/spark/jars/hadoop-client-3.3.6.jar


RUN rm -f /opt/bitnami/spark/jars/guava-14.0.1.jar && \
    rm -f /opt/bitnami/spark/jars/guava-27.0-jre.jar && \
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
