FROM bitnami/spark:3.5.1

USER root

RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# download hadoop-aws-3.2.1.jar and aws-java-sdk-bundle-1.11.1026.jar
RUN wget -q https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.2.1/hadoop-aws-3.2.1.jar -O /opt/bitnami/spark/jars/hadoop-aws-3.2.1.jar && \
    wget -q https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.11.1026/aws-java-sdk-bundle-1.11.1026.jar -O /opt/bitnami/spark/jars/aws-java-sdk-bundle-1.11.1026.jar

RUN rm -f /opt/bitnami/spark/jars/guava-14.0.1.jar && \
    wget -q https://repo1.maven.org/maven2/com/google/guava/guava/27.0-jre/guava-27.0-jre.jar -O /opt/bitnami/spark/jars/guava-27.0-jre.jar

USER 1001
