import os
from pyspark.sql import SparkSession

def create_spark_session(app_name, master="spark://127.0.0.1:7077"):
    jars_dir = "/home/lillie/Documents/Study/Improved-RF-for-Real-time-IDS-with-Spark/venv/Lib/site-packages/pyspark/jars"
    jars_list = [
        "hadoop-aws-3.3.6.jar",
        "aws-java-sdk-bundle-1.11.1026.jar",
        "guava-30.1-jre.jar",
        "hadoop-common-3.3.6.jar",
        "hadoop-client-3.3.6.jar",
        "xgboost4j-spark_2.12-2.0.3.jar",  # Thêm JAR cho XGBoost
        "xgboost4j_2.12-2.0.3.jar"
    ]
    jars = ",".join([os.path.join(jars_dir, jar) for jar in jars_list])

    spark = SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .config("spark.jars", jars) \
        .config("spark.driver.host", "host.docker.internal") \
        .config("spark.driver.bindAddress", "0.0.0.0") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "2") \
        .config("spark.executor.instances", "1") \
        .config("spark.network.timeout", "1200s") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "2g") \
        .config("spark.shuffle.file.buffer", "4096k") \
        .config("spark.default.parallelism", "24") \
        .config("spark.shuffle.io.maxRetries", "10") \
        .config("spark.shuffle.io.retryWait", "60s") \
        .config("spark.hadoop.fs.s3a.block.size", "33554432") \
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
        .config("spark.hadoop.fs.s3a.access.key", "admin") \
        .config("spark.hadoop.fs.s3a.secret.key", "password") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
        .config("spark.hadoop.fs.s3a.attempts.maximum", "1") \
        .config("spark.shuffle.reduceLocality.enabled", "false") \
        .config("spark.shuffle.service.enabled", "true") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.broadcastTimeout", "1200") \
        .config("spark.sql.autoBroadcastJoinThreshold", 10 * 1024 * 1024) \
        .config("spark.executor.heartbeatInterval", "60s") \
        .config("spark.storage.blockManagerSlaveTimeoutMs", "1800000") \
        .getOrCreate()
    
    return spark