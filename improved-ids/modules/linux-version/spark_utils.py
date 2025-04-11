import os
from pyspark.sql import SparkSession

def create_spark_session(app_name, master="spark://103.153.74.216:7077"):
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
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "0.0.0.0") \
        .config("spark.driver.memory", "512m") \
        .config("spark.driver.cores", "1") \
        .config("spark.executor.memory", "2g") \
        .config("spark.executor.cores", "1") \
        .config("spark.executor.instances", "1") \
        .config("spark.network.timeout", "3600s") \
        .config("spark.default.parallelism", "24") \
        .config("spark.shuffle.io.maxRetries", "10") \
        .config("spark.shuffle.io.retryWait", "60s") \
        .config("spark.hadoop.fs.s3a.block.size", "33554432") \
        .config("spark.hadoop.fs.s3a.endpoint", "https://hn.ss.bfcplatform.vn") \
        .config("spark.hadoop.fs.s3a.access.key", "2W91ETIDZ1R2IH5QO8R2") \
        .config("spark.hadoop.fs.s3a.secret.key", "r8qxbjXPUi0AtcrSyOocN77qBpngAzO4lsMW4EPm") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
        .config("spark.hadoop.fs.s3a.attempts.maximum", "1") \
        .config("spark.shuffle.reduceLocality.enabled", "false") \
        .config("spark.shuffle.service.enabled", "true") \
        .config("spark.sql.broadcastTimeout", "1200") \
        .config("spark.executor.heartbeatInterval", "120s") \
        .config("spark.storage.blockManagerSlaveTimeoutMs", "1800000") \
        .config("spark.rpc.retry.wait", "60s")\
        .config("spark.rpc.numRetries", "10") \
        .getOrCreate()

    # Đặt mức log cho SparkContext
    spark.sparkContext.setLogLevel("DEBUG")
    return spark
