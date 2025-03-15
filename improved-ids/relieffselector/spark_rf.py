
# Import necessary libraries
import time
import json
import os
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col, when, lit
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Import ReliefFSelector
from relieffselector import ReliefFSelector

# Initialize Spark Session
spark = SparkSession.builder \
     .appName("pyspark-notebook")\
    .master("spark://127.0.0.1:7077")\
    .config("spark.driver.host", "host.docker.internal") \
    .config("spark.driver.bindAddress", "0.0.0.0")\
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.cores", "1") \
    .config("spark.executor.instances", "2")\
    .config("spark.network.timeout", "800s") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "2g") \
    .config("spark.default.parallelism", "24") \
    .config("spark.shuffle.io.maxRetries", "10") \
    .config("spark.shuffle.io.retryWait", "60s") \
    .config("spark.hadoop.fs.s3a.block.size", "33554432") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
    .config("spark.hadoop.fs.s3a.access.key", "admin") \
    .config("spark.hadoop.fs.s3a.secret.key", "password") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")\
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
    .config("spark.hadoop.fs.s3a.attempts.maximum", "0") \
    .getOrCreate()

# Path của dataset sau khi chọn đặc trưng từ spark_relieffselector.py
data_path = "selected_features_dataset.csv"

# Kiểm tra xem file dữ liệu đã được tạo bởi spark_relieffselector.py hay chưa
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found. Run spark_relieffselector.py first.")
    exit(1)

df_pandas = pd.read_csv(data_path)

# Chuyển đổi Pandas DataFrame thành Spark DataFrame
df_vector = spark.createDataFrame(df_pandas)

# Chuyển đổi cột `features` từ danh sách về Vector
assembler = VectorAssembler(inputCols=[c for c in df_vector.columns if c != "label"], outputCol="features")
df_vector = assembler.transform(df_vector).select("features", "label")

print("Loaded selected features dataset from CSV.")

# Train-Test Split
train_df, test_df = df_vector.randomSplit([0.8, 0.2], seed=42)

# Train Random Forest Model
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100, maxDepth=10, seed=42)
rf_model = rf.fit(train_df)

# Predictions
predictions = rf_model.transform(test_df)

# Evaluate Model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)
precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="precisionByLabel")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="recallByLabel")

precision = precision_evaluator.evaluate(predictions)
recall = recall_evaluator.evaluate(predictions)

# Print Results
print(f"F1-score: {f1_score:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Show sample predictions
predictions.select("features", "label", "prediction").show(10)
