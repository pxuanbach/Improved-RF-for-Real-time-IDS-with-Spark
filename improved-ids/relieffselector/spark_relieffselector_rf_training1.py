import time
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col, sum, count, when, lit, udf
from pyspark.ml.feature import MinMaxScaler, VectorAssembler, StringIndexer
import numpy as np
import pandas as pd
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from relieffselector import ReliefFSelector
import os
print("Running on:", os.name)  # 'posix' nếu là Linux (container), 'nt' nếu là Windows
print("Current working directory:", os.getcwd())
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Khởi tạo Spark session
spark: SparkSession = SparkSession.builder\
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
    .config("spark.hadoop.io.nativeio.useLegacyFileSystem", "true")\
    .getOrCreate()

# Tạo DataFrame mock (Dữ liệu giả lập)
data = [
    (0.0, Vectors.dense([1.0, 2.0, 3.0])),
    (1.0, Vectors.dense([4.0, 5.0, 6.0])),
    (0.0, Vectors.dense([7.0, 8.0, 9.0])),
    (1.0, Vectors.dense([10.0, 11.0, 12.0])),
]
df_mock = spark.createDataFrame(data, ["label", "features"])

# Train mô hình
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=10, maxDepth=5, seed=42)
rf_model = rf.fit(df_mock)

# Đường dẫn local để lưu model
local_model_path = "file:///tmp/mock_rf_model"

# Xóa thư mục cũ nếu tồn tại
os.system(f"rm -rf {local_model_path.replace('file://', '')}")

# Lưu model
try:
    rf_model.write().overwrite().save(local_model_path)
    print(f"✅ Model saved locally at {local_model_path}")
except Exception as e:
    print(f"❌ Failed to save model: {e}")
