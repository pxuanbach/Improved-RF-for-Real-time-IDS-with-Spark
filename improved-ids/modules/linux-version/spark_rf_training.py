from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
import json
import os

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

jars_dir = "/home/lillie/Documents/Study/Improved-RF-for-Real-time-IDS-with-Spark/venv/Lib/site-packages/pyspark/jars"
jars_list = [
    "hadoop-aws-3.3.6.jar",
    "aws-java-sdk-bundle-1.11.1026.jar",
    "guava-30.1-jre.jar",
    "hadoop-common-3.3.6.jar",
    "hadoop-client-3.3.6.jar"
]
jars = ",".join([os.path.join(jars_dir, jar) for jar in jars_list])

# Khởi tạo Spark session
spark: SparkSession = SparkSession.builder\
    .appName("pyspark-rf-training")\
    .master("spark://127.0.0.1:7077")\
    .config("spark.jars", jars) \
    .config("spark.driver.host", "host.docker.internal") \
    .config("spark.driver.bindAddress", "0.0.0.0")\
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.executor.instances", "3")\
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
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")\
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
    .config("spark.hadoop.fs.s3a.attempts.maximum", "1") \
    .config("spark.shuffle.reduceLocality.enabled", "false") \
    .config("spark.shuffle.service.enabled", "true") \
    .getOrCreate()

# Load dữ liệu đã giảm chiều
reduced_data_path = "s3a://mybucket/preprocessed_data/reduced_data.parquet"
df_reduced = spark.read.parquet(reduced_data_path)
print(bcolors.OKGREEN + f"✅ Loaded reduced data from {reduced_data_path}" + bcolors.ENDC)

# Load global_top_features và label_to_name
global_top_features_json = spark.read.text("s3a://mybucket/models/global_top_features").collect()[0]["value"]
global_top_features = json.loads(global_top_features_json)
print(bcolors.OKGREEN + f"✅ Loaded {len(global_top_features)} global top features: {global_top_features}" + bcolors.ENDC)

label_to_name_json = spark.read.text("s3a://mybucket/models/label_to_name").collect()[0]["value"]
label_to_name = {int(k): v for k, v in json.loads(label_to_name_json).items()}
print(bcolors.OKGREEN + "✅ Loaded label mapping:" + bcolors.ENDC)
for index, label in label_to_name.items():
    print(f"  - Index {index}: {label}")

# Train-Test Split
train_df, test_df = df_reduced.randomSplit([0.8, 0.2], seed=42)

# Huấn luyện Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=200, maxDepth=15, seed=42)
rf_model = rf.fit(train_df)

# Lưu mô hình
model_path = "s3a://mybucket/models/random_forest_model"
rf_model.write().overwrite().save(model_path)
print(bcolors.OKGREEN + f"✅ Model saved to {model_path}" + bcolors.ENDC)

# Dự đoán trên tập test
predictions = rf_model.transform(test_df)

# Đánh giá mô hình
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)

y_validate = predictions.select("label").toPandas()
y_predicted = predictions.select("prediction").toPandas()
actual_labels = sorted(y_validate["label"].unique())
precision, recall, fscore, support = precision_recall_fscore_support(y_validate, y_predicted, labels=actual_labels, zero_division=0)
original_labels = [label_to_name[label] for label in actual_labels]

df_results = pd.DataFrame({
    'attack': actual_labels,
    'original_label': original_labels,
    'precision': precision,
    'recall': recall,
    'fscore': fscore
})

precision_macro, recall_macro, fscore_macro, _ = precision_recall_fscore_support(y_validate, y_predicted, labels=actual_labels, average='macro', zero_division=0)
accuracy = accuracy_score(y_validate, y_predicted)

print(f"\n✅ F1-score: {f1_score:.4f}")
print(df_results.to_string(index=False))
print(f"\n✅ Precision (macro): {precision_macro:.4f}")
print(f"✅ Recall (macro): {recall_macro:.4f}")
print(f"✅ F1-score (macro): {fscore_macro:.4f}")
print(f"✅ Accuracy: {accuracy:.4f}")

spark.stop()