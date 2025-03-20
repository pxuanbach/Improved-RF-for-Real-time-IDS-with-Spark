from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.functions import col, when, lit, udf, isnan, isnull
from pyspark.sql.types import StringType
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
    .appName("pyspark-rf-detection")\
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

# Load dữ liệu từ file CSV (chỉ lấy 1 file để test)
volume_files = [
    "s3a://mybucket/cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    # "s3a://mybucket/cicids2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    # "s3a://mybucket/cicids2017/Friday-WorkingHours-Morning.pcap_ISCX.csv",
    # "s3a://mybucket/cicids2017/Monday-WorkingHours.pcap_ISCX.csv",
    # "s3a://mybucket/cicids2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    # "s3a://mybucket/cicids2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    # "s3a://mybucket/cicids2017/Tuesday-WorkingHours.pcap_ISCX.csv",
    # "s3a://mybucket/cicids2017/Wednesday-workingHours.pcap_ISCX.csv",
]

df = spark.read \
    .option("nullValue", "NA") \
    .option("emptyValue", "unknown") \
    .csv(volume_files, header=True, inferSchema=True)

# Lấy ngẫu nhiên 50 bản ghi để test
sample_size = 50  # Lấy 50 bản ghi ngẫu nhiên
total_records = df.count()
if total_records > sample_size:
    df_sample = df.sample(False, sample_size / total_records, seed=42)
else:
    df_sample = df
print(bcolors.OKGREEN + f"✅ Sampled {df_sample.count()} records for detection" + bcolors.ENDC)

# Tiền xử lý dữ liệu
df_sample = df_sample.withColumnRenamed(' Label', 'Label')
df_sample = df_sample.replace(['Heartbleed', 'Web Attack � Sql Injection', 'Infiltration'], None, subset=['Label'])
df_sample = df_sample.dropna(how='any')
df_sample = df_sample.withColumn('Label', when(col('Label') == 'Web Attack � Brute Force', 'Brute Force').otherwise(col('Label')))
df_sample = df_sample.withColumn('Label', when(col('Label') == 'Web Attack � XSS', 'XSS').otherwise(col('Label')))
df_sample = df_sample.withColumn('Attack', when(col('Label') == 'BENIGN', 0).otherwise(1))

attack_group = {
    'BENIGN': 'benign', 
    'DoS Hulk': 'dos',
    'PortScan': 'probe', 
    'DDoS': 'ddos',
    'DoS GoldenEye': 'dos', 
    'FTP-Patator': 'brute_force',
    'SSH-Patator': 'brute_force', 
    'DoS slowloris': 'dos', 
    'DoS Slowhttptest': 'dos',
    'Bot': 'botnet',
    'Brute Force': 'web_attack', 
    'XSS': 'web_attack'
}

conditions = [when(col('Label') == k, lit(v)) for k, v in attack_group.items()]
df_sample = df_sample.withColumn('Label_Category', conditions[0])
for condition in conditions[1:]:
    df_sample = df_sample.withColumn('Label_Category', when(col('Label_Category').isNull(), condition).otherwise(col('Label_Category')))

# Load global_top_features và label_to_name
global_top_features_json = spark.read.text("s3a://mybucket/models/global_top_features").collect()[0]["value"]
global_top_features = json.loads(global_top_features_json)
print(bcolors.OKGREEN + f"✅ Loaded {len(global_top_features)} global top features: {global_top_features}" + bcolors.ENDC)

label_to_name_json = spark.read.text("s3a://mybucket/models/label_to_name").collect()[0]["value"]
label_to_name = {int(k): v for k, v in json.loads(label_to_name_json).items()}
print(bcolors.OKGREEN + "✅ Loaded label mapping:" + bcolors.ENDC)
for index, label in label_to_name.items():
    print(f"  - Index {index}: {label}")

# Tạo index cho nhãn
label_indexer = StringIndexer(inputCol="Label_Category", outputCol="label")
label_indexer_model = label_indexer.fit(df_sample)
df_sample = label_indexer_model.transform(df_sample)

# Giảm chiều dữ liệu với global_top_features
exclude_cols = ['Label', 'Label_Category', 'Attack']
columns_to_keep = global_top_features + exclude_cols
df_sample = df_sample.select(columns_to_keep)

# Kiểm tra và xử lý NaN/Infinity
print(bcolors.HEADER + "Checking for NaN or Infinity in sample data..." + bcolors.ENDC)
for col_name in global_top_features:
    count_nan = df_sample.filter(isnan(col(col_name)) | isnull(col(col_name))).count()
    count_inf = df_sample.filter(col(col_name) == float("inf")).count()
    
    if count_nan > 0 or count_inf > 0:
        print(f"{bcolors.WARNING}⚠️ Cột {col_name} có {count_nan} NaN và {count_inf} Infinity!{bcolors.ENDC}")
    
    df_sample = df_sample.withColumn(col_name, when(col(col_name) == float("inf"), None).otherwise(col(col_name)))
    median_value = df_sample.approxQuantile(col_name, [0.5], 0.25)[0]
    if median_value is None:
        median_value = 0.0
    df_sample = df_sample.fillna({col_name: median_value})

# Tạo vector đặc trưng
assembler = VectorAssembler(inputCols=global_top_features, outputCol="features")
df_sample_prepared = assembler.transform(df_sample).select("features", "label", "Label", "Label_Category")

# Load mô hình đã lưu
model_path = "s3a://mybucket/models/random_forest_model"
loaded_model = RandomForestClassificationModel.load(model_path)
print(bcolors.OKGREEN + "✅ Model loaded successfully" + bcolors.ENDC)

# Dự đoán trên dữ liệu mẫu
predictions = loaded_model.transform(df_sample_prepared)

# Ánh xạ nhãn dự đoán về nhãn gốc
def map_label_to_category(index):
    return label_to_name.get(int(index), "unknown")

map_label_to_category_udf = udf(map_label_to_category, StringType())
predictions_with_labels = predictions.withColumn("actual_label", map_label_to_category_udf(col("label")))
predictions_with_labels = predictions_with_labels.withColumn("predicted_label", map_label_to_category_udf(col("prediction")))

# Hiển thị kết quả dự đoán (toàn bộ sample)
print(bcolors.HEADER + f"Prediction Results ({df_sample.count()} rows):" + bcolors.ENDC)
predictions_with_labels.select("Label", "actual_label", "predicted_label").show(int(df_sample.count()))

# Đánh giá mô hình
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)

y_validate = predictions_with_labels.select("label").toPandas()
y_predicted = predictions_with_labels.select("prediction").toPandas()
actual_labels = sorted(y_validate["label"].unique())
print(f"Unique Labels in Sample Data: {actual_labels}")

precision, recall, fscore, support = precision_recall_fscore_support(y_validate, y_predicted, labels=actual_labels, zero_division=0)
original_labels = [label_to_name.get(int(label), "unknown") for label in actual_labels]

df_results = pd.DataFrame({
    'attack': actual_labels,
    'original_label': original_labels,
    'precision': precision,
    'recall': recall,
    'fscore': fscore
})

precision_macro, recall_macro, fscore_macro, _ = precision_recall_fscore_support(
    y_validate, y_predicted, labels=actual_labels, average='macro', zero_division=0
)
accuracy = accuracy_score(y_validate, y_predicted)

print(bcolors.HEADER + "\nDetailed Results:" + bcolors.ENDC)
print(df_results.to_string(index=False))
print(f"\n✅ F1-score: {f1_score:.4f}")
print(f"✅ Precision (macro): {precision_macro:.4f}")
print(f"✅ Recall (macro): {recall_macro:.4f}")
print(f"✅ F1-score (macro): {fscore_macro:.4f}")
print(f"✅ Accuracy: {accuracy:.4f}")

# Lưu kết quả dự đoán (tùy chọn)
# predictions_with_labels.write.mode("overwrite").parquet("s3a://mybucket/predictions/detection_results.parquet")
# print(bcolors.OKGREEN + "✅ Detection results saved to S3" + bcolors.ENDC)

spark.stop()