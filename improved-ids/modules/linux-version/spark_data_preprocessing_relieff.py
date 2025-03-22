import time
import os
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col, when, lit, isnan, isnull
from pyspark.ml.feature import VectorAssembler, StringIndexer
from relieffselector import ReliefFSelector
import json

print("Running on:", os.name)
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
    .appName("pyspark-notebook")\
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
    .config("spark.memory.offHeap.size", "4g") \
    .config("spark.shuffle.file.buffer", "2048k") \
    .config("spark.default.parallelism", "8") \
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

# Load dữ liệu từ 1 file (đã comment các file khác để test)
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
    .csv(volume_files, header=True, inferSchema=True) \
    .repartition(18)

df = df.withColumnRenamed(' Label', 'Label')

df = df.replace(['Heartbleed', 'Web Attack � Sql Injection', 'Infiltration'], None, subset=['Label'])
df = df.dropna(how='any')

# Replace 'Web Attack � Brute Force' with 'Brute Force'
df = df.withColumn('Label', when(col('Label') == 'Web Attack � Brute Force', 'Brute Force').otherwise(col('Label')))

# Replace 'Web Attack � XSS' with 'XSS'
df = df.withColumn('Label', when(col('Label') == 'Web Attack � XSS', 'XSS').otherwise(col('Label')))

df = df.withColumn('Attack', when(col('Label') == 'BENIGN', 0).otherwise(1))

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

# Create 'Label_Category' column by mapping 'Label' to attack_group
# Use when/otherwise to simulate mapping
conditions = [when(col('Label') == k, lit(v)) for k, v in attack_group.items()]
df = df.withColumn('Label_Category', conditions[0])  # Start with first condition
for condition in conditions[1:]:  # Chain the rest
    df = df.withColumn('Label_Category', when(col('Label_Category').isNull(), condition).otherwise(col('Label_Category')))

# Get numeric columns only and exclude label columns
exclude_cols = ['Label', 'Label_Category', 'Attack']
feature_cols = [col_name for col_name in df.columns 
               if col_name not in exclude_cols
               and df.schema[col_name].dataType.typeName() in ('double', 'integer', 'float')]

print(bcolors.OKBLUE + f"Selected {len(feature_cols)} numeric features" + bcolors.ENDC)


# Create label index
from pyspark.ml.feature import StringIndexer
label_indexer = StringIndexer(inputCol="Label_Category", outputCol="label")
label_indexer_model = label_indexer.fit(df)
df = label_indexer.fit(df).transform(df)
# In ánh xạ
labels = label_indexer_model.labels
print(bcolors.OKBLUE + "Mapping of Label_Category to label index:" + bcolors.ENDC)
for index, label in enumerate(labels):
    print(f"  - Index {index}: {label}")

# Lưu ánh xạ vào dictionary để dùng sau
label_to_name = {index: label for index, label in enumerate(labels)}

# Create feature vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_vector = assembler.transform(df).select("features", "label")

# print(df_vector.count())

def calculate_number_of_subsets(spark_df, max_records=15000):
    total_records = spark_df.count()
    
    num_subsets = (total_records + max_records - 1) // max_records
    
    return num_subsets


# Split the DataFrame into parts (e.g., 3 splits)
num_splits = calculate_number_of_subsets(df_vector, max_records=15000) # base 15000
split_weights = [1.0 / num_splits] * num_splits
df_splits = df_vector.randomSplit(split_weights, seed=42)


# Now run ReliefFSelector
selector = (
    ReliefFSelector()
    .setSelectionThreshold(0.3) # Select top 30% features (~ 23 features)
    .setNumNeighbors(10)
    .setSampleSize(8)
)

# Process each split and collect top features
top_features_per_split = []
total_time = 0.0
for i, df_split in enumerate(df_splits):
    start_time = time.time()
    total_instances = df_split.count()
    print(bcolors.HEADER + f"Processing split {i + 1}/{num_splits} with {total_instances} instances" + bcolors.ENDC)
    
    # Fit the model on the split
    model = selector.fit(df_split)
    
    # Extract weights from the model
    weights = model.weights  # Assuming ReliefFSelectorModel exposes weights
    n_select = int(len(weights) * selector.getSelectionThreshold())
    selected_indices = np.argsort(weights)[-n_select:]  # Top feature indices
    
    # Map indices to feature names
    selected_features = [feature_cols[idx] for idx in selected_indices]
    top_features_per_split.append((i, selected_features, weights[selected_indices]))
    
    # Transform the split
    # selected_df = model.transform(df_split)
    print(bcolors.OKGREEN + f"Split {i + 1}: Top {len(selected_features)} features: {selected_features}" + bcolors.ENDC)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(bcolors.OKCYAN + f"Elapsed time for split {i + 1}: {elapsed_time:.4f} seconds" + bcolors.ENDC)
    
    df_split.unpersist(blocking=True)  # Clear cache immediately
    spark.sparkContext._jvm.System.gc()  # Suggest garbage collection

    print(bcolors.OKCYAN + f"Released resources for split {i + 1}" + bcolors.ENDC)
    total_time += elapsed_time

print(bcolors.WARNING + f"Total elapsed time for {num_splits} splits: {total_time:.4f} seconds" + bcolors.ENDC)

# Combine top features (example: union of all selected features)
all_top_features = set()
for split_id, features, _ in top_features_per_split:
    all_top_features.update(features)
all_top_features_list = list(all_top_features)
print(bcolors.OKGREEN + f"Combined top features across all splits ({len(all_top_features_list)} features): {all_top_features_list}" + bcolors.ENDC)

# Print top features with averaged weights
global_weights = np.zeros(len(feature_cols))
for split_id, features, split_weights in top_features_per_split:
    for feature, weight in zip(features, split_weights):
        idx = feature_cols.index(feature)
        global_weights[idx] += weight
global_weights /= num_splits  # avg weights

n_global_select = int(len(feature_cols) * selector.getSelectionThreshold())
global_top_indices = np.argsort(global_weights)[-n_global_select:]
global_top_features = [feature_cols[idx] for idx in global_top_indices]
print(f"Selected {len(global_top_features)} features after ReliefF: {global_top_features}")
global_top_weights = global_weights[global_top_indices]
print(bcolors.OKGREEN + "Global top features with averaged weights:" + bcolors.ENDC)
for feature, weight in zip(global_top_features, global_top_weights):
    print(f"  - {feature}: {weight:.6f}")


# reduce phase
print(bcolors.HEADER + "Reducing dimensionality by selecting top feature columns..." + bcolors.ENDC)
columns_to_keep = global_top_features + exclude_cols  # keep Label, Label_Category, Attack columns
df_reduced = df.select(columns_to_keep)

df_reduced.show(10)
print(f"Shape of df_reduced: {len(df_reduced.columns)} columns")


from pyspark.sql.functions import isnan, isnull, when

# 🚀 Kiểm tra và xử lý NaN/Infinity
print(bcolors.HEADER + "Checking for NaN or Infinity before training..." + bcolors.ENDC)

for col_name in global_top_features:
    count_nan = df_reduced.filter(isnan(col(col_name)) | isnull(col(col_name))).count()
    count_inf = df_reduced.filter(col(col_name) == float("inf")).count()
    
    if count_nan > 0 or count_inf > 0:
        print(f"{bcolors.WARNING}⚠️ Cột {col_name} có {count_nan} NaN và {count_inf} Infinity!{bcolors.ENDC}")

    # Thay thế Infinity bằng NULL
    df_reduced = df_reduced.withColumn(
        col_name, when(col(col_name) == float("inf"), None).otherwise(col(col_name))
    )

    # Tính giá trị trung bình của cột
    mean_value = df_reduced.selectExpr(f"avg(`{col_name}`) as mean_val").collect()[0]["mean_val"]

    # Nếu mean_value là None (do toàn bộ cột có NaN), thay bằng 0
    if mean_value is None:
        mean_value = 0.0

    # Thay thế NaN bằng giá trị trung bình
    df_reduced = df_reduced.fillna({col_name: mean_value})

# 🚀 Chuyển đổi lại thành vector đặc trưng
assembler_selected = VectorAssembler(inputCols=global_top_features, outputCol="features")
df_reduced = assembler_selected.transform(df_reduced).select("features", "label")

# Lưu dữ liệu đã giảm chiều
reduced_data_path = "s3a://mybucket/preprocessed_data/reduced_data.parquet"
df_reduced.write.mode("overwrite").parquet(reduced_data_path)
print(bcolors.OKGREEN + f"✅ Reduced data saved to {reduced_data_path}" + bcolors.ENDC)

# Lưu global_top_features và label_to_name
spark.createDataFrame([(json.dumps(global_top_features),)], ["features"])\
    .write.mode("overwrite").text("s3a://mybucket/models/global_top_features")
spark.createDataFrame([(json.dumps(label_to_name),)], ["labels"])\
    .write.mode("overwrite").text("s3a://mybucket/models/label_to_name")
print(bcolors.OKGREEN + "✅ Metadata (features and labels) saved to S3" + bcolors.ENDC)

spark.stop()