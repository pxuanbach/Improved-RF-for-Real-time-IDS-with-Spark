import time
import os
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from relieffselector import ReliefFSelector
import json
from spark_utils import create_spark_session
from data_preprocessing import preprocess_data, create_label_index, reduce_dimensions, handle_nan_infinity, create_feature_vector, load_metadata
from model_utils import evaluate_model,bcolors


print("Running on:", os.name)
print("Current working directory:", os.getcwd())

# Khởi tạo Spark session
spark = create_spark_session("pyspark-notebook")

# Load dữ liệu
volume_files = [
    "s3a://mybucket/cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "s3a://mybucket/cicids2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "s3a://mybucket/cicids2017/Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "s3a://mybucket/cicids2017/Monday-WorkingHours.pcap_ISCX.csv",
    "s3a://mybucket/cicids2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "s3a://mybucket/cicids2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "s3a://mybucket/cicids2017/Tuesday-WorkingHours.pcap_ISCX.csv",
    "s3a://mybucket/cicids2017/Wednesday-workingHours.pcap_ISCX.csv",
]
df = spark.read.option("nullValue", "NA").option("emptyValue", "unknown").csv(volume_files, header=True, inferSchema=True).repartition(16).cache()

# Tiền xử lý dữ liệu
df = preprocess_data(df)

# Get numeric columns
exclude_cols = ['Label', 'Label_Category', 'Attack']
feature_cols = [col_name for col_name in df.columns 
                if col_name not in exclude_cols and df.schema[col_name].dataType.typeName() in ('double', 'integer', 'float')]
print(bcolors.OKBLUE + f"Selected {len(feature_cols)} numeric features" + bcolors.ENDC)

# Tạo label index
df, label_to_name, labels = create_label_index(df)
print(bcolors.OKBLUE + "Mapping of Label_Category to label index:" + bcolors.ENDC)
for index, label in enumerate(labels):
    print(f"  - Index {index}: {label}")

# Tạo feature vector
df_vector = create_feature_vector(df, feature_cols).select("features", "label")

def calculate_number_of_subsets(spark_df, max_records=15000):
    total_records = spark_df.count()
    num_subsets = (total_records + max_records - 1) // max_records
    return num_subsets

# Split DataFrame
num_splits = calculate_number_of_subsets(df_vector)
split_weights = [1.0 / num_splits] * num_splits
df_splits = df_vector.randomSplit(split_weights, seed=42)

# Run ReliefFSelector
selector = ReliefFSelector().setSelectionThreshold(0.3).setNumNeighbors(10).setSampleSize(8)
top_features_per_split = []
total_time = 0.0
for i, df_split in enumerate(df_splits):
    start_time = time.time()
    total_instances = df_split.count()
    print(bcolors.HEADER + f"Processing split {i + 1}/{num_splits} with {total_instances} instances" + bcolors.ENDC)
    
    model = selector.fit(df_split)
    weights = model.weights
    n_select = int(len(weights) * selector.getSelectionThreshold())
    selected_indices = np.argsort(weights)[-n_select:]
    selected_features = [feature_cols[idx] for idx in selected_indices]
    top_features_per_split.append((i, selected_features, weights[selected_indices]))
    
    print(bcolors.OKGREEN + f"Split {i + 1}: Top {len(selected_features)} features: {selected_features}" + bcolors.ENDC)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(bcolors.OKCYAN + f"Elapsed time for split {i + 1}: {elapsed_time:.4f} seconds" + bcolors.ENDC)
    
    df_split.unpersist(blocking=True)
    spark.sparkContext._jvm.System.gc()
    print(bcolors.OKCYAN + f"Released resources for split {i + 1}" + bcolors.ENDC)
    total_time += elapsed_time

print(bcolors.WARNING + f"Total elapsed time for {num_splits} splits: {total_time:.4f} seconds" + bcolors.ENDC)

# Combine top features
all_top_features = set()
for split_id, features, _ in top_features_per_split:
    all_top_features.update(features)
all_top_features_list = list(all_top_features)
print(bcolors.OKGREEN + f"Combined top features across all splits ({len(all_top_features_list)} features): {all_top_features_list}" + bcolors.ENDC)

global_weights = np.zeros(len(feature_cols))
for split_id, features, split_weights in top_features_per_split:
    for feature, weight in zip(features, split_weights):
        idx = feature_cols.index(feature)
        global_weights[idx] += weight
global_weights /= num_splits
n_global_select = int(len(feature_cols) * selector.getSelectionThreshold())
global_top_indices = np.argsort(global_weights)[-n_global_select:]
global_top_features = [feature_cols[idx] for idx in global_top_indices]
print(f"Selected {len(global_top_features)} features after ReliefF: {global_top_features}")
global_top_weights = global_weights[global_top_indices]
print(bcolors.OKGREEN + "Global top features with averaged weights:" + bcolors.ENDC)
for feature, weight in zip(global_top_features, global_top_weights):
    print(f"  - {feature}: {weight:.6f}")

# Giảm chiều dữ liệu
print(bcolors.HEADER + "Reducing dimensionality by selecting top feature columns..." + bcolors.ENDC)
df_reduced = reduce_dimensions(df, global_top_features)

df_reduced.show(10)
print(f"Shape of df_reduced: {len(df_reduced.columns)} columns")

# Xử lý NaN/Infinity
print(bcolors.HEADER + "Checking for NaN or Infinity before training..." + bcolors.ENDC)
df_reduced = handle_nan_infinity(df_reduced, global_top_features)

# Tạo vector đặc trưng cho dữ liệu đã giảm chiều
df_reduced = create_feature_vector(df_reduced, global_top_features).select("features", "label")

# Save reduced data
reduced_data_path = "s3a://mybucket/preprocessed_data/reduced_data.parquet"
df_reduced.write.mode("overwrite").parquet(reduced_data_path)
print(bcolors.OKGREEN + f"✅ Reduced data saved to {reduced_data_path}" + bcolors.ENDC)

# Save metadata
spark.createDataFrame([(json.dumps(global_top_features),)], ["features"]).write.mode("overwrite").text("s3a://mybucket/models/global_top_features")
spark.createDataFrame([(json.dumps(label_to_name),)], ["labels"]).write.mode("overwrite").text("s3a://mybucket/models/label_to_name")
print(bcolors.OKGREEN + "✅ Metadata (features and labels) saved to S3" + bcolors.ENDC)

spark.stop()