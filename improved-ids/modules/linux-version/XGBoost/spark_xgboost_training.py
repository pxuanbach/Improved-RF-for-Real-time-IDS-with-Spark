import sys
import os
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pyspark.sql import SparkSession
from xgboost.spark import SparkXGBClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import udf, lit
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from model_utils import evaluate_model, bcolors
from data_preprocessing import preprocess_data, create_label_index, reduce_dimensions, handle_nan_infinity, create_feature_vector, load_metadata
def create_spark_session(app_name, master="spark://spark-master:7077"):
    # Đường dẫn đến JAR trong container (cần được cài đặt trong Dockerfile hoặc ánh xạ qua volume)
    jars_dir = "/opt/bitnami/spark/jars"  # Đường dẫn mặc định của Spark trong container
    jars_list = [
        "hadoop-aws-3.3.6.jar",
        "aws-java-sdk-bundle-1.11.1026.jar",
        "guava-30.1-jre.jar",
        "hadoop-common-3.3.6.jar",
        "hadoop-client-3.3.6.jar",
        "xgboost4j-spark_2.12-2.0.3.jar",
        "xgboost4j_2.12-2.0.3.jar"
    ]
    jars = ",".join([os.path.join(jars_dir, jar) for jar in jars_list])

    spark = SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .config("spark.jars", jars) \
        .config("spark.driver.memory", "6g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "2") \
        .config("spark.executor.instances", "3") \
        .config("spark.network.timeout", "3600s") \
        .config("spark.driver.maxResultSize", "4g") \
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
        .config("spark.sql.adaptive.enabled", "false") \
        .config("spark.dynamicAllocation.enabled", "false") \
        .config("spark.sql.broadcastTimeout", "1200") \
        .config("spark.sql.autoBroadcastJoinThreshold", 10 * 1024 * 1024) \
        .config("spark.executor.heartbeatInterval", "120s") \
        .config("spark.storage.blockManagerSlaveTimeoutMs", "1800000") \
        .config("spark.log.level", "WARN") \
        .config("spark.rpc.retry.wait", "60s") \
        .config("spark.rpc.numRetries", "10") \
        .getOrCreate()
    
    # Đặt mức log cho SparkContext
    spark.sparkContext.setLogLevel("WARN")
    return spark
# Handle command-line arguments
parser = argparse.ArgumentParser(description="XGBoost Training with Spark")
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Directory to save results (confusion matrix, evaluation metrics, execution times). If not specified, defaults to 'results' directory in script location."
)
args = parser.parse_args()

# Determine output directory
if args.output_dir:
    output_dir = args.output_dir
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "results")

os.makedirs(output_dir, exist_ok=True)
print(bcolors.OKGREEN + f"✅ Results will be saved to {output_dir}" + bcolors.ENDC)

# Dictionary to store execution times
execution_times = {}

# Record total execution start time
total_start_time = time.time()

# Initialize Spark session
start_time = time.time()
spark = create_spark_session("pyspark-xgboost-training")
end_time = time.time()
execution_times["spark_session_creation"] = end_time - start_time
print(bcolors.OKCYAN + f"⏳ Spark session creation took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Load reduced data
start_time = time.time()
reduced_data_path = "s3a://mybucket/preprocessed_data/reduced_data.parquet"
df_reduced = spark.read.parquet(reduced_data_path)
print(bcolors.OKGREEN + f"✅ Loaded reduced data from {reduced_data_path}" + bcolors.ENDC)
end_time = time.time()
execution_times["load_reduced_data"] = end_time - start_time
print(bcolors.OKCYAN + f"⏳ Loading reduced data took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Load metadata
start_time = time.time()
global_top_features, label_to_name = load_metadata(spark)
print(bcolors.OKGREEN + f"✅ Loaded {len(global_top_features)} global top features: {global_top_features}" + bcolors.ENDC)
print(bcolors.OKGREEN + "✅ Loaded label mapping:" + bcolors.ENDC)
for index, label in label_to_name.items():
    print(f"  - Index {index}: {label}")
end_time = time.time()
execution_times["load_metadata"] = end_time - start_time
print(bcolors.OKCYAN + f"⏳ Loading metadata took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Chọn lọc đặc trưng bổ sung bằng RFSelector
start_time = time.time()
df_resampled = df_reduced
print(bcolors.OKCYAN + "⏳ Selecting top 18 features using RFSelector..." + bcolors.ENDC)
rf_selector = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=100,
    maxDepth=20,
    seed=42
)
rf_selector_model = rf_selector.fit(df_resampled)
feature_importances = rf_selector_model.featureImportances
importance_with_index = [(importance, index) for index, importance in enumerate(feature_importances)]
importance_with_index.sort(reverse=True)
selected_indices = [index for _, index in importance_with_index[:18]]
selected_features = [global_top_features[i] for i in selected_indices]
print(bcolors.OKGREEN + f"✅ Selected 18 features: {selected_features}" + bcolors.ENDC)

# Lưu danh sách 18 đặc trưng được chọn vào S3
selected_features_path = "s3a://mybucket/models/selected_features_18.parquet"
spark.createDataFrame([(selected_features,)], ["features"]).write.mode("overwrite").parquet(selected_features_path)
print(bcolors.OKGREEN + f"✅ Saved selected features to {selected_features_path}" + bcolors.ENDC)

# Cập nhật Spark DataFrame để chỉ chứa 18 đặc trưng được chọn
def extract_feature(vector, index):
    return float(vector[index])

extract_feature_udf = udf(extract_feature, DoubleType())

for i, feature_name in enumerate(global_top_features):
    df_resampled = df_resampled.withColumn(
        feature_name,
        extract_feature_udf("features", lit(i))
    )

columns_to_keep = selected_features + ["label"]
df_resampled = df_resampled.select(columns_to_keep)

assembler = VectorAssembler(inputCols=selected_features, outputCol="features")
df_resampled = assembler.transform(df_resampled).select("features", "label")
print(bcolors.OKGREEN + "✅ Updated DataFrame with 18 selected features" + bcolors.ENDC)
end_time = time.time()
execution_times["rf_selector"] = end_time - start_time
print(bcolors.OKCYAN + f"⏳ RFSelector processing took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Train-Test Split
start_time = time.time()
train_df, test_df = df_resampled.randomSplit([0.8, 0.2], seed=42)
end_time = time.time()
execution_times["train_test_split"] = end_time - start_time
print(bcolors.OKCYAN + f"⏳ Train-test split took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Train XGBoost
xgboost_params = {
    "features_col": "features",
    "label_col": "label",
    "num_round": 200,
    "max_depth": 10,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "num_class": len(label_to_name),
    "seed": 42,
    "num_workers": 3,  # Đặt thành 1 để chạy single node, tránh lỗi giao tiếp
}
xgboost = SparkXGBClassifier(**xgboost_params)
print(bcolors.OKCYAN + "⏳ Training XGBoost model..." + bcolors.ENDC)
xgboost_model = xgboost.fit(train_df)
end_time = time.time()
execution_times["training"] = end_time - start_time
print(bcolors.OKGREEN + f"✅ Training completed in {execution_times['training']:.2f} seconds" + bcolors.ENDC)

# Save model
start_time = time.time()
model_path = "s3a://mybucket/models/xgboost_model"
xgboost_model.write().overwrite().save(model_path)
print(bcolors.OKGREEN + f"✅ Model saved to {model_path}" + bcolors.ENDC)
end_time = time.time()
execution_times["save_model"] = end_time - start_time
print(bcolors.OKCYAN + f"⏳ Saving model took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Predict on test set
start_time = time.time()
predictions = xgboost_model.transform(test_df)
end_time = time.time()
execution_times["prediction"] = end_time - start_time
print(bcolors.OKCYAN + f"⏳ Prediction on test set took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Evaluate model
start_time = time.time()
f1_score, df_results, precision_macro, recall_macro, fscore_macro, accuracy = evaluate_model(predictions, label_to_name=label_to_name)
print(f"\n✅ F1-score: {f1_score:.4f}")
print(df_results.to_string(index=False))
print(f"\n✅ Precision (macro): {precision_macro:.4f}")
print(f"✅ Recall (macro): {recall_macro:.4f}")
print(f"✅ F1-score (macro): {fscore_macro:.4f}")
print(f"✅ Accuracy: {accuracy:.4f}")
end_time = time.time()
execution_times["evaluation"] = end_time - start_time
print(bcolors.OKCYAN + f"⏳ Model evaluation took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Compute and plot confusion matrix (in percentage)
start_time = time.time()
print(bcolors.OKCYAN + "⏳ Computing and plotting confusion matrix (in percentage)..." + bcolors.ENDC)

# Collect true and predicted labels
y_true = predictions.select("label").rdd.flatMap(lambda x: x).collect()
y_pred = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Normalize confusion matrix to percentages (by row)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0 if any row has no samples

# Get labels sorted by index
labels = [label_to_name[i] for i in sorted(label_to_name.keys())]

# Plot confusion matrix
plt.figure(figsize=(12, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix (%)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

# Save confusion matrix as image
cm_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print(bcolors.OKGREEN + f"✅ Confusion matrix saved to {cm_path}" + bcolors.ENDC)
end_time = time.time()
execution_times["confusion_matrix"] = end_time - start_time
print(bcolors.OKCYAN + f"⏳ Confusion matrix computation and plotting took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Save evaluation metrics to CSV
start_time = time.time()
print(bcolors.OKCYAN + "⏳ Saving evaluation metrics to CSV..." + bcolors.ENDC)

# Save general metrics
eval_metrics = {
    "F1-score (micro)": f1_score,
    "Precision (macro)": precision_macro,
    "Recall (macro)": recall_macro,
    "F1-score (macro)": fscore_macro,
    "Accuracy": accuracy
}
eval_df = pd.DataFrame([eval_metrics])
eval_path = os.path.join(output_dir, "evaluation_metrics.csv")
eval_df.to_csv(eval_path, index=False)
print(bcolors.OKGREEN + f"✅ General evaluation metrics saved to {eval_path}" + bcolors.ENDC)

# Save per-class metrics
per_class_df = df_results
per_class_path = os.path.join(output_dir, "per_class_metrics.csv")
per_class_df.to_csv(per_class_path, index=False)
print(bcolors.OKGREEN + f"✅ Per-class metrics saved to {per_class_path}" + bcolors.ENDC)
end_time = time.time()
execution_times["save_evaluation_metrics"] = end_time - start_time
print(bcolors.OKCYAN + f"⏳ Saving evaluation metrics took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Save execution times to CSV
start_time = time.time()
print(bcolors.OKCYAN + "⏳ Saving execution times to CSV..." + bcolors.ENDC)
times_df = pd.DataFrame([execution_times])
times_path = os.path.join(output_dir, "execution_times.csv")
times_df.to_csv(times_path, index=False)
print(bcolors.OKGREEN + f"✅ Execution times saved to {times_path}" + bcolors.ENDC)
end_time = time.time()
execution_times["save_execution_times"] = end_time - start_time
print(bcolors.OKCYAN + f"⏳ Saving execution times took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Total execution time
total_end_time = time.time()
execution_times["total_execution"] = total_end_time - total_start_time
print(bcolors.WARNING + f"⏳ Total execution time: {execution_times['total_execution']:.2f} seconds" + bcolors.ENDC)

# Stop Spark session
spark.stop()