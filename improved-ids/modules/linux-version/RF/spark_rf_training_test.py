import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Thêm cấp thư mục cha

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import udf, lit
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from spark_utils import create_spark_session
from model_utils import evaluate_model, bcolors
from data_preprocessing import preprocess_data, create_label_index, reduce_dimensions, handle_nan_infinity, create_feature_vector, load_metadata
from imblearn.over_sampling import SMOTE

# Xử lý tham số dòng lệnh
parser = argparse.ArgumentParser(description="Random Forest Training with Spark")
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Directory to save results (confusion matrix, evaluation metrics, execution times). If not specified, defaults to 'results' directory in script location."
)
args = parser.parse_args()

# Xác định thư mục lưu kết quả
if args.output_dir:
    output_dir = args.output_dir
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "results")

os.makedirs(output_dir, exist_ok=True)
print(bcolors.OKGREEN + f"✅ Results will be saved to {output_dir}" + bcolors.ENDC)

# Dictionary để lưu thời gian của từng bước
execution_times = {}

# Lưu thời gian bắt đầu toàn bộ quy trình
total_start_time = time.time()

# Khởi tạo Spark session
start_time = time.time()
spark = create_spark_session("pyspark-rf-training")
end_time = time.time()
execution_times["spark_session_creation"] = end_time - start_time
print(bcolors.OKCYAN + f"⏳ Spark session creation took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Tạo mock data thay vì load từ S3
start_time = time.time()
print(bcolors.OKCYAN + "⏳ Creating mock data..." + bcolors.ENDC)

# Tạo dữ liệu giả: 100 mẫu, 10 đặc trưng, 3 lớp
np.random.seed(42)
num_samples = 100
num_features = 10
num_classes = 3

# Tạo đặc trưng ngẫu nhiên
features = np.random.randn(num_samples, num_features)
labels = np.random.randint(0, num_classes, num_samples)

# Chuyển thành Spark DataFrame
mock_data = [(Vectors.dense(features[i]), float(labels[i])) for i in range(num_samples)]
df_reduced = spark.createDataFrame(mock_data, ["features", "label"])
print(bcolors.OKGREEN + f"✅ Created mock data with {num_samples} samples, {num_features} features, {num_classes} classes" + bcolors.ENDC)
end_time = time.time()
execution_times["create_mock_data"] = end_time - start_time
print(bcolors.OKCYAN + f"⏳ Creating mock data took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Tạo metadata giả
start_time = time.time()
global_top_features = [f"feature_{i}" for i in range(num_features)]
label_to_name = {0: "normal", 1: "attack", 2: "botnet"}
print(bcolors.OKGREEN + f"✅ Loaded {len(global_top_features)} global top features: {global_top_features}" + bcolors.ENDC)
print(bcolors.OKGREEN + "✅ Loaded label mapping:" + bcolors.ENDC)
for index, label in label_to_name.items():
    print(f"  - Index {index}: {label}")
end_time = time.time()
execution_times["load_metadata"] = end_time - start_time
print(bcolors.OKCYAN + f"⏳ Loading metadata took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Chọn lọc đặc trưng bổ sung bằng RFSelector (chọn 5 đặc trưng thay vì 18 để nhanh hơn)
start_time = time.time()
df_resampled = df_reduced
print(bcolors.OKCYAN + "⏳ Selecting top 5 features using RFSelector..." + bcolors.ENDC)
rf_selector = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=10,
    maxDepth=5,
    seed=42
)
rf_selector_model = rf_selector.fit(df_resampled)
feature_importances = rf_selector_model.featureImportances
importance_with_index = [(importance, index) for index, importance in enumerate(feature_importances)]
importance_with_index.sort(reverse=True)
selected_indices = [index for _, index in importance_with_index[:5]]
selected_features = [global_top_features[i] for i in selected_indices]
print(bcolors.OKGREEN + f"✅ Selected 5 features: {selected_features}" + bcolors.ENDC)

# Lưu danh sách đặc trưng được chọn (giả lập lưu vào S3)
selected_features_path = "s3a://mybucket/models/selected_features_5.parquet"
print(bcolors.OKGREEN + f"✅ (Mock) Saved selected features to {selected_features_path}" + bcolors.ENDC)

# Cập nhật Spark DataFrame để chỉ chứa 5 đặc trưng được chọn
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
print(bcolors.OKGREEN + "✅ Updated DataFrame with 5 selected features" + bcolors.ENDC)
end_time = time.time()
execution_times["rf_selector"] = end_time - start_time
print(bcolors.OKCYAN + f"⏳ RFSelector processing took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Train-Test Split
start_time = time.time()
train_df, test_df = df_resampled.randomSplit([0.8, 0.2], seed=42)
end_time = time.time()
execution_times["train_test_split"] = end_time - start_time
print(bcolors.OKCYAN + f"⏳ Train-test split took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Huấn luyện Random Forest
start_time = time.time()
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=5,
    maxDepth=3,
    minInstancesPerNode=2,
    featureSubsetStrategy="sqrt",
    impurity="gini",
    seed=42
)
print(bcolors.OKCYAN + "⏳ Training Random Forest model..." + bcolors.ENDC)
rf_model = rf.fit(train_df)
end_time = time.time()
execution_times["training"] = end_time - start_time
print(bcolors.OKGREEN + f"✅ Training completed in {execution_times['training']:.2f} seconds" + bcolors.ENDC)

# Lưu mô hình (giả lập lưu vào S3)
start_time = time.time()
model_path = "s3a://mybucket/models/random_forest_model"
print(bcolors.OKGREEN + f"✅ (Mock) Model saved to {model_path}" + bcolors.ENDC)
end_time = time.time()
execution_times["save_model"] = end_time - start_time
print(bcolors.OKCYAN + f"⏳ Saving model took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Dự đoán trên tập test
start_time = time.time()
predictions = rf_model.transform(test_df)
end_time = time.time()
execution_times["prediction"] = end_time - start_time
print(bcolors.OKCYAN + f"⏳ Prediction on test set took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Đánh giá mô hình
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

# Tính và vẽ confusion matrix
start_time = time.time()
print(bcolors.OKCYAN + "⏳ Computing and plotting confusion matrix..." + bcolors.ENDC)

y_true = predictions.select("label").rdd.flatMap(lambda x: x).collect()
y_pred = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()

cm = confusion_matrix(y_true, y_pred)
labels = [label_to_name[i] for i in range(len(label_to_name))]

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

cm_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(bcolors.OKGREEN + f"✅ Confusion matrix saved to {cm_path}" + bcolors.ENDC)
end_time = time.time()
execution_times["confusion_matrix"] = end_time - start_time
print(bcolors.OKCYAN + f"⏳ Confusion matrix computation and plotting took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Lưu các giá trị đánh giá vào CSV
start_time = time.time()
print(bcolors.OKCYAN + "⏳ Saving evaluation metrics to CSV..." + bcolors.ENDC)

# Lưu các chỉ số tổng quát
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

# Lưu bảng chi tiết cho từng lớp
per_class_df = df_results
per_class_path = os.path.join(output_dir, "per_class_metrics.csv")
per_class_df.to_csv(per_class_path, index=False)
print(bcolors.OKGREEN + f"✅ Per-class metrics saved to {per_class_path}" + bcolors.ENDC)

end_time = time.time()
execution_times["save_evaluation_metrics"] = end_time - start_time
print(bcolors.OKCYAN + f"⏳ Saving evaluation metrics took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Lưu thời gian chạy vào CSV
start_time = time.time()
print(bcolors.OKCYAN + "⏳ Saving execution times to CSV..." + bcolors.ENDC)
times_df = pd.DataFrame([execution_times])
times_path = os.path.join(output_dir, "execution_times.csv")
times_df.to_csv(times_path, index=False)
print(bcolors.OKGREEN + f"✅ Execution times saved to {times_path}" + bcolors.ENDC)
end_time = time.time()
execution_times["save_execution_times"] = end_time - start_time
print(bcolors.OKCYAN + f"⏳ Saving execution times took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Tổng thời gian thực hiện
total_end_time = time.time()
execution_times["total_execution"] = total_end_time - total_start_time
print(bcolors.WARNING + f"⏳ Total execution time: {execution_times['total_execution']:.2f} seconds" + bcolors.ENDC)

spark.stop()