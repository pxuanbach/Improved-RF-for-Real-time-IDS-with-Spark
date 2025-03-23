import sys
import os
import time  # Thêm import time để đo thời gian
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Thêm cấp thư mục cha
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import XGBoostClassifier
from spark_utils import create_spark_session
from model_utils import evaluate_model, bcolors
from data_preprocessing import preprocess_data, create_label_index, reduce_dimensions, handle_nan_infinity, create_feature_vector, load_metadata

# Khởi tạo Spark session
spark = create_spark_session("pyspark-xgboost-training")

# Load dữ liệu đã giảm chiều
reduced_data_path = "s3a://mybucket/preprocessed_data/reduced_data.parquet"
df_reduced = spark.read.parquet(reduced_data_path)
print(bcolors.OKGREEN + f"✅ Loaded reduced data from {reduced_data_path}" + bcolors.ENDC)

# Load metadata
global_top_features, label_to_name = load_metadata(spark)
print(bcolors.OKGREEN + f"✅ Loaded {len(global_top_features)} global top features: {global_top_features}" + bcolors.ENDC)
print(bcolors.OKGREEN + "✅ Loaded label mapping:" + bcolors.ENDC)
for index, label in label_to_name.items():
    print(f"  - Index {index}: {label}")

# Train-Test Split
train_df, test_df = df_reduced.randomSplit([0.8, 0.2], seed=42)

# Huấn luyện XGBoost với các tham số đã đề xuất
xgboost_params = {
    "featuresCol": "features",
    "labelCol": "label",
    "num_round": 200,           # Số cây (tương ứng n_estimators=200)
    "max_depth": 10,            # Độ sâu tối đa của mỗi cây
    "eta": 0.05,                # Tốc độ học (tương ứng learning_rate=0.05)
    "subsample": 0.8,           # Tỷ lệ dữ liệu dùng cho mỗi cây
    "colsample_bytree": 0.8,    # Tỷ lệ cột dùng cho mỗi cây
    "objective": "multi:softprob",  # Phân loại đa lớp
    "eval_metric": "mlogloss",  # Đánh giá bằng multiclass logloss
    "num_class": len(label_to_name),  # Số lớp (dựa trên label_to_name)
    "seed": 42,                 # Đảm bảo tính tái lập
    "num_workers": 3            # Số worker (tùy thuộc vào cấu hình Spark cluster)
}

xgboost = XGBoostClassifier(**xgboost_params)
print(bcolors.OKCYAN + "⏳ Training XGBoost model..." + bcolors.ENDC)
start_time = time.time()  # Bắt đầu đo thời gian
xgboost_model = xgboost.fit(train_df)
end_time = time.time()  # Kết thúc đo thời gian
training_time = end_time - start_time
print(bcolors.OKGREEN + f"✅ Training completed in {training_time:.2f} seconds" + bcolors.ENDC)

# Lưu mô hình
model_path = "s3a://mybucket/models/xgboost_model"
xgboost_model.write().overwrite().save(model_path)
print(bcolors.OKGREEN + f"✅ Model saved to {model_path}" + bcolors.ENDC)

# Dự đoán trên tập test
predictions = xgboost_model.transform(test_df)

# Đánh giá mô hình
f1_score, df_results, precision_macro, recall_macro, fscore_macro, accuracy = evaluate_model(predictions, label_to_name=label_to_name)
print(f"\n✅ F1-score: {f1_score:.4f}")
print(df_results.to_string(index=False))
print(f"\n✅ Precision (macro): {precision_macro:.4f}")
print(f"✅ Recall (macro): {recall_macro:.4f}")
print(f"✅ F1-score (macro): {fscore_macro:.4f}")
print(f"✅ Accuracy: {accuracy:.4f}")

spark.stop()