import sys
import os
import time  # Thêm import time để đo thời gian
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Thêm cấp thư mục cha
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from spark_utils import create_spark_session
from model_utils import evaluate_model, bcolors
from data_preprocessing import preprocess_data, create_label_index, reduce_dimensions, handle_nan_infinity, create_feature_vector, load_metadata

# Khởi tạo Spark session
spark = create_spark_session("pyspark-lr-training")

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

# Huấn luyện Logistic Regression với đo thời gian
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=15000,  # Tương ứng với max_iter=15000
    regParam=0.01,  # Tương ứng với C=100 (regParam = 1/C)
    elasticNetParam=0.0,  # Chỉ dùng L2 regularization
    family="multinomial"  # Hỗ trợ phân loại đa lớp
)
print(bcolors.OKCYAN + "⏳ Training Logistic Regression model..." + bcolors.ENDC)
start_time = time.time()  # Bắt đầu đo thời gian
lr_model = lr.fit(train_df)
end_time = time.time()  # Kết thúc đo thời gian
training_time = end_time - start_time
print(bcolors.OKGREEN + f"✅ Training completed in {training_time:.2f} seconds" + bcolors.ENDC)

# Lưu mô hình
model_path = "s3a://mybucket/models/logistic_regression_model"
lr_model.write().overwrite().save(model_path)
print(bcolors.OKGREEN + f"✅ Model saved to {model_path}" + bcolors.ENDC)

# Dự đoán trên tập test
predictions = lr_model.transform(test_df)

# Đánh giá mô hình
f1_score, df_results, precision_macro, recall_macro, fscore_macro, accuracy = evaluate_model(predictions, label_to_name=label_to_name)
print(f"\n✅ F1-score: {f1_score:.4f}")
print(df_results.to_string(index=False))
print(f"\n✅ Precision (macro): {precision_macro:.4f}")
print(f"✅ Recall (macro): {recall_macro:.4f}")
print(f"✅ F1-score (macro): {fscore_macro:.4f}")
print(f"✅ Accuracy: {accuracy:.4f}")

spark.stop()