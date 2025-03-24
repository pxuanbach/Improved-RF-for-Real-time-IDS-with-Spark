import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Thêm cấp thư mục cha

import time
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import udf, lit
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from spark_utils import create_spark_session
from model_utils import evaluate_model, bcolors
from data_preprocessing import preprocess_data, create_label_index, reduce_dimensions, handle_nan_infinity, create_feature_vector, load_metadata
from imblearn.over_sampling import SMOTE

# Khởi tạo Spark session
start_time = time.time()
spark = create_spark_session("pyspark-rf-training")
end_time = time.time()
print(bcolors.OKCYAN + f"⏳ Spark session creation took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Load dữ liệu đã giảm chiều
start_time = time.time()
reduced_data_path = "s3a://mybucket/preprocessed_data/reduced_data.parquet"
df_reduced = spark.read.parquet(reduced_data_path)
print(bcolors.OKGREEN + f"✅ Loaded reduced data from {reduced_data_path}" + bcolors.ENDC)
end_time = time.time()
print(bcolors.OKCYAN + f"⏳ Loading reduced data took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Load metadata
start_time = time.time()
global_top_features, label_to_name = load_metadata(spark)
print(bcolors.OKGREEN + f"✅ Loaded {len(global_top_features)} global top features: {global_top_features}" + bcolors.ENDC)
print(bcolors.OKGREEN + "✅ Loaded label mapping:" + bcolors.ENDC)
for index, label in label_to_name.items():
    print(f"  - Index {index}: {label}")
end_time = time.time()
print(bcolors.OKCYAN + f"⏳ Loading metadata took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Chuyển đổi dữ liệu về Pandas để áp dụng SMOTE (phần này vẫn bị comment như trong script gốc)
# start_time = time.time()
# print(bcolors.OKCYAN + "⏳ Applying SMOTE to handle class imbalance..." + bcolors.ENDC)
# pandas_df = df_reduced.select("features", "label").toPandas()
# X = np.array([row["features"].toArray() for row in pandas_df.itertuples()])
# y = pandas_df["label"].values
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)
# from pyspark.ml.linalg import Vectors
# resampled_data = [(Vectors.dense(x), float(y)) for x, y in zip(X_resampled, y_resampled)]
# df_resampled = spark.createDataFrame(resampled_data, ["features", "label"])
# print(bcolors.OKGREEN + "✅ SMOTE applied successfully" + bcolors.ENDC)
# end_time = time.time()
# print(bcolors.OKCYAN + f"⏳ SMOTE processing took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

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
# Định nghĩa UDF để trích xuất giá trị từ vector
def extract_feature(vector, index):
    return float(vector[index])

extract_feature_udf = udf(extract_feature, DoubleType())

# Trích xuất các đặc trưng thành các cột riêng
for i, feature_name in enumerate(global_top_features):
    df_resampled = df_resampled.withColumn(
        feature_name,
        extract_feature_udf("features", lit(i))
    )

# Chỉ giữ lại các cột đặc trưng được chọn và nhãn
columns_to_keep = selected_features + ["label"]
df_resampled = df_resampled.select(columns_to_keep)

# Tái tạo vector đặc trưng từ các cột đã chọn
assembler = VectorAssembler(inputCols=selected_features, outputCol="features")
df_resampled = assembler.transform(df_resampled).select("features", "label")
print(bcolors.OKGREEN + "✅ Updated DataFrame with 18 selected features" + bcolors.ENDC)
end_time = time.time()
print(bcolors.OKCYAN + f"⏳ RFSelector processing took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Train-Test Split
start_time = time.time()
train_df, test_df = df_resampled.randomSplit([0.8, 0.2], seed=42)
end_time = time.time()
print(bcolors.OKCYAN + f"⏳ Train-test split took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Huấn luyện Random Forest
start_time = time.time()
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=200,
    maxDepth=15,
    minInstancesPerNode=2,
    featureSubsetStrategy="sqrt",
    impurity="gini",
    seed=42
)
print(bcolors.OKCYAN + "⏳ Training Random Forest model..." + bcolors.ENDC)
rf_model = rf.fit(train_df)
end_time = time.time()
training_time = end_time - start_time
print(bcolors.OKGREEN + f"✅ Training completed in {training_time:.2f} seconds" + bcolors.ENDC)

# Lưu mô hình
start_time = time.time()
model_path = "s3a://mybucket/models/random_forest_model"
rf_model.write().overwrite().save(model_path)
print(bcolors.OKGREEN + f"✅ Model saved to {model_path}" + bcolors.ENDC)
end_time = time.time()
print(bcolors.OKCYAN + f"⏳ Saving model took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Dự đoán trên tập test
start_time = time.time()
predictions = rf_model.transform(test_df)
end_time = time.time()
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
print(bcolors.OKCYAN + f"⏳ Model evaluation took {end_time - start_time:.2f} seconds" + bcolors.ENDC)

# Tổng thời gian thực hiện
total_start_time = time.time() - sum([end_time - start_time for start_time, end_time in zip([start_time], [end_time])])
print(bcolors.WARNING + f"⏳ Total execution time: {total_start_time:.2f} seconds" + bcolors.ENDC)

spark.stop()