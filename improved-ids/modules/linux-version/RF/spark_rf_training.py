import sys
import os
import time  # Thêm import time để đo thời gian
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Thêm cấp thư mục cha
import time
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from spark_utils import create_spark_session
from model_utils import evaluate_model, bcolors
from data_preprocessing import preprocess_data, create_label_index, reduce_dimensions, handle_nan_infinity, create_feature_vector, load_metadata
from imblearn.over_sampling import SMOTE  # Thêm import SMOTE

# Khởi tạo Spark session
spark = create_spark_session("pyspark-rf-training")

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

# Chuyển đổi dữ liệu về Pandas để áp dụng SMOTE
print(bcolors.OKCYAN + "⏳ Applying SMOTE to handle class imbalance..." + bcolors.ENDC)
# Chuyển cột features (vector) và label thành Pandas DataFrame
pandas_df = df_reduced.select("features", "label").toPandas()
# Chuyển cột features (vector) thành mảng numpy
X = np.array([row["features"].toArray() for row in pandas_df.itertuples()])
y = pandas_df["label"].values

# Áp dụng SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Chuyển dữ liệu đã resample về Spark DataFrame
from pyspark.ml.linalg import Vectors
resampled_data = [(Vectors.dense(x), float(y)) for x, y in zip(X_resampled, y_resampled)]
df_resampled = spark.createDataFrame(resampled_data, ["features", "label"])
print(bcolors.OKGREEN + "✅ SMOTE applied successfully" + bcolors.ENDC)

# Train-Test Split
train_df, test_df = df_resampled.randomSplit([0.8, 0.2], seed=42)

# Huấn luyện Random Forest
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=200,
    maxDepth=42,
    minInstancesPerNode=2,
    featureSubsetStrategy="sqrt",
    impurity="gini",
    seed=42
)
print(bcolors.OKCYAN + "⏳ Training Random Forest model..." + bcolors.ENDC)
start_time = time.time()
rf_model = rf.fit(train_df)
end_time = time.time()
training_time = end_time - start_time
print(bcolors.OKGREEN + f"✅ Training completed in {training_time:.2f} seconds" + bcolors.ENDC)

# Lưu mô hình
model_path = "s3a://mybucket/models/random_forest_model"
rf_model.write().overwrite().save(model_path)
print(bcolors.OKGREEN + f"✅ Model saved to {model_path}" + bcolors.ENDC)

# Dự đoán trên tập test
predictions = rf_model.transform(test_df)

# Đánh giá mô hình
f1_score, df_results, precision_macro, recall_macro, fscore_macro, accuracy = evaluate_model(predictions, label_to_name=label_to_name)
print(f"\n✅ F1-score: {f1_score:.4f}")
print(df_results.to_string(index=False))
print(f"\n✅ Precision (macro): {precision_macro:.4f}")
print(f"✅ Recall (macro): {recall_macro:.4f}")
print(f"✅ F1-score (macro): {fscore_macro:.4f}")
print(f"✅ Accuracy: {accuracy:.4f}")

spark.stop()
