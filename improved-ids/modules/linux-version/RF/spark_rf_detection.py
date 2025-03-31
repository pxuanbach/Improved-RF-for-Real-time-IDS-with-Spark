
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Thêm cấp thư mục cha

from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from spark_utils import create_spark_session
from data_preprocessing import preprocess_data, create_label_index, reduce_dimensions, handle_nan_infinity, create_feature_vector, load_metadata
from model_utils import evaluate_model,bcolors


# Khởi tạo Spark session
spark = create_spark_session("pyspark-rf-detection")

# Load dữ liệu
volume_files = ["s3a://mybucket/cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"]
df = spark.read.option("nullValue", "NA").option("emptyValue", "unknown").csv(volume_files, header=True, inferSchema=True)

# Lấy mẫu ngẫu nhiên
sample_size = 50
total_records = df.count()
df_sample = df.sample(False, sample_size / total_records, seed=42) if total_records > sample_size else df
print(bcolors.OKGREEN + f"✅ Sampled {df_sample.count()} records for detection" + bcolors.ENDC)

# Tiền xử lý dữ liệu
df_sample = preprocess_data(df_sample)

# Load metadata
global_top_features, label_to_name = load_metadata(spark)
print(bcolors.OKGREEN + f"✅ Loaded {len(global_top_features)} global top features: {global_top_features}" + bcolors.ENDC)
print(bcolors.OKGREEN + "✅ Loaded label mapping:" + bcolors.ENDC)
for index, label in label_to_name.items():
    print(f"  - Index {index}: {label}")

# Tạo label index
df_sample, _, _ = create_label_index(df_sample)

# Giảm chiều dữ liệu
df_sample = reduce_dimensions(df_sample, global_top_features)

# Xử lý NaN/Infinity
print(bcolors.HEADER + "Checking for NaN or Infinity in sample data..." + bcolors.ENDC)
df_sample = handle_nan_infinity(df_sample, global_top_features)

# Tạo vector đặc trưng
df_sample_prepared = create_feature_vector(df_sample, global_top_features).select("features", "label", "Label", "Label_Category")

# Load mô hình
model_path = "s3a://mybucket/models/random_forest_model"
loaded_model = RandomForestClassificationModel.load(model_path)
print(bcolors.OKGREEN + "✅ Model loaded successfully" + bcolors.ENDC)

# Dự đoán
predictions = loaded_model.transform(df_sample_prepared)

# Ánh xạ nhãn dự đoán
def map_label_to_category(index):
    return label_to_name.get(int(index), "unknown")

map_label_to_category_udf = udf(map_label_to_category, StringType())
predictions_with_labels = predictions.withColumn("actual_label", map_label_to_category_udf(col("label")))
predictions_with_labels = predictions_with_labels.withColumn("predicted_label", map_label_to_category_udf(col("prediction")))

# Hiển thị kết quả
print(bcolors.HEADER + f"Prediction Results ({df_sample.count()} rows):" + bcolors.ENDC)
predictions_with_labels.select("Label", "actual_label", "predicted_label").show(int(df_sample.count()))

# Đánh giá mô hình
f1_score, df_results, precision_macro, recall_macro, fscore_macro, accuracy = evaluate_model(predictions, label_to_name=label_to_name)
print(bcolors.HEADER + "\nDetailed Results:" + bcolors.ENDC)
print(df_results.to_string(index=False))
print(f"\n✅ F1-score: {f1_score:.4f}")
print(f"✅ Precision (macro): {precision_macro:.4f}")
print(f"✅ Recall (macro): {recall_macro:.4f}")
print(f"✅ F1-score (macro): {fscore_macro:.4f}")
print(f"✅ Accuracy: {accuracy:.4f}")

spark.stop()