import os
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler, StringIndexer
import pandas as pd
from pyspark.sql.functions import col, when, lit, udf, isnan, isnull
from pyspark.sql.types import StringType
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json

# Đường dẫn tới thư mục jars
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
spark = SparkSession.builder\
    .appName("pyspark-notebook")\
    .master("spark://127.0.0.1:7077")\
    .config("spark.jars", jars) \
    .config("spark.driver.host", "host.docker.internal") \
    .config("spark.driver.bindAddress", "0.0.0.0")\
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.executor.instances", "3")\
    .config("spark.network.timeout", "800s") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "2g") \
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
    .config("spark.hadoop.fs.s3a.attempts.maximum", "0") \
    .getOrCreate()

# Tải mô hình
model_path = "s3a://mybucket/models/random_forest_model"
loaded_model = RandomForestClassificationModel.load(model_path)
print("✅ Model loaded successfully")

# Đọc dữ liệu mới từ S3
new_data_path = "s3a://mybucket/cicids2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
new_df = spark.read \
    .option("nullValue", "NA") \
    .option("emptyValue", "unknown") \
    .csv(new_data_path, header=True, inferSchema=True) \
    .repartition(18)# Giữ giống file huấn luyện


# Đổi tên cột Label
new_df = new_df.withColumnRenamed(" Label", "Label")

# Loại bỏ nhãn không mong muốn (giống huấn luyện)
new_df = new_df.replace(['Heartbleed', 'Web Attack � Sql Injection', 'Infiltration'], None, subset=['Label'])
new_df = new_df.dropna(how='any')  # Loại bỏ hàng có NaN

# Đổi tên nhãn (giống huấn luyện)
new_df = new_df.withColumn('Label', when(col('Label') == 'Web Attack � Brute Force', 'Brute Force').otherwise(col('Label')))
new_df = new_df.withColumn('Label', when(col('Label') == 'Web Attack � XSS', 'XSS').otherwise(col('Label')))

# Tạo cột Label_Category
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
new_df = new_df.withColumn('Label_Category', conditions[0])
for condition in conditions[1:]:
    new_df = new_df.withColumn('Label_Category', when(col('Label_Category').isNull(), condition).otherwise(col('Label_Category')))

# Tải label_to_name từ huấn luyện và dùng ánh xạ này
label_to_name_json = spark.read.text("s3a://mybucket/models/label_to_name").collect()[0]["value"]
label_to_name = json.loads(label_to_name_json)
print("Label to Name Mapping from Model:", label_to_name)

# Tạo cột label dựa trên label_to_name từ huấn luyện
def map_to_label(category):
    for idx, name in label_to_name.items():
        if name == category:
            return float(idx)
    return None

map_to_label_udf = udf(map_to_label, StringType())
new_df = new_df.withColumn("label", map_to_label_udf(col("Label_Category")))
new_df = new_df.filter(col("label").isNotNull())  # Loại bỏ nhãn không có trong huấn luyện

# Tải global_top_features từ S3
global_top_features_json = spark.read.text("s3a://mybucket/models/global_top_features").collect()[0]["value"]
global_top_features = json.loads(global_top_features_json)

# Xử lý NaN/Infinity trong các cột đặc trưng
print("Kiểm tra và xử lý NaN/Infinity trong các cột đặc trưng:")
for feature in global_top_features:
    nan_count = new_df.filter(isnan(col(feature)) | isnull(col(feature))).count()
    inf_count = new_df.filter(col(feature) == float("inf")).count()
    if nan_count > 0 or inf_count > 0:
        print(f"Cột {feature}: {nan_count} NaN, {inf_count} Infinity")
        new_df = new_df.withColumn(feature, when(col(feature) == float("inf"), None).otherwise(col(feature)))
        mean_val = new_df.selectExpr(f"avg(`{feature}`) as mean_val").collect()[0]["mean_val"] or 0.0
        new_df = new_df.fillna({feature: mean_val})

# Chuẩn bị vector đặc trưng
assembler = VectorAssembler(inputCols=global_top_features, outputCol="features", handleInvalid="skip")
new_df_prepared = assembler.transform(new_df).select("features", "label", "Label", "Label_Category")

# Dự đoán
predictions = loaded_model.transform(new_df_prepared)

# Ánh xạ nhãn
def map_label(index):
    return label_to_name.get(str(int(float(index))), "unknown")

map_label_udf = udf(map_label, StringType())
predictions_with_labels = predictions\
    .withColumn("predicted_label", map_label_udf(col("prediction")))\
    .withColumn("actual_label", map_label_udf(col("label")))

# Hiển thị kết quả
print("Kết quả dự đoán (10 dòng đầu tiên):")
predictions_with_labels.select("Label", "actual_label", "predicted_label").show(10)

# Đánh giá
y_validate = predictions_with_labels.select("label").toPandas()["label"]
y_predicted = predictions_with_labels.select("prediction").toPandas()["prediction"]

all_labels = sorted(set(y_validate.unique()) | set(y_predicted.unique()))
precision, recall, fscore, support = precision_recall_fscore_support(
    y_validate, y_predicted, labels=all_labels, zero_division=0
)

df_results = pd.DataFrame({
    'attack': all_labels,
    'original_label': [label_to_name.get(str(int(label)), "unknown") for label in all_labels],
    'precision': precision,
    'recall': recall,
    'fscore': fscore
})

precision_macro, recall_macro, fscore_macro, _ = precision_recall_fscore_support(
    y_validate, y_predicted, average='macro', zero_division=0
)
accuracy = accuracy_score(y_validate, y_predicted)

# Hiển thị thông số
print("\nDetailed Results:")
print(df_results.to_string(index=False))
print(f"\n✅ Precision (macro): {precision_macro:.4f}")
print(f"✅ Recall (macro): {recall_macro:.4f}")
print(f"✅ F1-score (macro): {fscore_macro:.4f}")
print(f"✅ Accuracy: {accuracy:.4f}")

# Dừng SparkSession
spark.stop()