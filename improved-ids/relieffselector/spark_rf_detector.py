from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler, StringIndexer
import pandas as pd
from pyspark.sql.functions import col, when, lit
from pyspark.sql.types import StringType
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json

# Khởi tạo Spark session
spark = SparkSession.builder\
    .appName("predict-attack")\
    .master("spark://127.0.0.1:7077")\
    .config("spark.driver.host", "host.docker.internal") \
    .config("spark.driver.bindAddress", "0.0.0.0")\
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.cores", "1") \
    .config("spark.executor.instances", "2")\
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

# Đường dẫn tới mô hình đã lưu
model_path = "s3a://mybucket/models/random_forest_model"

# Tải mô hình
loaded_model = RandomForestClassificationModel.load(model_path)
print("✅ Model loaded successfully")

# Đọc dữ liệu mới từ S3
new_data_path = "s3a://mybucket/cicids2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
new_df = spark.read \
    .option("nullValue", "NA") \
    .option("emptyValue", "unknown") \
    .csv(new_data_path, header=True, inferSchema=True)

# Đổi tên cột Label
new_df = new_df.withColumnRenamed(" Label", "Label")

# Tạo cột Label_Category giống như trong code huấn luyện
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

# Tạo cột label bằng StringIndexer
label_indexer = StringIndexer(inputCol="Label_Category", outputCol="label")
label_indexer_model = label_indexer.fit(new_df)
new_df = label_indexer_model.transform(new_df)

# Tải global_top_features từ S3
global_top_features_json = spark.read.text("s3a://mybucket/models/global_top_features").collect()[0]["value"]
global_top_features = json.loads(global_top_features_json)

# Chuẩn bị dữ liệu mới: tạo vector đặc trưng
assembler = VectorAssembler(inputCols=global_top_features, outputCol="features")
new_df_prepared = assembler.transform(new_df).select("features", "label", "Label")

# Dự đoán trên dữ liệu mới
predictions = loaded_model.transform(new_df_prepared)

# Hiển thị kết quả dự đoán (10 dòng đầu tiên)
print("Dự đoán trên 10 dòng đầu tiên:")
predictions.select("Label", "label", "prediction").show(10)

# Tải label_to_name từ S3 để ánh xạ nhãn dự đoán
label_to_name_json = spark.read.text("s3a://mybucket/models/label_to_name").collect()[0]["value"]
label_to_name = json.loads(label_to_name_json)

# Ánh xạ prediction thành nhãn gốc
def map_label(index):
    return label_to_name.get(index, "unknown")

map_label_udf = udf(map_label, StringType())
predictions_with_labels = predictions.withColumn("predicted_label", map_label_udf(col("prediction")))

# Hiển thị kết quả với nhãn dự đoán
print("Kết quả với nhãn dự đoán (10 dòng đầu tiên):")
predictions_with_labels.select("Label", "label", "prediction", "predicted_label").show(10)

# Đánh giá độ chính xác của mô hình
y_validate = predictions_with_labels.select("label").toPandas()
y_predicted = predictions_with_labels.select("prediction").toPandas()

precision, recall, fscore, support = precision_recall_fscore_support(y_validate, y_predicted)
actual_labels = sorted(y_validate["label"].unique())
print(f"Unique Labels in Data: {actual_labels}")

# Ánh xạ nhãn số về nhãn gốc
original_labels = [label_to_name[label] for label in actual_labels]

# Tạo DataFrame kết quả
df_results = pd.DataFrame({
    'attack': actual_labels,
    'original_label': original_labels,
    'precision': precision,
    'recall': recall,
    'fscore': fscore
})

precision_macro, recall_macro, fscore_macro, _ = precision_recall_fscore_support(y_validate, y_predicted, average='macro')
accuracy = accuracy_score(y_validate, y_predicted)

# Hiển thị thông số đánh giá
print("\nDetailed Results:")
print(df_results.to_string(index=False))
print(f"\n✅ Precision (macro): {precision_macro:.4f}")
print(f"✅ Recall (macro): {recall_macro:.4f}")
print(f"✅ F1-score (macro): {fscore_macro:.4f}")
print(f"✅ Accuracy: {accuracy:.4f}")

# Lưu kết quả dự đoán nếu cần
predictions_with_labels.write.mode("overwrite").parquet("s3a://mybucket/predictions/predicted_results.parquet")
print("✅ Predictions saved to S3")