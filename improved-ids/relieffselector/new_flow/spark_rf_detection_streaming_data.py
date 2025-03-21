from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, when, lit, udf, isnan, isnull, current_timestamp
from pyspark.sql.types import StringType, StructType, StructField, IntegerType, DoubleType, TimestampType, DoubleType
import json
import os
from datetime import datetime

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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
spark: SparkSession = SparkSession.builder\
    .appName("pyspark-rf-detection")\
    .master("spark://127.0.0.1:7077")\
    .config("spark.jars", jars) \
    .config("spark.driver.host", "host.docker.internal") \
    .config("spark.driver.bindAddress", "0.0.0.0")\
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.executor.instances", "3")\
    .config("spark.network.timeout", "1200s") \
    .config("spark.driver.maxResultSize", "2g") \
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
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")\
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
    .config("spark.hadoop.fs.s3a.attempts.maximum", "3") \
    .config("spark.task.maxFailures", "3") \
    .config("spark.shuffle.reduceLocality.enabled", "false") \
    .config("spark.shuffle.service.enabled", "true") \
    .getOrCreate()

# Load global_top_features và label_to_name
try:
    global_top_features_json = spark.read.text("s3a://mybucket/models/global_top_features").collect()[0]["value"]
    global_top_features = json.loads(global_top_features_json)
    print(bcolors.OKGREEN + f"✅ Loaded {len(global_top_features)} global top features: {global_top_features}" + bcolors.ENDC)
except Exception as e:
    print(bcolors.FAIL + f"❌ Failed to load global_top_features: {str(e)}" + bcolors.ENDC)
    spark.stop()
    exit(1)

try:
    label_to_name_json = spark.read.text("s3a://mybucket/models/label_to_name").collect()[0]["value"]
    label_to_name = {int(k): v for k, v in json.loads(label_to_name_json).items()}
    print(bcolors.OKGREEN + "✅ Loaded label mapping:" + bcolors.ENDC)
    for index, label in label_to_name.items():
        print(f"  - Index {index}: {label}")
except Exception as e:
    print(bcolors.FAIL + f"❌ Failed to load label_to_name: {str(e)}" + bcolors.ENDC)
    spark.stop()
    exit(1)

# Đảo ngược label_to_name để tạo ánh xạ từ Label_Category sang label (chỉ số)
name_to_label = {v: k for k, v in label_to_name.items()}

# Định nghĩa schema cho dữ liệu CSV (chỉ giữ một cột Label)
schema_fields = []
for feature in global_top_features:
    if "Port" in feature:
        schema_fields.append(StructField(feature, IntegerType(), True))
    else:
        schema_fields.append(StructField(feature, DoubleType(), True))
schema_fields.append(StructField(" Label", StringType(), True))  # Chỉ giữ cột " Label" (có khoảng trắng)
schema = StructType(schema_fields)

# Đọc dữ liệu từ thư mục chứa các file CSV
streaming_input_path = "s3a://mybucket/cicids2017/"
streaming_df = spark.readStream \
    .schema(schema) \
    .option("header", "true") \
    .option("maxFilesPerTrigger", "1") \
    .option("nullValue", "NA") \
    .option("emptyValue", "unknown") \
    .option("cleanSource", "delete") \
    .csv(streaming_input_path)

# Hàm tiền xử lý dữ liệu
def preprocess_data(df):
    # Đổi tên cột " Label" thành "Label" và loại bỏ cột " Label"
    df = df.withColumnRenamed(' Label', 'Label')
    if ' Label' in df.columns:  # Đảm bảo cột " Label" không còn tồn tại
        df = df.drop(' Label')

    # Thay thế các giá trị không mong muốn trong cột Label
    df = df.replace(['Heartbleed', 'Web Attack � Sql Injection', 'Infiltration'], None, subset=['Label'])
    df = df.dropna(how='any')

    # Thay thế các giá trị cụ thể trong cột Label
    df = df.withColumn('Label', when(col('Label') == 'Web Attack � Brute Force', 'Brute Force').otherwise(col('Label')))
    df = df.withColumn('Label', when(col('Label') == 'Web Attack � XSS', 'XSS').otherwise(col('Label')))
    df = df.withColumn('Attack', when(col('Label') == 'BENIGN', 0).otherwise(1))

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
    df = df.withColumn('Label_Category', conditions[0])
    for condition in conditions[1:]:
        df = df.withColumn('Label_Category', when(col('Label_Category').isNull(), condition).otherwise(col('Label_Category')))

    # Tạo cột label bằng UDF thay vì StringIndexer
    def category_to_label(category):
        return float(name_to_label.get(category, -1))  # Trả về -1 nếu không tìm thấy

    category_to_label_udf = udf(category_to_label, DoubleType())
    df = df.withColumn("label", category_to_label_udf(col("Label_Category")))

    # Giảm chiều dữ liệu với global_top_features
    exclude_cols = ['Label', 'Label_Category', 'Attack']
    columns_to_keep = global_top_features + exclude_cols
    df = df.select(columns_to_keep)

    # Kiểm tra và xử lý NaN/Infinity
    for col_name in global_top_features:
        df = df.withColumn(col_name, when(col(col_name) == float("inf"), None).otherwise(col(col_name)))
        df = df.withColumn(col_name, when(isnan(col(col_name)) | isnull(col(col_name)), 0.0).otherwise(col(col_name)))

    # Tạo vector đặc trưng
    assembler = VectorAssembler(inputCols=global_top_features, outputCol="features")
    df = assembler.transform(df).select("features", "label", "Label", "Label_Category")
    return df

# Ánh xạ nhãn dự đoán về nhãn gốc
def map_label_to_category(index):
    return label_to_name.get(int(index), "unknown")

map_label_to_category_udf = udf(map_label_to_category, StringType())

# Load mô hình đã lưu
try:
    model_path = "s3a://mybucket/models/random_forest_model"
    loaded_model = RandomForestClassificationModel.load(model_path)
    print(bcolors.OKGREEN + "✅ Model loaded successfully" + bcolors.ENDC)
except Exception as e:
    print(bcolors.FAIL + f"❌ Failed to load model: {str(e)}" + bcolors.ENDC)
    spark.stop()
    exit(1)

# Xử lý dữ liệu streaming
streaming_df_prepared = preprocess_data(streaming_df)

# Dự đoán trên dữ liệu streaming
streaming_predictions = loaded_model.transform(streaming_df_prepared)
streaming_predictions_with_labels = streaming_predictions.withColumn("actual_label", map_label_to_category_udf(col("label")))
streaming_predictions_with_labels = streaming_predictions_with_labels.withColumn("predicted_label", map_label_to_category_udf(col("prediction")))
streaming_predictions_with_labels = streaming_predictions_with_labels.withColumn("timestamp", current_timestamp())

# Hiển thị kết quả dự đoán (stream)
def display_predictions(df_batch, batch_id):
    print(bcolors.HEADER + f"Prediction Results (Batch {batch_id}):" + bcolors.ENDC)
    df_batch.select("Label", "actual_label", "predicted_label", "timestamp").show()

display_query = streaming_predictions_with_labels.writeStream \
    .foreachBatch(display_predictions) \
    .trigger(processingTime="10 seconds") \
    .start()

# Gửi cảnh báo cho dữ liệu streaming
streaming_attack_predictions = streaming_predictions_with_labels.filter(col("predicted_label") != "benign")
def send_alert(df_batch, batch_id):
    if df_batch.count() > 0:
        print(bcolors.FAIL + f"⚠️ ALERT: Detected {df_batch.count()} potential attacks in batch {batch_id}!" + bcolors.ENDC)
        df_batch.select("Label", "actual_label", "predicted_label", "timestamp").show()
        # Ghi log vào file
        with open("alerts.log", "a") as f:
            f.write(f"{datetime.now()}: Detected {df_batch.count()} potential attacks in batch {batch_id}\n")
            f.write(df_batch.select("Label", "actual_label", "predicted_label", "timestamp").toPandas().to_string() + "\n")

attack_query = streaming_attack_predictions.writeStream \
    .foreachBatch(send_alert) \
    .trigger(processingTime="10 seconds") \
    .start()

# Lưu kết quả streaming
query = streaming_predictions_with_labels.writeStream \
    .format("parquet") \
    .option("path", "s3a://mybucket/predictions/streaming_results/") \
    .option("checkpointLocation", "s3a://mybucket/checkpoints/") \
    .trigger(processingTime="10 seconds") \
    .start()

# Xuất báo cáo cho dữ liệu streaming
def save_report(df_batch, batch_id):
    # Đường dẫn lưu trên S3
    report_path = f"s3a://mybucket/reports/detection_report_batch_{batch_id}.csv"
    # Đường dẫn lưu local
    local_file = f"local_report_batch_{batch_id}.csv"

    # Tạo DataFrame báo cáo
    report_df = df_batch.select("Label", "actual_label", "predicted_label", "timestamp")
    report_df = report_df.withColumn("attack_count", lit(df_batch.filter(col("predicted_label") != "benign").count()))

    # Lưu bản local
    report_df.toPandas().to_csv(local_file, index=False)
    print(bcolors.OKGREEN + f"✅ Local report for batch {batch_id} saved to {local_file}" + bcolors.ENDC)

    # Lưu lên S3 bằng Spark
    report_df.coalesce(1).write \
        .mode("overwrite") \
        .csv(report_path)
    print(bcolors.OKGREEN + f"✅ Report for batch {batch_id} saved to {report_path}" + bcolors.ENDC)

report_query = streaming_predictions_with_labels.writeStream \
    .foreachBatch(save_report) \
    .trigger(processingTime="10 seconds") \
    .start()

# Chờ pipeline streaming hoàn tất
query.awaitTermination()
attack_query.awaitTermination()
display_query.awaitTermination()
report_query.awaitTermination()

spark.stop()