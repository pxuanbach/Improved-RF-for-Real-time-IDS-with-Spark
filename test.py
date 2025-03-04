from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Read CSV") \
    .getOrCreate()

csv_file_path = "/opt/bitnami/spark/data/cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
df.show()

spark.stop()