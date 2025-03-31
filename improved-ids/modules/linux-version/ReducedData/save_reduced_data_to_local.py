import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pyspark.sql import SparkSession
from model_utils import bcolors  # Nếu không có bcolors, có thể xóa hoặc thay bằng print thường
from spark_utils import create_spark_session

# Thiết lập parser để nhận tham số dòng lệnh
parser = argparse.ArgumentParser(description="Download reduced data from S3 to local with Spark")
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Directory to save the reduced data. If not specified, defaults to 'preprocessed_data' directory in script location."
)
args = parser.parse_args()

# Xác định thư mục lưu kết quả
if args.output_dir:
    output_dir = args.output_dir
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "preprocessed_data")

# Đảm bảo thư mục đầu ra tồn tại
os.makedirs(output_dir, exist_ok=True)

# Đường dẫn file trên S3
s3_path = "s3a://mybucket/preprocessed_data/reduced_data.parquet"

# Đường dẫn local để lưu file
local_path = os.path.join(output_dir, "reduced_data.parquet")

# Khởi tạo Spark session
spark = create_spark_session("download-from-s3")

# Load file từ S3
print(bcolors.HEADER + f"Loading data from {s3_path}..." + bcolors.ENDC)
df_reduced = spark.read.parquet(s3_path)

# Lưu file về local
print(bcolors.HEADER + f"Saving data to {local_path}..." + bcolors.ENDC)
df_reduced.write.mode("overwrite").parquet(local_path)
print(bcolors.OKGREEN + f"✅ Data successfully saved to {local_path}" + bcolors.ENDC)

# Dừng Spark session
spark.stop()