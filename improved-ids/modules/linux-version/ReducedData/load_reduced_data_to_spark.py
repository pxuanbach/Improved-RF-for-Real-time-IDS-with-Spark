import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pyspark.sql import SparkSession
from spark_utils import create_spark_session
from model_utils import bcolors

# Thiết lập parser để nhận tham số dòng lệnh
parser = argparse.ArgumentParser(description="Upload reduced data from local to S3 with Spark")
parser.add_argument(
    "--input-dir",
    type=str,
    default=None,
    help="Directory containing the reduced data to upload. If not specified, defaults to 'preprocessed_data' directory in script location."
)
args = parser.parse_args()

# Xác định thư mục chứa file local
if args.input_dir:
    input_dir = args.input_dir
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "preprocessed_data")

print("Running on:", os.name)
print("Current working directory:", os.getcwd())

# Đường dẫn file local
local_path = os.path.join(input_dir, "reduced_data.parquet")

# Khởi tạo Spark session
spark = create_spark_session("upload-to-s3")

# Kiểm tra file có tồn tại không
if not os.path.exists(local_path):
    print(bcolors.FAIL + f"Error: File {local_path} does not exist!" + bcolors.ENDC)
    spark.stop()
    exit(1)

# Load file từ local
print(bcolors.HEADER + f"Loading data from {local_path}..." + bcolors.ENDC)
df_reduced = spark.read.parquet(local_path)
print(bcolors.OKBLUE + f"Loaded data with {df_reduced.count()} rows and {len(df_reduced.columns)} columns" + bcolors.ENDC)

# Đường dẫn S3 đích
s3_path = "s3a://mybucket/preprocessed_data/reduced_data.parquet"

# Upload lên S3
print(bcolors.HEADER + f"Uploading data to {s3_path}..." + bcolors.ENDC)
df_reduced.write.mode("overwrite").parquet(s3_path)
print(bcolors.OKGREEN + f"✅ Data successfully uploaded to {s3_path}" + bcolors.ENDC)

# Dừng Spark session
spark.stop()