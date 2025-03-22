# Standard library imports
import os
import json
import time
from datetime import datetime

# Third-party imports
from pyspark.sql import SparkSession

# Local application imports
from modules.data_preprocessor import DataPreprocessor
from modules.feature_engineer import FeatureEngineer
from modules.relieff_selector import ReliefFFeatureSelector
from modules.dimension_reducer import DimensionReducer
from modules.rf_selector import RFSelector
from modules.model_trainer import ModelTrainer
from modules.relieffselector import ReliefFSelector

def log_time(section_name: str, start_time: float) -> None:
    elapsed = time.time() - start_time
    print(f"⏱️ {section_name} took {elapsed:.2f} seconds")
    return elapsed

def main():
    # System information
    print("Running on:", os.name)
    print("Current working directory:", os.getcwd())

    total_start = time.time()
    section_times = {}

    #region SPARK INITIALIZATION
    spark: SparkSession = SparkSession.builder\
        .appName("pyspark-notebook")\
        .master("spark://127.0.0.1:7077")\
        .config("spark.driver.host", "host.docker.internal") \
        .config("spark.driver.bindAddress", "0.0.0.0")\
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "2") \
        .config("spark.executor.instances", "1")\
        .config("spark.network.timeout", "1200s") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "2g") \
        .config("spark.default.parallelism", "24") \
        .config("spark.shuffle.io.maxRetries", "5") \
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
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.broadcastTimeout", "1200") \
        .config("spark.sql.autoBroadcastJoinThreshold", 10 * 1024 * 1024) \
        .config("spark.executor.heartbeatInterval", "60s") \
        .config("spark.storage.blockManagerSlaveTimeoutMs", "1800000") \
        .getOrCreate()
    #endregion

    # Data source configuration
    volume_files = [
        "s3a://mybucket/cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "s3a://mybucket/cicids2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "s3a://mybucket/cicids2017/Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "s3a://mybucket/cicids2017/Monday-WorkingHours.pcap_ISCX.csv",
        "s3a://mybucket/cicids2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "s3a://mybucket/cicids2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "s3a://mybucket/cicids2017/Tuesday-WorkingHours.pcap_ISCX.csv",
        "s3a://mybucket/cicids2017/Wednesday-workingHours.pcap_ISCX.csv",
    ]

    #region DATA PREPROCESSING AND FEATURE ENGINEERING
    start_time = time.time()
    # Initialize preprocessor with JSON format
    preprocessor = DataPreprocessor(spark)

    # Load or preprocess data
    df = preprocessor.process(volume_files)
    feature_cols = preprocessor.get_feature_cols()
    section_times['preprocessing'] = log_time("Preprocessing", start_time)

    #region FEATURE ENGINEERING
    start_time = time.time()
    # Initialize feature engineer with JSON format
    feature_engineer = FeatureEngineer(spark)

    # Feature engineering
    df_vector, label_to_name = feature_engineer.engineer_features(df, feature_cols)
    df_splits = feature_engineer.calculate_splits(df_vector)
    section_times['feature_engineering'] = log_time("Feature Engineering", start_time)
    #endregion

    #region RELIEFF FEATURE SELECTION
    start_time = time.time()
    # Feature selection
    relieff_selector = ReliefFFeatureSelector(spark, feature_cols)
    selector = ReliefFSelector().setSelectionThreshold(0.3).setNumNeighbors(10).setSampleSize(8)
    global_top_features, global_top_weights = relieff_selector.load_or_select_features(df_splits, selector)
    section_times['feature_selection'] = log_time("Feature Selection", start_time)
    #endregion

    #region DIMENSION REDUCTION
    start_time = time.time()
    # Dimension reduction with JSON format
    dimension_reducer = DimensionReducer(spark, output_format="csv", batch_size=50000)
    df_reduced = dimension_reducer.reduce_dimensions(
        df, global_top_features, list(preprocessor.exclude_cols) + ['label_idx']
    )
    section_times['dimension_reduction'] = log_time("Dimension Reduction", start_time)
    #endregion

    #region RF FEATURE SELECTION
    start_time = time.time()
    # Initialize RF selector with desired number of features
    rf_selector = RFSelector(spark, n_features=18, n_trees=50, max_depth=10)

    # Select best features using RF importance
    best_features, df_rf_selected = rf_selector.select_features(
        df_reduced,
        global_top_features,
        label_col="label_idx"
    )
    section_times['rf_selection'] = log_time("RF Feature Selection", start_time)
    #endregion

    #region MODEL TRAINING AND EVALUATION
    start_time = time.time()
    # Initialize and use model trainer with RF-selected features
    model_trainer = ModelTrainer(spark, numTrees=20, maxDepth=10)
    rf_model, metrics = model_trainer.train_and_evaluate(df_rf_selected, label_to_name, best_features)
    section_times['model_training'] = log_time("Model Training", start_time)
    #endregion

    # Save execution times
    total_time = time.time() - total_start
    section_times['total'] = total_time

    execution_log = {
        'timestamp': datetime.now().isoformat(),
        'total_time': total_time,
        'section_times': section_times
    }

    os.makedirs('./data/logs', exist_ok=True)
    with open('./data/logs/execution_times.json', 'w') as f:
        json.dump(execution_log, f, indent=2)

    print(f"\n✨ Total execution time: {total_time:.2f} seconds")

    # Stop SparkSession
    spark.stop()

if __name__ == "__main__":
    main()
