from typing import List, Set, Optional, Literal
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, lit, isnan, isnull, sum
from pyspark.sql.types import DoubleType
import os
from modules.utils import bcolors

class DataPreprocessor:
    def __init__(
        self,
        spark: SparkSession,
    ) -> None:
        self.spark = spark

        # Initialize variables
        self.feature_cols: List[str] = []
        self.exclude_cols: Set[str] = {'Label', 'Label_Category', 'Attack'}

    def process(self, volume_files: List[str]) -> DataFrame:
        return self._preprocess_data(volume_files)

    def _calculate_optimal_partitions(self, spark_df: DataFrame, target_size_mb: int = 64) -> int:
        # Reduce target partition size for better network handling
        total_size = spark_df.count() * len(spark_df.columns) * 8
        total_size_mb = total_size / (1024 * 1024)
        return max(100, int(total_size_mb / target_size_mb))

    def _handle_invalid_values(self, df: DataFrame, col_name: str) -> DataFrame:
        # Cache DataFrame if not already cached
        if not df.is_cached:
            df.cache()

        # Check invalid values using a single pass
        invalid_counts = df.select([
            sum((isnan(col(col_name)) | isnull(col(col_name))).cast("long")).alias("nan_count"),
            sum((col(col_name) == float("inf")).cast("long")).alias("inf_count")
        ]).collect()[0]

        count_nan = invalid_counts["nan_count"]
        count_inf = invalid_counts["inf_count"]

        if count_nan == 0 and count_inf == 0:
            return df

        print(f"{bcolors.WARNING}⚠️ Column {col_name} has {count_nan} NaN and {count_inf} Infinity values{bcolors.ENDC}")

        # Use more efficient column operations
        df_cleaned = df.withColumn(
            col_name,
            when(col(col_name) == float("inf"), None)
            .when(isnan(col(col_name)) | isnull(col(col_name)), None)
            .otherwise(col(col_name))
        )

        # Calculate mean value more efficiently
        mean_value = df_cleaned.agg({col_name: "avg"}).collect()[0][0]
        mean_value = 0.0 if mean_value is None else mean_value

        # Fill null values
        df_cleaned = df_cleaned.fillna({col_name: mean_value})

        return df_cleaned

    def _preprocess_data(self, volume_files: List[str]) -> DataFrame:
        try:
            # Read data with optimized settings and retry logic
            df = (self.spark.read
                .option("header", "true")
                .option("inferSchema", "true")
                .option("mode", "PERMISSIVE")
                .option("nullValue", "NA")
                .option("escape", "\"")
                .option("multiLine", "true")
                .option("maxFilesPerTrigger", "1")  # Process files one at a time
                .csv(volume_files))

            # Optimize memory usage
            num_partitions = self._calculate_optimal_partitions(df, 128)
            print(f"Repartitioning to {num_partitions} partitions")
            df = df.repartition(num_partitions).cache()
            df.count()  # Force caching

            # Basic preprocessing
            df = df.withColumnRenamed(' Label', 'Label')
            df = df.replace(['Heartbleed', 'Web Attack � Sql Injection', 'Infiltration'], None, subset=['Label'])
            df = df.dropna(how='any')

            # Label transformation
            df = df.withColumn('Label',
                when(col('Label') == 'Web Attack � Brute Force', 'Brute Force')
                .when(col('Label') == 'Web Attack � XSS', 'XSS')
                .otherwise(col('Label')))

            df = df.withColumn('Attack', when(col('Label') == 'BENIGN', 0).otherwise(1))

            # Create Label_Category
            attack_group = {
                'BENIGN': 'benign', 'DoS Hulk': 'dos', 'PortScan': 'probe',
                'DDoS': 'ddos', 'DoS GoldenEye': 'dos', 'FTP-Patator': 'brute_force',
                'SSH-Patator': 'brute_force', 'DoS slowloris': 'dos',
                'DoS Slowhttptest': 'dos', 'Bot': 'botnet',
                'Brute Force': 'web_attack', 'XSS': 'web_attack'
            }

            conditions = [when(col('Label') == k, lit(v)) for k, v in attack_group.items()]
            df = df.withColumn('Label_Category', conditions[0])
            for condition in conditions[1:]:
                df = df.withColumn('Label_Category',
                    when(col('Label_Category').isNull(), condition).otherwise(col('Label_Category')))

            # Extract features before saving
            self.feature_cols = self._extract_feature_cols(df)

            # Handle invalid values in numeric columns
            print(f"{bcolors.HEADER}Checking for NaN or Infinity values...{bcolors.ENDC}")
            for col_name in self.feature_cols:
                df = self._handle_invalid_values(df, col_name)

            distinct_labels = df.select("Label_Category").distinct().collect()
            label_mapping = {label.Label_Category: idx for idx, label in enumerate(distinct_labels)}

            # Apply numeric mapping using when/otherwise
            mapping_expr = None
            for label, idx in label_mapping.items():
                if mapping_expr is None:
                    mapping_expr = when(col("Label_Category") == label, float(idx))  # Cast to float explicitly
                else:
                    mapping_expr = mapping_expr.when(col("Label_Category") == label, float(idx))

            # Add default value to handle any unmapped categories
            mapping_expr = mapping_expr.otherwise(float(-1))

            # Create and verify numeric labels
            df = df.withColumn("label_idx", mapping_expr.cast(DoubleType()))

            # Verify the transformation
            print("Label column type:", df.select('label_idx').dtypes)
            df.select('Label_Category', 'label_idx').distinct().show(10)

            # Clean up cache after processing
            df.unpersist()
            return df

        except Exception as e:
            print(f"Error during preprocessing: {e}")
            raise

    def _extract_feature_cols(self, df: DataFrame) -> List[str]:
        cols = [col_name for col_name in df.columns
               if col_name not in self.exclude_cols
               and df.schema[col_name].dataType.typeName() in ('double', 'integer', 'float')]
        self.feature_cols = cols
        return cols

    def get_feature_cols(self) -> List[str]:
        if not self.feature_cols:
            raise ValueError("Feature columns not yet extracted. Run process first.")
        return self.feature_cols
