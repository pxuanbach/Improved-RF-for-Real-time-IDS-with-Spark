from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import StringIndexer, VectorAssembler
from typing import Tuple, Dict, List, Literal
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, when

class FeatureEngineer:
    def __init__(
        self,
        spark: SparkSession,
    ) -> None:
        self.spark = spark

    def engineer_features(self, df: DataFrame, feature_cols: List[str]) -> Tuple[DataFrame, Dict[int, str]]:
        # First map Label_Category to numeric values
        distinct_labels = df.select("Label_Category").distinct().collect()
        label_mapping = {label.Label_Category: idx for idx, label in enumerate(distinct_labels)}

        # Create reverse mapping for later use
        label_to_name = {float(v): k for k, v in label_mapping.items()}
        print("Label mapping:", label_mapping)

        # Create feature vector
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        df_vector = assembler.transform(df).select("features", "Label_Category", "label_idx")

        return df_vector, label_to_name

    @staticmethod
    def calculate_splits(df_vector: DataFrame, max_records=15000) -> List[DataFrame]:
        total_records = df_vector.count()
        num_subsets = (total_records + max_records - 1) // max_records
        split_weights = [1.0 / num_subsets] * num_subsets
        return df_vector.randomSplit(split_weights, seed=42)
