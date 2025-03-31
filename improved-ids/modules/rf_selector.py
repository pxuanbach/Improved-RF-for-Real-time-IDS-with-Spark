from typing import List, Dict, Tuple
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import col
import os
import json

from modules.utils import bcolors

FEATURES_DIR = "./data/features"

class RFSelector:
    def __init__(
        self,
        spark: SparkSession,
        n_features: int = 10,
        n_trees: int = 50,
        max_depth: int = 10
    ) -> None:
        self.spark = spark
        self.n_features = n_features
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.features_path = os.path.join(FEATURES_DIR, "best_features.json")
        os.makedirs(FEATURES_DIR, exist_ok=True)

    def select_features(
        self,
        df: DataFrame,
        feature_cols: List[str],
        label_col: str = "label_idx"
    ) -> Tuple[List[str], DataFrame]:
        """Select top features using Random Forest importance scores"""
        try:
            # First check if we have cached features
            if os.path.exists(self.features_path):
                with open(self.features_path, 'r') as f:
                    selected_features = json.load(f)
                print(f"{bcolors.OKGREEN}Loaded {len(selected_features)} cached features{bcolors.ENDC}")
                return selected_features, self._transform_data(df, selected_features)

            print(f"{bcolors.HEADER}Training Random Forest for feature selection...{bcolors.ENDC}")

            # Prepare feature vector
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            df_vector = assembler.transform(df)

            # Train Random Forest
            rf = RandomForestClassifier(
                featuresCol="features",
                labelCol=label_col,
                numTrees=self.n_trees,
                maxDepth=self.max_depth,
                seed=42
            )
            rf_model = rf.fit(df_vector)

            # Get feature importances
            importances = rf_model.featureImportances
            feature_scores = [
                (feature, float(importance))
                for feature, importance in zip(feature_cols, importances)
            ]

            # Sort by importance
            feature_scores.sort(key=lambda x: x[1], reverse=True)

            # Select top N features
            selected_features = [f[0] for f in feature_scores[:self.n_features]]

            # Save selected features
            with open(self.features_path, 'w') as f:
                json.dump(selected_features, f)

            print(f"{bcolors.OKGREEN}Selected {len(selected_features)} best features{bcolors.ENDC}")
            print("Top features:", selected_features[:10])

            # Transform data with selected features
            df_selected = self._transform_data(df, selected_features)

            return selected_features, df_selected

        except Exception as e:
            print(f"{bcolors.FAIL}Error during feature selection: {str(e)}{bcolors.ENDC}")
            raise

    def _transform_data(self, df: DataFrame, selected_features: List[str]) -> DataFrame:
        """Transform data to include only selected features"""
        try:
            # Select only important features plus labels
            columns_to_keep = selected_features + ["Label_Category", "label_idx"]
            df_selected = df.select(columns_to_keep)

            df.unpersist(blocking=True)
            self.spark.sparkContext._jvm.System.gc()

            return df_selected

        except Exception as e:
            print(f"{bcolors.FAIL}Error transforming data: {str(e)}{bcolors.ENDC}")
            raise
