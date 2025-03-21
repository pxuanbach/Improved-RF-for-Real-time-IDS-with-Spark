from typing import List, Set, Literal, Dict
import json
import os
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, isnan, isnull

from modules.utils import bcolors

REDUCED_PATH = "./data/reduced"
FEATURES_DIR = "./data/features"

class DimensionReducer:
    def __init__(
        self,
        spark: SparkSession,
        output_format: Literal["json", "csv"] = "json",
        batch_size: int = 15000
    ) -> None:
        self.spark = spark
        self.output_format = output_format
        self.batch_size = batch_size
        os.makedirs(REDUCED_PATH, exist_ok=True)

        self.reduced_data_path = f"{REDUCED_PATH}/reduced_data.{output_format}"
        # Ensure reduced path exists
        os.makedirs(os.path.dirname(self.reduced_data_path), exist_ok=True)

    def _load_existing_reduced_data(self) -> DataFrame:
        """Load existing reduced data files if they exist"""
        try:
            # Check for existing part files
            part_files = [f for f in os.listdir(REDUCED_PATH)
                         if f.startswith('reduced_data.') and '.part' in f]

            if not part_files:
                return None

            print(f"{bcolors.HEADER}Found {len(part_files)} existing reduced data files{bcolors.ENDC}")

            # Load all part files
            dfs = []
            for file in sorted(part_files):
                file_path = os.path.join(REDUCED_PATH, file)
                if self.output_format == "json":
                    pdf = pd.read_json(file_path, lines=True)
                else:  # csv
                    pdf = pd.read_csv(file_path)
                df = self.spark.createDataFrame(pdf)
                dfs.append(df)

            # Union all DataFrames
            if dfs:
                final_df = dfs[0]
                for df in dfs[1:]:
                    final_df = final_df.union(df)
                print(f"{bcolors.OKGREEN}Successfully loaded existing reduced data{bcolors.ENDC}")
                return final_df

            return None

        except Exception as e:
            print(f"{bcolors.WARNING}Error loading existing data: {str(e)}{bcolors.ENDC}")
            return None

    def reduce_dimensions(
        self,
        df: DataFrame,
        global_top_features: List[str],
        exclude_cols: List[str],
        load_exists: bool = True
    ) -> DataFrame:
        """Reduce dimensions using pre-selected important features"""
        try:
            if load_exists:
                # First try to load existing reduced data
                existing_df = self._load_existing_reduced_data()
                if existing_df is not None:
                    return existing_df

            print(f"{bcolors.HEADER}No existing reduced data found, performing reduction...{bcolors.ENDC}")

            # Original reduction logic
            print(f"{bcolors.HEADER}Found {len(global_top_features)} important features{bcolors.ENDC}")

            # Make sure we include 'label' column in the selection
            columns_to_select = exclude_cols + global_top_features
            # if 'label' not in columns_to_select:
            #     columns_to_select.append('label')

            df_reduced = df.select(columns_to_select)

            # Verify label column type
            print("Label column type:", df_reduced.select('label').dtypes)
            print("Sample labels:", df_reduced.select('label').distinct().show())

            self._save_reduced_data(df_reduced)
            print(f"{bcolors.OKGREEN}Reduced data saved to {self.reduced_data_path}{bcolors.ENDC}")
            return df_reduced

        except Exception as e:
            print(f"{bcolors.FAIL}Error during dimension reduction: {str(e)}{bcolors.ENDC}")
            raise

    def _save_reduced_data(self, df_reduced: DataFrame) -> None:
        """Save reduced data to local file in batches"""
        try:
            total_count = df_reduced.count()
            num_batches = (total_count + self.batch_size - 1) // self.batch_size

            num_subsets = (total_count + self.batch_size - 1) // self.batch_size
            split_weights = [1.0 / num_subsets] * num_subsets
            df_reduced_split = df_reduced.randomSplit(split_weights, seed=42)

            print(f"{bcolors.HEADER}Saving {total_count} records in {num_batches} batches...{bcolors.ENDC}")

            for batch_idx, batch_df in enumerate(df_reduced_split):
                # Get batch of records
                pd_batch = batch_df.toPandas()

                # Construct batch filename
                batch_path = f"{self.reduced_data_path}.part{batch_idx}"

                # Save batch based on format
                if self.output_format == "json":
                    pd_batch.to_json(
                        batch_path,
                        orient='records',
                        lines=True,
                        force_ascii=False
                    )
                else:  # csv
                    pd_batch.to_csv(
                        batch_path,
                        index=False,
                        encoding='utf-8'
                    )

                batch_df.unpersist(blocking=True)
                self.spark.sparkContext._jvm.System.gc()

                print(f"{bcolors.OKGREEN}✓ Saved batch {batch_idx + 1}/{num_batches} to {batch_path}{bcolors.ENDC}")

            # Uncache DataFrame after we're done
            df_reduced.unpersist()
            print(f"{bcolors.OKGREEN}✓ All batches saved successfully to {self.reduced_data_path}*{bcolors.ENDC}")

        except Exception as e:
            print(f"{bcolors.FAIL}Error saving data locally: {str(e)}{bcolors.ENDC}")
            raise
