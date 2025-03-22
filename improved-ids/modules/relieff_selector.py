import time
import json
import numpy as np
from typing import List, Tuple, Dict
from pyspark.sql import SparkSession, DataFrame
from datetime import datetime
import os

from modules.utils import save_checkpoint, clear_checkpoint, load_checkpoint, bcolors
from relieffselector import ReliefFSelector


FEATURES_DIR = "./data/features"

class ReliefFFeatureSelector:
    def __init__(self, spark: SparkSession, feature_cols: List[str]) -> None:
        self.spark = spark
        self.feature_cols = feature_cols
        os.makedirs(FEATURES_DIR, exist_ok=True)
        self.important_features_path = f"{FEATURES_DIR}/important_features.json"
        self.fs_stats_path = f"{FEATURES_DIR}/fs_stats.json"

    def load_or_select_features(
        self,
        df_splits: List[DataFrame],
        selector: ReliefFSelector
    ) -> Tuple[List[str], np.ndarray]:
        try:
            with open(self.important_features_path, "r") as f:
                selected_features = json.load(f)
                return selected_features["global_top_features"], selected_features["global_top_weights"]
        except:
            return self._select_features(df_splits, selector)
        finally:
            [df_split.unpersist(blocking=True) for df_split in df_splits]
            self.spark.sparkContext._jvm.System.gc()

    def _save_training_stats(
        self,
        split_id: int,
        start_time: float,
        end_time: float,
        num_instances: int,
        selected_features: List[str],
        selected_weights: List[float]
    ) -> None:
        stats = {
            "split_id": split_id,
            "timestamp": datetime.now().isoformat(),
            "training_time": end_time - start_time,
            "num_instances": num_instances,
            "num_selected_features": len(selected_features),
            "selected_features": selected_features,
            "feature_weights": selected_weights
        }

        directory = os.path.dirname(self.fs_stats_path)
        os.makedirs(directory, exist_ok=True)

        # Atomic write
        temp_path = f"{self.fs_stats_path}.tmp"
        try:
            with open(temp_path, 'w') as f:
                json.dump(stats, f)
            if os.path.exists(self.fs_stats_path):
                os.remove(self.fs_stats_path)
            os.rename(temp_path, self.fs_stats_path)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    def _select_features(
        self,
        df_splits: List[DataFrame],
        selector: ReliefFSelector
    ) -> Tuple[List[str], np.ndarray]:
        # Load previous checkpoint if exists
        last_split, top_features_per_split = load_checkpoint()
        if last_split >= 0:
            print(bcolors.OKBLUE + f"Resuming from split {last_split + 1}" + bcolors.ENDC)
        else:
            top_features_per_split = []

        total_time = 0.0
        try:
            num_splits = len(df_splits)
            for i, df_split in enumerate(df_splits):
                if i <= last_split:
                    continue

                current_split = i
                start_time = time.time()
                total_instances = df_split.count()
                print(bcolors.HEADER + f"Processing split {i + 1}/{num_splits} with {total_instances} instances" + bcolors.ENDC)

                model = selector.fit(df_split)
                weights = model.weights
                n_select = int(len(weights) * selector.getSelectionThreshold())
                selected_indices = np.argsort(weights)[-n_select:]

                selected_features = [self.feature_cols[idx] for idx in selected_indices]
                selected_weights = weights[selected_indices].tolist()

                end_time = time.time()
                # Save training stats for this split
                self._save_training_stats(
                    split_id=i,
                    start_time=start_time,
                    end_time=end_time,
                    num_instances=total_instances,
                    selected_features=selected_features,
                    selected_weights=selected_weights
                )

                top_features_per_split.append((i, selected_features, selected_weights))

                print(bcolors.OKGREEN + f"Split {i + 1}: Top {len(selected_features)} features: {selected_features}" + bcolors.ENDC)

                # ... timing and cleanup code ...
                elapsed_time = time.time() - start_time
                print(bcolors.OKCYAN + f"Elapsed time for split {i + 1}: {elapsed_time:.4f} seconds" + bcolors.ENDC)
                save_checkpoint(i, top_features_per_split)

                df_split.unpersist(blocking=True)
                self.spark.sparkContext._jvm.System.gc()

                total_time += elapsed_time

            # clear_checkpoint()

        except Exception as e:
            print(bcolors.FAIL + f"Error occurred: {str(e)}" + bcolors.ENDC)
            save_checkpoint(current_split, top_features_per_split)
            raise e

        # Process results
        global_weights = np.zeros(len(self.feature_cols))
        for _, features, split_weights in top_features_per_split:
            for feature, weight in zip(features, split_weights):
                idx = self.feature_cols.index(feature)
                global_weights[idx] += weight
        global_weights /= num_splits

        n_global_select = int(len(self.feature_cols) * selector.getSelectionThreshold())
        global_top_indices = np.argsort(global_weights)[-n_global_select:]
        global_top_features = [self.feature_cols[idx] for idx in global_top_indices]
        global_top_weights = global_weights[global_top_indices]

        # Save results
        selected_features = {
            "global_top_features": global_top_features,
            "global_top_weights": global_top_weights.tolist()
        }
        with open(self.important_features_path, "w") as f:
            json.dump(selected_features, f)

        return global_top_features, global_top_weights
