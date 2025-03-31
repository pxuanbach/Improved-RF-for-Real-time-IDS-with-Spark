import time
from typing import Dict, Tuple, Any, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
import json
import os
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col
from pathlib import Path
from joblib import dump, load

from modules.utils import bcolors


# Change to local path relative to working directory
MODEL_PATH = "./data/models"

class ModelTrainer:
    def __init__(self, spark: SparkSession, numTrees: int = 20, maxDepth: int = 10) -> None:
        self.spark = spark
        self.numTrees = numTrees
        self.maxDepth = maxDepth

        # Create model directory if it doesn't exist
        os.makedirs(MODEL_PATH, exist_ok=True)

        # Use simple relative paths
        self.metrics_path = os.path.join(MODEL_PATH, "model_metrics.json")
        self.model_joblib_path = os.path.join(MODEL_PATH, "rf_params.joblib")
        self.model_path = os.path.join(MODEL_PATH, "rf_model")

    def train_and_evaluate(
        self,
        df_reduced: DataFrame,
        label_to_name: Dict[int, str],
        global_top_features: List[str]
    ) -> Tuple[RandomForestClassificationModel, Dict[str, Any]]:
        # Create feature vector
        print(bcolors.HEADER + "Creating feature vectors..." + bcolors.ENDC)

        assembler = VectorAssembler(inputCols=global_top_features, outputCol="features")
        df_vector = assembler.transform(df_reduced).select("features", "Label_Category", "label_idx")

        # Train-Test Split
        train_df, test_df = df_vector.randomSplit([0.8, 0.2], seed=42)

        # Check if model exists
        rf_model = None
        if os.path.exists(self.model_path):
            print(bcolors.HEADER + "Loading existing model..." + bcolors.ENDC)
            try:
                rf_model = RandomForestClassificationModel.load(self.model_path)
                print(bcolors.OKGREEN + "✅ Model loaded successfully" + bcolors.ENDC)
            except Exception as e:
                print(f"{bcolors.WARNING}Warning: Could not load existing model: {str(e)}{bcolors.ENDC}")
                rf_model = None

        # Train Random Forest if no existing model or loading failed
        if rf_model is None:
            print(bcolors.HEADER + "Training new Random Forest model" + bcolors.ENDC)
            rf = RandomForestClassifier(
                featuresCol="features",
                labelCol="label_idx",
                numTrees=self.numTrees,
                maxDepth=self.maxDepth,
                seed=42
            )
            rf_model = rf.fit(train_df)

            # Save the model
            print(bcolors.HEADER + "Saving model and metadata" + bcolors.ENDC)
            self._save_model(rf_model, label_to_name)

        print(bcolors.HEADER + "Evaluating model" + bcolors.ENDC)
        predictions = rf_model.transform(test_df)
        metrics = self._evaluate_model(predictions, label_to_name)

        print(bcolors.HEADER + "Saving metrics" + bcolors.ENDC)
        self._save_metrics(metrics)

        return rf_model, metrics

    def _save_model(
        self,
        model: RandomForestClassificationModel,
        label_to_name: Dict[int, str]
    ) -> None:
        try:
            # Save picklable parameters with joblib
            model_params = {
                'hyperparameters': {
                    'numTrees': self.numTrees,
                    'maxDepth': self.maxDepth
                },
                'feature_importances': model.featureImportances.toArray().tolist(),
                'num_classes': model.numClasses,
                'label_mapping': label_to_name,
                'trees_info': [
                    {
                        'depth': tree.depth,
                        'num_nodes': tree.numNodes
                    }
                    for tree in model.trees
                ]
            }

            # Save parameters with joblib
            dump(model_params, self.model_joblib_path, compress=3)
            print(f"{bcolors.OKGREEN}✅ Model parameters saved to {self.model_joblib_path}{bcolors.ENDC}")

        except Exception as e:
            print(f"{bcolors.FAIL}Error saving model: {str(e)}{bcolors.ENDC}")

    def _evaluate_model(
        self,
        predictions: DataFrame,
        label_to_name: Dict[int, str]
    ) -> Dict[str, Any]:
        evaluator = MulticlassClassificationEvaluator(labelCol="label_idx", predictionCol="prediction", metricName="f1")
        f1_score = evaluator.evaluate(predictions)

        y_validate = predictions.select("label_idx").toPandas()
        y_predicted = predictions.select("prediction").toPandas()

        precision, recall, fscore, support = precision_recall_fscore_support(y_validate, y_predicted)
        actual_labels = sorted(y_validate["label_idx"].unique())
        original_labels = [label_to_name[label] for label in actual_labels]

        df_results = pd.DataFrame({
            'attack': actual_labels,
            'original_label': original_labels,
            'precision': precision,
            'recall': recall,
            'fscore': fscore
        })

        precision_macro, recall_macro, fscore_macro, _ = precision_recall_fscore_support(y_validate, y_predicted, average='macro')
        accuracy = accuracy_score(y_validate, y_predicted)

        self._print_results(f1_score, df_results, precision_macro, recall_macro, fscore_macro, accuracy)

        return {
            "f1_score": f1_score,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "fscore_macro": fscore_macro,
            "accuracy": accuracy,
            "detailed_results": df_results.to_dict()
        }

    def _save_metrics(self, metrics: Dict[str, Any]) -> None:
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics, f)
        print(bcolors.OKGREEN + f"✅ Metrics saved to {self.metrics_path}" + bcolors.ENDC)

    def _print_results(
        self,
        f1_score: float,
        df_results: Any,
        precision_macro: float,
        recall_macro: float,
        fscore_macro: float,
        accuracy: float
    ) -> None:
        print(f"\n✅ F1-score: {f1_score:.4f}")
        print(df_results.to_string(index=False))
        print(f"\n✅ Precision (macro): {precision_macro:.4f}")
        print(f"✅ Recall (macro): {recall_macro:.4f}")
        print(f"✅ F1-score (macro): {fscore_macro:.4f}")
        print(f"✅ Accuracy: {accuracy:.4f}")
