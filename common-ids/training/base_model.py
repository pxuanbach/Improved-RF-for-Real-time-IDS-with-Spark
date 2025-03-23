from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import joblib
from pathlib import Path

from logger import logger


EVALUATION_DIR = "./data/model_evaluation"
MODELS_DIR = "./data/saved_models"


class BaseModel(ABC):
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.train_time = 0
        self.test_time = 0
        self.metrics = {}
        self.output_dir = EVALUATION_DIR + f"/{name.lower()}"
        self.model_path = Path(MODELS_DIR) / f"{name.lower()}.joblib"
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        pass

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
        """Calculate and store performance metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred,
            average='weighted',
            zero_division=1  # Explicitly set zero_division parameter
        )

        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'train_time': self.train_time,
            'test_time': self.test_time
        }

        logger.info("Model Performance Metrics:")
        for metric, value in self.metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        self.plot_confusion_matrix(y_true, y_pred)
        self.detailed_metrics(y_true, y_pred)

        return self.metrics

    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: pd.Series) -> None:
        """Plot confusion matrix"""
        plt.figure(figsize=(12, 8))
        conf_matrix = confusion_matrix(y_true, y_pred)
        conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

        sns.heatmap(conf_matrix_percent,
                   annot=True,
                   fmt='.2%',
                   cmap='Blues',
                   xticklabels=sorted(pd.unique(y_true)),
                   yticklabels=sorted(pd.unique(y_true)))

        plt.title('Confusion Matrix (%)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        output_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Confusion matrix saved to {output_path}")

    def detailed_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> None:
        """Calculate and log detailed metrics for each attack type"""
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=1)
        df_report = pd.DataFrame(report).transpose()

        output_path = os.path.join(self.output_dir, 'classification_report.csv')
        df_report.to_csv(output_path)

        logger.info("\nDetailed Performance Metrics by Attack Type:")
        for label in sorted(pd.unique(y_true)):
            metrics = df_report.loc[label]
            logger.info(f"\n{label}:")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1-score: {metrics['f1-score']:.4f}")
            logger.info(f"  Support: {metrics['support']:.0f}")

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics

    def save_model(self) -> None:
        """Save trained model to disk"""
        logger.info(f"Saving model to {self.model_path}")
        joblib.dump(self.model, self.model_path)
        logger.info("Model saved successfully")

    def load_model(self) -> None:
        """Load trained model from disk"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"No saved model found at {self.model_path}")

        logger.info(f"Loading model from {self.model_path}")
        self.model = joblib.load(self.model_path)
        logger.info("Model loaded successfully")
