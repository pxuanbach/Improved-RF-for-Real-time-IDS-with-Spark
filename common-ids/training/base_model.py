from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from logger import logger

class BaseModel(ABC):
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.train_time = 0
        self.test_time = 0
        self.metrics = {}
        self.output_dir = f"./data/model_evaluation/{name.lower()}"
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
            zero_division=0  # Explicitly set zero_division parameter
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
        report = classification_report(y_true, y_pred, output_dict=True)
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
