import time
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from typing import Optional, Literal

from logger import logger
from .base_model import BaseModel


class GradientBoostingModel(BaseModel):
    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 10,
        random_state: int = 42,
        subsample: float = 0.8,
        max_features: float | Literal['sqrt', 'log2'] | None = 'sqrt',
    ):
        super().__init__(f"gradient_boosting_{n_estimators}_{learning_rate}_{max_depth}")

        # Create base GradientBoostingClassifier
        base_classifier = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_features=max_features,
            subsample=subsample,
            random_state=random_state,
        )

        # Wrap with OneVsRestClassifier for multi-class
        self.model = OneVsRestClassifier(
            base_classifier,
            n_jobs=4  # Parallel processing
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        logger.info(f"Training Gradient Boosting model (OneVsRest) on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        logger.info(f"Number of unique classes: {len(y_train.unique())}")
        logger.info(f"Base classifier parameters: {self.model.estimator.get_params()}")

        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - start_time

        logger.info(f"Training completed in {self.train_time:.2f} seconds")
        logger.info(f"Average time per classifier: {self.train_time/len(self.model.estimators_):.2f} seconds")
        logger.info(f"Memory usage: {X_train.memory_usage().sum() / 1024**2:.2f} MB")

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        logger.info("Making predictions...")
        start_time = time.time()
        predictions = self.model.predict(X_test)
        self.test_time = time.time() - start_time
        logger.info(f"Predictions completed in {self.test_time:.2f} seconds")
        return predictions
