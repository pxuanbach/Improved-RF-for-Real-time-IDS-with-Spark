import time
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any

from logger import logger
from .base_model import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        super().__init__("RandomForest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,  # Allow full depth
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=random_state,
            n_jobs=4,
            verbose=1
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        logger.info(f"Training Random Forest model on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        logger.info(f"Model parameters: {self.model.get_params()}")

        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - start_time

        logger.info(f"Training completed in {self.train_time:.2f} seconds")
        logger.info(f"Average time per tree: {self.train_time/self.model.n_estimators:.2f} seconds")
        logger.info(f"Memory usage: {X_train.memory_usage().sum() / 1024**2:.2f} MB")

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        logger.info("Making predictions...")
        start_time = time.time()
        predictions = self.model.predict(X_test)
        self.test_time = time.time() - start_time
        logger.info(f"Predictions completed in {self.test_time:.2f} seconds")
        return predictions
