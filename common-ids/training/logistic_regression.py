import time
from typing import Literal
import pandas as pd
from sklearn.linear_model import LogisticRegression

from logger import logger
from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    def __init__(
        self,
        max_iter: int = 10000,
        C: float = 0.1,
        solver: Literal['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'] = "saga",
        random_state: int = 42
    ):
        super().__init__(f"logistic_regression_{max_iter}_{C}_{solver}")
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            n_jobs=4,
            random_state=random_state,
            solver=solver,
            verbose=1
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        logger.info(f"Training Logistic Regression model on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        logger.info(f"Model parameters: {self.model.get_params()}")

        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - start_time

        logger.info(f"Training completed in {self.train_time:.2f} seconds")
        logger.info(f"Memory usage: {X_train.memory_usage().sum() / 1024**2:.2f} MB")

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        logger.info("Making predictions...")
        start_time = time.time()
        predictions = self.model.predict(X_test)
        self.test_time = time.time() - start_time
        logger.info(f"Predictions completed in {self.test_time:.2f} seconds")
        return predictions

