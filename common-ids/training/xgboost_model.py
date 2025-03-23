import time
import pandas as pd
from typing import Optional
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from logger import logger
from .base_model import BaseModel


# https://www.dpss.inesc-id.pt/~mpc/pubs/XGBoost_chapter.pdf
class XGBoostModel(BaseModel):
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        random_state: int = 42,
    ):
        # https://www.dpss.inesc-id.pt/~mpc/pubs/XGBoost_chapter.pdf
        super().__init__(f"xgboost_{n_estimators}_{learning_rate}_{max_depth}")
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=4,
            verbosity=1
        )
        self.label_encoder = LabelEncoder()

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        logger.info(f"Training XGBoost model on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        logger.info(f"Model parameters: {self.model.get_params()}")

        # Encode string labels to numbers
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        start_time = time.time()
        self.model.fit(X_train, y_train_encoded)
        self.train_time = time.time() - start_time

        logger.info(f"Training completed in {self.train_time:.2f} seconds")
        logger.info(f"Memory usage: {X_train.memory_usage().sum() / 1024**2:.2f} MB")

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        logger.info("Making predictions...")
        start_time = time.time()
        predictions = self.model.predict(X_test)
        self.test_time = time.time() - start_time
        logger.info(f"Predictions completed in {self.test_time:.2f} seconds")
        # Convert back to original labels
        return self.label_encoder.inverse_transform(predictions)
