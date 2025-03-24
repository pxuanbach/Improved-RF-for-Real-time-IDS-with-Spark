import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from logger import logger


class DataNormalizer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.train_df = None
        self.test_df = None
        self.train_labels = None
        self.test_labels = None

    def split_data(self, df: pd.DataFrame, test_size: float = 0.3) -> 'DataNormalizer':
        """Split data into training and testing sets"""
        logger.info("Splitting data into training and testing sets...")

        # 3 Different labeling options
        attacks = ['Label', 'Label_Category', 'Attack']

        # xs=feature vectors, ys=labels
        xs = df.drop(attacks, axis=1)
        ys = df[attacks]

        # split dataset - stratified
        x_train, x_test, y_train, y_test = train_test_split(
            xs, ys, test_size=test_size, random_state=0, stratify=ys['Label']
        )

        # Remove constant columns
        column_names = np.array(list(x_train))
        to_drop = [x for x in column_names
                  if len(x_train.groupby([x]).size().unique()) == 1]

        if to_drop:
            logger.info(f"Dropping {len(to_drop)} constant columns")
            x_train = x_train.drop(to_drop, axis=1)
            x_test = x_test.drop(to_drop, axis=1)

        self.train_df = x_train
        self.test_df = x_test
        self.train_labels = y_train
        self.test_labels = y_test

        logger.info(f"Training set size: {len(x_train)}")
        logger.info(f"Test set size: {len(x_test)}")
        logger.info(f"Remaining features: {x_train.shape[1]}")

        return self

    def normalize_data(self) -> None:
        """Normalize training and test data"""
        if self.train_df is None or self.test_df is None:
            raise ValueError("No data to normalize. Run split_data first.")

        self.train_df = pd.DataFrame(
            self.scaler.fit_transform(self.train_df),
            columns=self.train_df.columns,
            index=self.train_df.index
        )

        self.test_df = pd.DataFrame(
            self.scaler.transform(self.test_df),
            columns=self.test_df.columns,
            index=self.test_df.index
        )

        return self

    def get_train_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.train_df, self.train_labels

    def get_test_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.test_df, self.test_labels

