from typing import Union, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from logger import logger
import os
from feature_selection import FEATURES_DIR


class RFSelector:
    def __init__(self) -> None:
        self.feature_scores: Optional[np.ndarray] = None
        self.selected_features: Optional[np.ndarray] = None
        self.feature_selector: Optional[RandomForestClassifier] = None
        self.features_df: Optional[pd.DataFrame] = None
        self.n_features: Optional[int] = None
        self.output_dir: str = FEATURES_DIR
        self.train_cache = os.path.join(self.output_dir, 'train_selected_features_rf.csv')
        self.test_cache = os.path.join(self.output_dir, 'test_selected_features_rf.csv')

        os.makedirs(self.output_dir, exist_ok=True)

    def has_cached_features(self) -> bool:
        return os.path.exists(self.train_cache) and os.path.exists(self.test_cache)

    def load_cached_features(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not self.has_cached_features():
            raise ValueError("No cached features found.")

        logger.info("Loading cached feature selection results...")
        train_df = pd.read_csv(self.train_cache)
        test_df = pd.read_csv(self.test_cache)

        return train_df, test_df

    def fit(self,
            x_train: Union[pd.DataFrame, np.ndarray],
            y_train: Union[pd.Series, pd.DataFrame, np.ndarray],
            n_features: Union[int, str] = 'all',
            n_estimators: int = 100,
            max_depth: int = 10
    ) -> 'RFSelector':
        """
        Fit Random Forest feature selector
        """
        logger.info("Starting Random Forest feature selection...")

        if isinstance(n_features, int):
            self.n_features = n_features
        else:
            self.n_features = x_train.shape[1]

        # Initialize Random Forest
        self.feature_selector = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42
        )

        # Convert labels if they're in DataFrame
        if isinstance(y_train, pd.DataFrame):
            y_labels = y_train['Label'].values
        else:
            y_labels = y_train

        # Fit Random Forest
        self.feature_selector.fit(
            x_train.values if isinstance(x_train, pd.DataFrame) else x_train,
            y_labels
        )

        # Get feature names
        feature_names = (x_train.columns if isinstance(x_train, pd.DataFrame)
                        else [f'feature_{i}' for i in range(x_train.shape[1])])

        # Create DataFrame with feature importances
        self.features_df = pd.DataFrame({
            'feature': feature_names,
            'score': self.feature_selector.feature_importances_
        })
        self.features_df = self.features_df.sort_values('score', ascending=False)

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform data by selecting top features"""
        # Get selected feature indices
        selected_features = self.get_top_features(n_features=self.n_features)
        feature_mask = np.isin(
            X.columns if isinstance(X, pd.DataFrame) else np.arange(X.shape[1]),
            selected_features
        )

        # Select features
        if isinstance(X, pd.DataFrame):
            return X.loc[:, feature_mask].values
        return X[:, feature_mask]

    def transform_and_save(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        prefix: str = 'train'
    ) -> pd.DataFrame:
        logger.info(f"Transforming {prefix} data...")
        transformed_X = self.transform(X)

        # Get selected feature names
        selected_features = self.get_top_features(n_features=self.n_features)
        logger.info(f"Number of selected features: {len(selected_features)}")

        # Create DataFrame with selected features
        df_transformed = pd.DataFrame(transformed_X, columns=selected_features)

        # Save to CSV
        output_path = os.path.join(self.output_dir, f'{prefix}_selected_features_rf.csv')
        df_transformed.to_csv(output_path, index=False)
        logger.info(f"Saved transformed data to {output_path}")
        logger.info(f"Selected features shape: {df_transformed.shape}")

        return df_transformed

    def plot_feature_scores(self, filename: str = 'feature_scores_rf.png') -> None:
        """Plot feature importance scores"""
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(self.features_df)), self.features_df['score'])
        plt.xticks(range(len(self.features_df)),
                  self.features_df['feature'],
                  rotation=90,
                  fontsize=5)
        plt.title('Random Forest Feature Importance Scores')
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()

    def plot_cumulative_scores(self,
                             filename: str = 'cumulative_scores_rf.png',
                             threshold: float = 0.99) -> None:
        """Plot cumulative importance scores"""
        sorted_importances = self.features_df['score'].values
        sorted_features = self.features_df['feature'].values
        x_values = range(len(sorted_importances))

        plt.figure(figsize=(12, 6))
        cumulative_importances = np.cumsum(sorted_importances)
        plt.plot(x_values, cumulative_importances)

        threshold_value = cumulative_importances[-1] * threshold
        plt.hlines(y=threshold_value,
                  xmin=0,
                  xmax=len(sorted_importances),
                  color='r',
                  linestyles='dashed')

        plt.xticks(x_values, sorted_features, rotation='vertical', fontsize=5)
        plt.yticks([], [])
        plt.xlabel('Feature Variable', fontsize=8)
        plt.title('Cumulative Feature Importance', fontsize=8)
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()

    def get_top_features(self,
                        n_features: Optional[int] = None,
                        threshold: float = 0.99) -> np.ndarray:
        """Get top N features or features that explain threshold% of variance"""
        if n_features is not None:
            return self.features_df['feature'].values[:n_features]

        cumsum = np.cumsum(self.features_df['score'])
        threshold_value = cumsum[-1] * threshold
        n_features = np.where(cumsum >= threshold_value)[0][0] + 1
        return self.features_df['feature'].values[:n_features]
