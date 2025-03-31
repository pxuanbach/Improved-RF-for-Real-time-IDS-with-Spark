from typing import Union, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from skrebate import ReliefF
from sklearn.ensemble import RandomForestClassifier
from logger import logger
import os
from feature_selection import FEATURES_DIR


class FeatureSelector:
    def __init__(self) -> None:
        self.feature_scores: Optional[np.ndarray] = None
        self.selected_features: Optional[np.ndarray] = None
        self.feature_selector: Optional[SelectKBest] = None
        self.features_df: Optional[pd.DataFrame] = None
        self.n_features: Optional[int] = None
        self.output_dir: str = FEATURES_DIR
        self.train_cache = os.path.join(self.output_dir, 'train_selected_features.csv')
        self.test_cache = os.path.join(self.output_dir, 'test_selected_features.csv')

        os.makedirs(self.output_dir, exist_ok=True)

    def has_cached_features(self) -> bool:
        """Check if feature selection results are already cached"""
        return os.path.exists(self.train_cache) and os.path.exists(self.test_cache)

    def load_cached_features(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load cached feature selection results"""
        if not self.has_cached_features():
            raise ValueError("No cached features found.")

        logger.info("Loading cached feature selection results...")
        train_df = pd.read_csv(self.train_cache)
        test_df = pd.read_csv(self.test_cache)

        return train_df, test_df

    def fit(self,
            x_train: Union[pd.DataFrame, np.ndarray],
            y_train: Union[pd.Series, pd.DataFrame, np.ndarray],
            n_features: Union[int, str] = 'all') -> 'FeatureSelector':
        """
        Fit the feature selector to the training data

        Parameters:
        -----------
        x_train : Union[pd.DataFrame, np.ndarray]
            Training feature matrix
        y_train : Union[pd.Series, pd.DataFrame, np.ndarray]
            Training labels
        n_features : Union[int, str], default='all'
            Number of features to select. If 'all', uses threshold-based selection.
        """
        logger.info("Starting feature selection...")

        # Store n_features for later use
        if isinstance(n_features, int):
            self.n_features = n_features
            k = n_features
        else:  # n_features == 'all'
            k = x_train.shape[1]  # Initially select all features

        self.feature_selector = SelectKBest(score_func=chi2, k=k)

        # Handle different types of y_train
        if isinstance(y_train, pd.DataFrame):
            y_labels = y_train['Label']
        else:
            y_labels = y_train

        self.feature_selector.fit(x_train, y_labels)

        # Create DataFrame with feature scores
        feature_names = (x_train.columns if isinstance(x_train, pd.DataFrame)
                        else [f'feature_{i}' for i in range(x_train.shape[1])])

        self.features_df = pd.DataFrame({
            'feature': feature_names,
            'score': self.feature_selector.scores_
        })
        self.features_df = self.features_df.sort_values('score', ascending=False)

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform the data by selecting features"""
        return self.feature_selector.transform(X)

    def transform_and_save(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        df: pd.DataFrame,
        prefix: str = 'train'
    ) -> np.ndarray:
        """Transform the data and save to CSV with labels

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Feature matrix to transform
        df : pd.DataFrame
            Original processed dataframe containing all columns including labels
        prefix : str, default='train'
            Prefix for output filename (e.g., 'train' or 'test')

        Returns
        -------
        np.ndarray
            Transformed feature matrix
        """
        logger.info(f"Transforming {prefix} data...")
        transformed_X = self.transform(X)

        # Get selected feature names
        selected_features = self.get_top_features(n_features=self.n_features)
        logger.info(f"Number of selected features: {len(selected_features)}")

        # Create DataFrame with selected features without index
        df_transformed = pd.DataFrame(transformed_X, columns=selected_features)

        # Save to CSV
        output_path = os.path.join(self.output_dir, f'{prefix}_selected_features.csv')
        df_transformed.to_csv(output_path, index=False)
        logger.info(f"Saved transformed data to {output_path}")
        logger.info(f"Selected features shape: {df_transformed.shape}")

        return transformed_X

    def plot_feature_scores(self, filename: str = 'feature_scores.png') -> None:
        """Plot individual feature scores"""
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(self.features_df)), self.features_df['score'])
        plt.xticks(range(len(self.features_df)),
                  self.features_df['feature'],
                  rotation=90,
                  fontsize=5)
        plt.title('Feature Importance Scores')
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()

    def plot_cumulative_scores(self,
                             filename: str = 'cumulative_scores.png',
                             threshold: float = 0.99) -> None:
        """Plot cumulative feature scores"""
        sorted_importances = self.features_df['score'].values
        sorted_features = self.features_df['feature'].values
        x_values = range(len(sorted_importances))

        plt.figure(figsize=(12, 6))
        cumulative_importances = np.cumsum(sorted_importances)
        plt.plot(x_values, cumulative_importances)

        # Draw threshold line
        threshold_value = cumulative_importances[-1] * threshold
        plt.hlines(y=threshold_value,
                  xmin=0,
                  xmax=len(sorted_importances),
                  color='r',
                  linestyles='dashed')

        plt.xticks(x_values, sorted_features, rotation='vertical', fontsize=5)
        plt.yticks([], [])
        plt.xlabel('Feature Variable', fontsize=8)
        plt.title('Cumulative Feature Scores', fontsize=8)
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
