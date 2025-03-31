from typing import Union, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skrebate import ReliefF
from logger import logger
import os
from feature_selection import FEATURES_DIR


class ReliefFSelector:
    def __init__(self) -> None:
        self.feature_scores: Optional[np.ndarray] = None
        self.selected_features: Optional[np.ndarray] = None
        self.feature_selector: Optional[ReliefF] = None
        self.features_df: Optional[pd.DataFrame] = None
        self.n_features: Optional[int] = None
        self.output_dir: str = FEATURES_DIR
        self.train_cache = os.path.join(self.output_dir, 'train_selected_features_relief.csv')
        self.test_cache = os.path.join(self.output_dir, 'test_selected_features_relief.csv')
        self.subset_size = 15000

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

    def create_random_subsets(self, X: Union[pd.DataFrame, np.ndarray],
                            y: Union[pd.Series, np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create random subsets of data for ReliefF"""
        total_samples = len(X)
        subsets = []

        for _ in range(self.n_subsets):
            # Generate random indices without replacement
            indices = np.random.choice(total_samples, size=min(self.subset_size, total_samples),
                                     replace=False)

            # Extract subset using indices
            if isinstance(X, pd.DataFrame):
                X_subset = X.iloc[indices]
            else:
                X_subset = X[indices]

            if isinstance(y, pd.Series):
                y_subset = y.iloc[indices]
            else:
                y_subset = y[indices]

            subsets.append((X_subset, y_subset))

        return subsets

    def fit(self,
            x_train: Union[pd.DataFrame, np.ndarray],
            y_train: Union[pd.Series, pd.DataFrame, np.ndarray],
            n_features: Union[int, str] = 'all',
            n_neighbors: Optional[int] = 100
    ) -> 'ReliefFSelector':
        """Fit ReliefF feature selector using data chunks"""
        logger.info("Starting ReliefF feature selection with data chunks...")

        if isinstance(n_features, int):
            self.n_features = n_features
        else:
            self.n_features = x_train.shape[1]
        self.n_neighbors = n_neighbors

        # Get feature names
        feature_names = (x_train.columns if isinstance(x_train, pd.DataFrame)
                        else [f'feature_{i}' for i in range(x_train.shape[1])])

        # Convert labels if they're in DataFrame
        if isinstance(y_train, pd.DataFrame):
            y_labels = y_train['Label'].values
        else:
            y_labels = y_train

        # Calculate number of chunks
        total_samples = len(x_train)
        n_chunks = total_samples // self.subset_size + (1 if total_samples % self.subset_size != 0 else 0)
        accumulated_scores = np.zeros(x_train.shape[1])

        n_chunks = 5

        # Process each chunk
        for i in range(n_chunks):
            start_idx = i * self.subset_size
            end_idx = min((i + 1) * self.subset_size, total_samples)

            logger.info(f"Processing chunk {i+1}/{n_chunks} (samples {start_idx} to {end_idx})")

            # Extract chunk
            if isinstance(x_train, pd.DataFrame):
                X_chunk = x_train.iloc[start_idx:end_idx]
            else:
                X_chunk = x_train[start_idx:end_idx]

            y_chunk = y_labels[start_idx:end_idx]

            # Initialize and fit ReliefF for this chunk
            chunk_selector = ReliefF(
                n_features_to_select=self.n_features,
                n_neighbors=min(self.n_neighbors, len(X_chunk)-1),  # Adjust n_neighbors if chunk is small
                n_jobs=1,
                verbose=True
            )

            chunk_selector.fit(
                X_chunk.values if isinstance(X_chunk, pd.DataFrame) else X_chunk,
                y_chunk
            )

            # Accumulate feature scores
            accumulated_scores += np.abs(chunk_selector.feature_importances_)

        # Average the scores across all chunks
        final_scores = accumulated_scores / n_chunks

        # Store the last fitted selector for transform operations
        self.feature_selector = chunk_selector

        # Create DataFrame with averaged feature scores
        self.features_df = pd.DataFrame({
            'feature': feature_names,
            'score': final_scores
        })
        self.features_df = self.features_df.sort_values('score', ascending=False)

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform the data by selecting features"""
        return self.feature_selector.transform(X.values if isinstance(X, pd.DataFrame) else X)

    def transform_and_save(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        prefix: str = 'train'
    ) -> np.ndarray:
        """Transform the data and save to CSV with labels

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Feature matrix to transform
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
        output_path = os.path.join(self.output_dir, f'{prefix}_selected_features_relief.csv')
        df_transformed.to_csv(output_path, index=False)
        logger.info(f"Saved transformed data to {output_path}")
        logger.info(f"Selected features shape: {df_transformed.shape}")

        return df_transformed

    def plot_feature_scores(self, filename: str = 'feature_scores_relief.png') -> None:
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
                             filename: str = 'cumulative_scores_relief.png',
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
