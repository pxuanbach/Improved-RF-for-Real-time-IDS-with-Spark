import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split

from logger import logger


CACHE_DIR = './data/cache'

class DataPreprocessor:
    def __init__(self, path_dir: str = '../dataset/CICIDS2017', use_cache: bool = False):
        self.path_dir = path_dir
        self.all_files = glob.glob(self.path_dir + "/*.csv")
        self.df = None
        self.df_copy = None
        self.use_cache = use_cache
        self.cache_file = os.path.join(CACHE_DIR, 'preprocessed_data.csv')
        os.makedirs(CACHE_DIR, exist_ok=True)

    def log_column_info(self, column: str = 'Label') -> None:
        if column in self.df.columns:
            label_dist = self.df[column].value_counts()
            logger.info(f"Column '{column}' value distribution:")
            for label, count in label_dist.items():
                percentage = (count / len(self.df)) * 100
                logger.info(f"  {label}: {count} ({percentage:.2f}%)")

    def load_dataset(self):
        if self.use_cache and os.path.exists(self.cache_file):
            logger.info("Loading preprocessed data from cache...")
            self.df = pd.read_csv(self.cache_file)
            return self

        logger.info("Loading dataset...")
        self.df = pd.concat((pd.read_csv(f) for f in self.all_files))

        logger.info(f"Total number of records: {len(self.df)}")
        logger.info(f"Number of features: {len(self.df.columns)}")

        self.log_column_info()
        return self

    def _clean_column_names(self) -> None:
        """Clean column names by removing leading/trailing whitespace"""
        # Get columns with leading whitespace
        whitespace_cols = [col for col in self.df.columns if str(col).startswith(' ')]
        if whitespace_cols:
            logger.info("Found columns with leading whitespace:")
            for col in whitespace_cols:
                cleaned_name = col.strip()
                logger.info(f"  Renaming '{col}' to '{cleaned_name}'")
                self.df = self.df.rename(columns={col: cleaned_name})

    def preprocess_data(self):
        if self.df is None or self.df.empty:
            raise ValueError('No data found. Please load the dataset first.')

        if self.use_cache and os.path.exists(self.cache_file):
            return self

        logger.info("Preprocessing data...")
        initial_size = len(self.df)

        # Clean column names first
        self._clean_column_names()

        if self.df.isnull().any().any():
            self.df = self.df.replace([np.inf, -np.inf], np.nan)
            self.df = self.df.dropna()

        # rename columns
        self.df.loc[self.df.Label == 'Web Attack � Brute Force', ['Label']] = 'Brute Force'
        self.df.loc[self.df.Label == 'Web Attack � XSS', ['Label']] = 'XSS'

        # Create attack column, containing binary labels
        self.df['Attack'] = np.where(self.df['Label'] == 'BENIGN', 0, 1)
        self.log_column_info('Attack')

        self.df = self.df.replace(['Heartbleed', 'Web Attack � Sql Injection', 'Infiltration'], np.nan)
        self.df = self.df.dropna()

        # Proposed Groupings
        attack_group = {'BENIGN': 'benign',
                'DoS Hulk': 'dos',
                'PortScan': 'probe',
                'DDoS': 'ddos',
                'DoS GoldenEye': 'dos',
                'FTP-Patator': 'brute_force',
                'SSH-Patator': 'brute_force',
                'DoS slowloris': 'dos',
                'DoS Slowhttptest': 'dos',
                'Bot': 'botnet',
                'Brute Force': 'web_attack',
                'XSS': 'web_attack'}
        # Create grouped label column
        self.df['Label_Category'] = self.df['Label'].map(lambda x: attack_group[x])
        self.log_column_info('Label_Category')

        # Log changes after preprocessing
        logger.info(f"Records removed during preprocessing: {initial_size - len(self.df)}")
        return self

    def cache(self):
        if os.path.exists(self.cache_file):
            logger.info("Cache file already exists.")
            return self

        logger.info("Caching preprocessed data...")
        self.df.to_csv(self.cache_file, index=False)
        logger.info(f"Cached data saved to {self.cache_file}")
        return self

    def get_df(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("No data available. Run load_dataset first.")
        return self.df

    def get_df_copy(self):
        return self.df_copy
