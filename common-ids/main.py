import os
import time
import pandas as pd
import enum

from logger import initialize_logger, logger
from training.random_forest import RandomForestModel
from training.logistic_regression import LogisticRegressionModel
from training.xgboost_model import XGBoostModel


class TrainType(str, enum.Enum):
    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"
    XGBOOST = "xgboost"


train_type = TrainType.XGBOOST
os.makedirs("./log/", exist_ok=True)
initialize_logger(log_file="./log/" + train_type.value + ".log", log_level=20)


if __name__ == "__main__":
    from preprocessing import DataPreprocessor
    from normalization import DataNormalizer
    from feature_selection import FeatureSelector

    #region PREPROCESSING
    start_time = time.time()
    preprocessor = (
        DataPreprocessor(use_cache=True)
        .load_dataset()
        .preprocess_data()
        .cache()
    )
    processed_df = preprocessor.get_df()
    end_time = time.time()
    logger.warning(f"Preprocessing time: {(end_time - start_time):.2f} seconds")
    #endregion


    #region NORMALIZATION & SPLITTING
    start_time = time.time()
    normalizer = DataNormalizer()
    normalizer.split_data(processed_df).normalize_data()
    train_df, train_labels = normalizer.get_train_data()
    test_df, test_labels = normalizer.get_test_data()
    end_time = time.time()
    logger.warning(f"Normalization time: {(end_time - start_time):.2f} seconds")
    #endregion


    #region FEATURE SELECTION
    start_time = time.time()
    selector = FeatureSelector()

    if selector.has_cached_features():
        # Load cached features
        train_df_selected, test_df_selected = selector.load_cached_features()
    else:
        # Perform feature selection
        selector.fit(train_df, train_labels, n_features=40)

        # Generate plots
        selector.plot_feature_scores()
        selector.plot_cumulative_scores()

        # Transform and save data
        train_df_selected = selector.transform_and_save(train_df, processed_df, 'train')
        test_df_selected = selector.transform_and_save(test_df, processed_df, 'test')

    end_time = time.time()
    logger.warning(f"Feature selection time: {(end_time - start_time):.2f} seconds")
    #endregion


    #region MODEL TRAINING
    if train_type == TrainType.RANDOM_FOREST:
        model = RandomForestModel(n_estimators=200, max_depth=15, max_features=None)
    elif train_type == TrainType.LOGISTIC_REGRESSION:
        model = LogisticRegressionModel(max_iter=15000, C=100, solver="sag")
    elif train_type == TrainType.XGBOOST:
        model = XGBoostModel(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=20,
        )
    else:
        raise ValueError("Invalid train type")

    start_time = time.time()
    model.train(train_df_selected, train_labels.Label)
    end_time = time.time()
    model.save_model()
    logger.warning(f"{model.name} training time: {(end_time - start_time):.2f} seconds")
    #endregion

    #region MODEL EVALUATION
    start_time = time.time()
    model.load_model()
    predictions = model.predict(test_df_selected)
    metrics = model.evaluate(test_labels.Label, predictions)
    end_time = time.time()
    logger.warning(f"{model.name} evaluation time: {(end_time - start_time):.2f} seconds")
    #endregion
