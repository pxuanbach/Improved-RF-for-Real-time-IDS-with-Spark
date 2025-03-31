from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def evaluate_model(predictions, label_col="label", prediction_col="prediction", label_to_name=None):
    evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="f1")
    f1_score = evaluator.evaluate(predictions)
    
    y_validate = predictions.select(label_col).toPandas()
    y_predicted = predictions.select(prediction_col).toPandas()
    actual_labels = sorted(y_validate[label_col].unique())
    
    precision, recall, fscore, support = precision_recall_fscore_support(y_validate, y_predicted, labels=actual_labels, zero_division=0)
    original_labels = [label_to_name.get(int(label), "unknown") if label_to_name else str(label) for label in actual_labels]
    
    df_results = pd.DataFrame({
        'attack': actual_labels,
        'original_label': original_labels,
        'precision': precision,
        'recall': recall,
        'fscore': fscore
    })
    
    precision_macro, recall_macro, fscore_macro, _ = precision_recall_fscore_support(
        y_validate, y_predicted, labels=actual_labels, average='macro', zero_division=0
    )
    accuracy = accuracy_score(y_validate, y_predicted)
    
    return f1_score, df_results, precision_macro, recall_macro, fscore_macro, accuracy