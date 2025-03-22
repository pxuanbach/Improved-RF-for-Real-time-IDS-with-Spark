from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, lit, isnan, isnull
from pyspark.ml.feature import StringIndexer, VectorAssembler

def preprocess_data(df: DataFrame) -> DataFrame:
    df = df.withColumnRenamed(' Label', 'Label')
    df = df.replace(['Heartbleed', 'Web Attack � Sql Injection', 'Infiltration'], None, subset=['Label'])
    df = df.dropna(how='any')
    df = df.withColumn('Label', when(col('Label') == 'Web Attack � Brute Force', 'Brute Force').otherwise(col('Label')))
    df = df.withColumn('Label', when(col('Label') == 'Web Attack � XSS', 'XSS').otherwise(col('Label')))
    df = df.withColumn('Attack', when(col('Label') == 'BENIGN', 0).otherwise(1))
    
    attack_group = {
        'BENIGN': 'benign', 'DoS Hulk': 'dos', 'PortScan': 'probe', 'DDoS': 'ddos',
        'DoS GoldenEye': 'dos', 'FTP-Patator': 'brute_force', 'SSH-Patator': 'brute_force',
        'DoS slowloris': 'dos', 'DoS Slowhttptest': 'dos', 'Bot': 'botnet',
        'Brute Force': 'web_attack', 'XSS': 'web_attack'
    }
    
    conditions = [when(col('Label') == k, lit(v)) for k, v in attack_group.items()]
    df = df.withColumn('Label_Category', conditions[0])
    for condition in conditions[1:]:
        df = df.withColumn('Label_Category', when(col('Label_Category').isNull(), condition).otherwise(col('Label_Category')))
    
    return df

def handle_nan_infinity(df: DataFrame, feature_cols: list) -> DataFrame:
    for col_name in feature_cols:
        count_nan = df.filter(isnan(col(col_name)) | isnull(col(col_name))).count()
        count_inf = df.filter(col(col_name) == float("inf")).count()
        
        if count_nan > 0 or count_inf > 0:
            print(f"⚠️ Cột {col_name} có {count_nan} NaN và {count_inf} Infinity!")
        
        df = df.withColumn(col_name, when(col(col_name) == float("inf"), None).otherwise(col(col_name)))
        median_value = df.approxQuantile(col_name, [0.5], 0.25)[0] or 0.0
        df = df.fillna({col_name: median_value})
    
    return df

def create_label_index(df: DataFrame, input_col="Label_Category", output_col="label"):
    indexer = StringIndexer(inputCol=input_col, outputCol=output_col)
    model = indexer.fit(df)
    df = model.transform(df)
    labels = model.labels
    label_to_name = {index: label for index, label in enumerate(labels)}
    return df, label_to_name, labels

def reduce_dimensions(df: DataFrame, feature_cols: list, exclude_cols=['Label', 'Label_Category', 'Attack']) -> DataFrame:
    columns_to_keep = feature_cols + exclude_cols
    return df.select(columns_to_keep)

def create_feature_vector(df: DataFrame, feature_cols: list, output_col="features") -> DataFrame:
    assembler = VectorAssembler(inputCols=feature_cols, outputCol=output_col)
    return assembler.transform(df)

import json

def load_metadata(spark, features_path="s3a://mybucket/models/global_top_features", 
                  labels_path="s3a://mybucket/models/label_to_name"):
    # Load global_top_features
    global_top_features_json = spark.read.text(features_path).collect()[0]["value"]
    global_top_features = json.loads(global_top_features_json)
    
    # Load label_to_name
    label_to_name_json = spark.read.text(labels_path).collect()[0]["value"]
    label_to_name = {int(k): v for k, v in json.loads(label_to_name_json).items()}
    
    return global_top_features, label_to_name